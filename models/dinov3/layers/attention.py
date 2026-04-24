# Copyright (c) Meta Platforms, Inc. and affiliates.
# DINOv3 SelfAttention + DINOv2-style LoRA (correct, list-compatible)

import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.utils import cat_keep_shapes, uncat_with_shapes


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    """
    DINOv3 SelfAttention + LoRA (DINOv2 semantics).

    Fixes included:
    - safe adapt_params default (None -> {})
    - forward_list supports rope_list=None
    - LoRA applied to QKV using the ORIGINAL input x (not post-projection q/k/v)
    - consistent SDPA dropout in both tensor and list paths
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
        adapt: bool = False,
        adapt_params: Optional[dict] = None,  # FIX: avoid mutable default {}
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

        adapt_params = adapt_params or {}  # FIX
        self.adapt = bool(adapt) and bool(adapt_params.get("w_lora", False))  # FIX: .get
        self.new_lora = False  # FIX: always defined

        if self.adapt:
            lora_rank = adapt_params["lora_rank"]
            lora_a = adapt_params["lora_a"]
            lora_drop = adapt_params["lora_drop"]
            self.lora_scaling = lora_a / lora_rank
            self.lora_drop = nn.Dropout(lora_drop)

            if "new" not in adapt_params:
                # Old-style: single low-rank update for combined QKV and output
                self.lora_w1_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                self.lora_w1_l2 = nn.Linear(lora_rank, dim * 3, bias=False, device=device)
                self.lora_w2_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                self.lora_w2_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)

                nn.init.normal_(self.lora_w1_l1.weight.data, 0, std=0.02)
                nn.init.normal_(self.lora_w2_l1.weight.data, 0, std=0.02)
                nn.init.constant_(self.lora_w1_l2.weight.data, 0)
                nn.init.constant_(self.lora_w2_l2.weight.data, 0)

                self.new_lora = False
            else:
                # New-style: separate Q/K/V updates + output update
                self.lora_wq_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                self.lora_wq_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)
                self.lora_wk_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                self.lora_wk_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)
                self.lora_wv_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                self.lora_wv_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)
                self.lora_wo_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                self.lora_wo_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)

                nn.init.normal_(self.lora_wq_l1.weight.data, 0, std=0.02)
                nn.init.normal_(self.lora_wk_l1.weight.data, 0, std=0.02)
                nn.init.normal_(self.lora_wv_l1.weight.data, 0, std=0.02)
                nn.init.normal_(self.lora_wo_l1.weight.data, 0, std=0.02)
                nn.init.constant_(self.lora_wq_l2.weight.data, 0)
                nn.init.constant_(self.lora_wk_l2.weight.data, 0)
                nn.init.constant_(self.lora_wv_l2.weight.data, 0)
                nn.init.constant_(self.lora_wo_l2.weight.data, 0)

                self.new_lora = True

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype

        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)

        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0

        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)

        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)

        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    # ---- LoRA helpers (keeps forward/forward_list consistent) ----
    def _apply_lora_to_qkv(self, x: Tensor, qkv_base: Tensor) -> Tensor:
        if not self.adapt:
            return qkv_base
        if not self.new_lora:
            return qkv_base + (self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling)
        qlora = self.lora_wq_l2(self.lora_wq_l1(self.lora_drop(x))) * self.lora_scaling
        klora = self.lora_wk_l2(self.lora_wk_l1(self.lora_drop(x))) * self.lora_scaling
        vlora = self.lora_wv_l2(self.lora_wv_l1(self.lora_drop(x))) * self.lora_scaling
        return qkv_base + torch.cat([qlora, klora, vlora], dim=-1)

    def _apply_lora_to_out(self, attn_out: Tensor, proj_base: Tensor) -> Tensor:
        if not self.adapt:
            return proj_base
        if not self.new_lora:
            return proj_base + (self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(attn_out))) * self.lora_scaling)
        return proj_base + (self.lora_wo_l2(self.lora_wo_l1(self.lora_drop(attn_out))) * self.lora_scaling)

    def compute_attention(self, qkv: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        dropout_p = self.attn_drop.p if self.training else 0.0  # FIX: consistent dropout
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(B, N, C)
        return x

    def forward(self, x: Tensor, attn_bias=None, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        assert attn_bias is None
        qkv = self._apply_lora_to_qkv(x, self.qkv(x))
        attn_out = self.compute_attention(qkv=qkv, rope=rope)
        out = self._apply_lora_to_out(attn_out, self.proj(attn_out))
        out = self.proj_drop(out)
        return out

    def forward_list(self, x_list: List[Tensor], attn_bias=None, rope_list=None) -> List[Tensor]:
        assert attn_bias is None

        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)

        # FIX: rope_list may be None (drop-path branch)
        if rope_list is None:
            rope_list = [None] * len(x_list)

        qkv_flat = self._apply_lora_to_qkv(x_flat, self.qkv(x_flat))
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)

        att_out_list: List[Tensor] = []
        for qkv, rope in zip(qkv_list, rope_list):
            att_out_list.append(self.compute_attention(qkv=qkv, rope=rope))

        att_flat, shapes2, num_tokens2 = cat_keep_shapes(att_out_list)
        proj_flat = self._apply_lora_to_out(att_flat, self.proj(att_flat))
        proj_flat = self.proj_drop(proj_flat)
        return uncat_with_shapes(proj_flat, shapes2, num_tokens2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0.0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
