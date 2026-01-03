# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from ..utils.utils import cat_keep_shapes, uncat_with_shapes
from torch import Tensor, nn


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
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
        adapt_params={},
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

        self.adapt = adapt
        if self.adapt:
            if not adapt_params["w_lora"]:
                self.adapt = False
            else:
                lora_rank = adapt_params["lora_rank"]
                lora_a = adapt_params["lora_a"]
                lora_drop = adapt_params["lora_drop"]

                if "new" not in adapt_params:
                    self.lora_w1_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                    self.lora_w1_l2 = nn.Linear(lora_rank, dim * 3, bias=False, device=device)
                    self.lora_w2_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                    self.lora_w2_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)
                    self.lora_scaling = lora_a / lora_rank
                    nn.init.normal_(self.lora_w1_l1.weight.data, 0, std=0.02)
                    nn.init.normal_(self.lora_w2_l1.weight.data, 0, std=0.02)
                    nn.init.constant_(self.lora_w1_l2.weight.data, 0)
                    nn.init.constant_(self.lora_w2_l2.weight.data, 0)
                    self.new_lora = False
                else:
                    self.lora_wq_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                    self.lora_wq_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)
                    self.lora_wk_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                    self.lora_wk_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)
                    self.lora_wv_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                    self.lora_wv_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)
                    self.lora_wo_l1 = nn.Linear(dim, lora_rank, bias=False, device=device)
                    self.lora_wo_l2 = nn.Linear(lora_rank, dim, bias=False, device=device)

                    self.lora_scaling = lora_a / lora_rank
                    nn.init.normal_(self.lora_wq_l1.weight.data, 0, std=0.02)
                    nn.init.normal_(self.lora_wk_l1.weight.data, 0, std=0.02)
                    nn.init.normal_(self.lora_wv_l1.weight.data, 0, std=0.02)
                    nn.init.normal_(self.lora_wo_l1.weight.data, 0, std=0.02)
                    nn.init.constant_(self.lora_wq_l2.weight.data, 0)
                    nn.init.constant_(self.lora_wk_l2.weight.data, 0)
                    nn.init.constant_(self.lora_wv_l2.weight.data, 0)
                    nn.init.constant_(self.lora_wo_l2.weight.data, 0)
                    self.new_lora = True
                self.lora_drop = nn.Dropout(lora_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
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
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        B, N, C = x.shape
        
        # DEBUG: Log once per training to verify LoRA is active
        # if not hasattr(self, '_logged_adapt_status'):
        #     from loguru import logger
        #     logger.info(f"🔍 SelfAttention.forward: adapt={self.adapt}, new_lora={getattr(self, 'new_lora', 'N/A')}")
        #     if self.adapt:
        #         logger.info(f"   LoRA scaling: {self.lora_scaling}")
        #     self._logged_adapt_status = True
        
        # Match DINOv2's exact structure - apply LoRA and reshape in one flow
        if self.adapt:
            if not self.new_lora:
                # Old-style LoRA: apply to combined QKV, reshape immediately like DINOv2
                qkv = (
                    self.qkv(x) + (
                        self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling
                    )
                ).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                q, k, v = torch.unbind(qkv, 2)  # [B, N, num_heads, C//num_heads]
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  # [B, num_heads, N, C//num_heads]
            else:
                # New-style LoRA: separate Q, K, V like DINOv2
                qkv_base = self.qkv(x)
                qlora = self.lora_wq_l2(self.lora_wq_l1(self.lora_drop(x))) * self.lora_scaling
                klora = self.lora_wk_l2(self.lora_wk_l1(self.lora_drop(x))) * self.lora_scaling
                vlora = self.lora_wv_l2(self.lora_wv_l1(self.lora_drop(x))) * self.lora_scaling
                qkv = (qkv_base + torch.cat([qlora, klora, vlora], dim=-1)).reshape(
                    B, N, 3, self.num_heads, C // self.num_heads
                )
                q, k, v = torch.unbind(qkv, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        
        # Apply RoPE if provided (DINOv3-specific)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        
        # Use PyTorch's optimized attention (equivalent to DINOv2's manual implementation)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        
        # Apply projection with LoRA
        if self.adapt:
            if not self.new_lora:
                x = self.proj(attn_out) + (
                    self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(attn_out))) * self.lora_scaling
                )
            else:
                x = self.proj(attn_out) + (
                    self.lora_wo_l2(self.lora_wo_l1(self.lora_drop(attn_out))) * self.lora_scaling
                )
        else:
            x = self.proj(attn_out)
        
        x = self.proj_drop(x)
        return x

    # def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
    #     assert len(x_list) == len(rope_list)  # should be enforced by the Block
    #     x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
    #     qkv_flat = self.qkv(x_flat)
    #     qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
    #     att_out = []
    #     for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
    #         att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
    #     x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
    #     x_flat = self.proj(x_flat)
    #     return uncat_with_shapes(x_flat, shapes, num_tokens)
        
    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        
        # Apply QKV with LoRA (matching forward() logic)
        if self.adapt and not self.new_lora:
            # Old-style LoRA on QKV
            qkv_flat = self.qkv(x_flat) + (
                self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x_flat))) * self.lora_scaling
            )
        elif self.adapt and self.new_lora:
            # New-style LoRA: full QKV, LoRA applied in compute_attention
            qkv_flat = self.qkv(x_flat)
        else:
            qkv_flat = self.qkv(x_flat)
        
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        
        # Apply projection with LoRA (matching forward() logic)
        if self.adapt:
            if not self.new_lora:
                x_flat = self.proj(x_flat) + (
                    self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(x_flat))) * self.lora_scaling
                )
            else:
                x_flat = self.proj(x_flat) + (
                    self.lora_wo_l2(self.lora_wo_l1(self.lora_drop(x_flat))) * self.lora_scaling
                )
        else:
            x_flat = self.proj(x_flat)
        
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        if self.adapt and self.new_lora:
            # New style LoRA: apply to Q, K, V separately
            qkv_orig = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv_orig, 2)
            
            # Apply LoRA to Q, K, V
            q_flat = q.reshape(B, N, C)
            k_flat = k.reshape(B, N, C)
            v_flat = v.reshape(B, N, C)
            
            q_lora = self.lora_wq_l2(self.lora_wq_l1(self.lora_drop(q_flat))) * self.lora_scaling
            k_lora = self.lora_wk_l2(self.lora_wk_l1(self.lora_drop(k_flat))) * self.lora_scaling
            v_lora = self.lora_wv_l2(self.lora_wv_l1(self.lora_drop(v_flat))) * self.lora_scaling
            
            q = (q_flat + q_lora).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = (k_flat + k_lora).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = (v_flat + v_lora).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        else:
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


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
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
