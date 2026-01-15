# Copyright (c) Meta Platforms, Inc. and affiliates.
# DINOv3 FFN layers + DINOv2-style LoRA, keeps forward_list

from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.utils import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    """
    DINOv3 Mlp + LoRA like DINOv2 Mlp.

    Uses:
      - adapt_params["w_lora_ff"] to enable FFN LoRA
      - LoRA on fc1 and fc2
      - optional adapt_params["fixed_adapt_style"]
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
        adapt: bool = False,
        adapt_params: Optional[dict] = None,  # FIX: avoid mutable default {}
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

        adapt_params = adapt_params or {}  # FIX
        self.adapt = bool(adapt) and bool(adapt_params.get("w_lora_ff", False))  # FIX: .get

        if self.adapt:
            lora_rank = adapt_params["lora_rank"]
            lora_a = adapt_params["lora_a"]
            lora_drop = adapt_params["lora_drop"]

            self.lora_scaling = lora_a / lora_rank
            self.lora_drop = nn.Dropout(lora_drop)

            self.lora_w1_l1 = nn.Linear(in_features, lora_rank, bias=False, device=device)
            self.lora_w1_l2 = nn.Linear(lora_rank, hidden_features, bias=False, device=device)
            self.lora_w2_l1 = nn.Linear(hidden_features, lora_rank, bias=False, device=device)
            self.lora_w2_l2 = nn.Linear(lora_rank, out_features, bias=False, device=device)

            nn.init.normal_(self.lora_w1_l1.weight.data, 0, std=0.02)
            nn.init.normal_(self.lora_w2_l1.weight.data, 0, std=0.02)
            nn.init.constant_(self.lora_w1_l2.weight.data, 0)
            nn.init.constant_(self.lora_w2_l2.weight.data, 0)

            self.fixed_adapt_style = bool(adapt_params.get("fixed_adapt_style", False))
        else:
            self.fixed_adapt_style = False

    def forward(self, x: Tensor) -> Tensor:
        if self.adapt:
            if not self.fixed_adapt_style:
                x = self.fc1(x) + (self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x) + (self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(x))) * self.lora_scaling)
                x = self.drop(x)
                return x
            else:
                lora_x = (self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling)
                lora_x = (self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(lora_x))) * self.lora_scaling)

                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = x + lora_x
                x = self.drop(x)
                return x

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    """
    Optional: LoRA support if you use ffn_layer="swiglu".
    Uses the SAME enable flag: adapt_params["w_lora_ff"].
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
        adapt: bool = False,
        adapt_params: Optional[dict] = None,  # FIX
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)

        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)

        adapt_params = adapt_params or {}
        self.adapt = bool(adapt) and bool(adapt_params.get("w_lora_ff", False))

        if self.adapt:
            r = adapt_params["lora_rank"]
            alpha = adapt_params["lora_a"]
            lora_drop = adapt_params["lora_drop"]
            self.lora_scaling = alpha / r
            self.lora_drop = nn.Dropout(lora_drop)

            self.lora_w1_l1 = nn.Linear(in_features, r, bias=False, device=device)
            self.lora_w1_l2 = nn.Linear(r, swiglu_hidden_features, bias=False, device=device)

            self.lora_w2_l1 = nn.Linear(in_features, r, bias=False, device=device)
            self.lora_w2_l2 = nn.Linear(r, swiglu_hidden_features, bias=False, device=device)

            self.lora_w3_l1 = nn.Linear(swiglu_hidden_features, r, bias=False, device=device)
            self.lora_w3_l2 = nn.Linear(r, out_features, bias=False, device=device)

            nn.init.normal_(self.lora_w1_l1.weight.data, 0, std=0.02)
            nn.init.normal_(self.lora_w2_l1.weight.data, 0, std=0.02)
            nn.init.normal_(self.lora_w3_l1.weight.data, 0, std=0.02)
            nn.init.constant_(self.lora_w1_l2.weight.data, 0)
            nn.init.constant_(self.lora_w2_l2.weight.data, 0)
            nn.init.constant_(self.lora_w3_l2.weight.data, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.adapt:
            x1 = self.w1(x) + (self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling)
            x2 = self.w2(x) + (self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(x))) * self.lora_scaling)
            hidden = F.silu(x1) * x2
            out = self.w3(hidden) + (self.lora_w3_l2(self.lora_w3_l1(self.lora_drop(hidden))) * self.lora_scaling)
            return out

        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
