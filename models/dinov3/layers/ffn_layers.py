# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional

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
        adapt_params={},
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

        self.adapt = adapt
        if self.adapt:
            if not adapt_params["w_lora_ff"]:
                self.adapt = False
            else:
                lora_rank = adapt_params["lora_rank"]
                lora_a = adapt_params["lora_a"]
                lora_drop = adapt_params["lora_drop"]
                self.lora_w1_l1 = nn.Linear(in_features, lora_rank, bias=False, device=device)
                self.lora_w1_l2 = nn.Linear(lora_rank, hidden_features, bias=False, device=device)
                self.lora_w2_l1 = nn.Linear(hidden_features, lora_rank, bias=False, device=device)
                self.lora_w2_l2 = nn.Linear(lora_rank, out_features, bias=False, device=device)
                self.lora_scaling = lora_a / lora_rank
                nn.init.normal_(self.lora_w1_l1.weight.data, 0, std=0.02)
                nn.init.normal_(self.lora_w2_l1.weight.data, 0, std=0.02)
                nn.init.constant_(self.lora_w1_l2.weight.data, 0)
                nn.init.constant_(self.lora_w2_l2.weight.data, 0)
                self.lora_drop = nn.Dropout(lora_drop)

                if "fixed_adapt_style" in adapt_params:
                    self.fixed_adapt_style = adapt_params["fixed_adapt_style"]
                else:
                    self.fixed_adapt_style = False

    def forward(self, x: Tensor) -> Tensor:
        if self.adapt:
            if not self.fixed_adapt_style:
                x = self.fc1(x) + (
                    self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling
                )
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x) + (
                    self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(x))) * self.lora_scaling
                )
                x = self.drop(x)
                return x
            else:
                lora_x = (
                    self.lora_w1_l2(self.lora_w1_l1(self.lora_drop(x))) * self.lora_scaling
                )
                lora_x = (
                    self.lora_w2_l2(self.lora_w2_l1(self.lora_drop(lora_x))) * self.lora_scaling
                )
                
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = x + lora_x
                x = self.drop(x)
                return x
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
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
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
