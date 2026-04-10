# models/spatial_models/frame_models/dino_adaptor_model.py
#
# CHANGES vs current version:
#   REMOVED: runtime RAFT, flow_dino, fusion_lin, fusion_bn
#   REMOVED: flow_ckpt_dir parameter
#   ADDED:   FlowBranch CNN (lightweight, with dropout=0.3)
#   ADDED:   GatedFusion with gate initialised to -4.0 (near-zero at start)
#   CHANGED: forward() accepts list_of_flows as optional argument

import os
from typing import Dict, List, Optional, Any

import torch
from torch import nn
import ignite.distributed as idist
from loguru import logger

from models.dinov3.model.vision_transformer import vit_small


# ---------------------------------------------------------------------------
# FlowBranch
# ---------------------------------------------------------------------------

class FlowBranch(nn.Module):
    """
    Lightweight CNN to encode precomputed optical flow.

    Input:  (N, 2, 64, 64)  — precomputed dx/dy channels from LMDB
    Output: (N, out_dim)

    dropout=0.3 added because Phoenix14T is small (~7k videos).
    Without it, the flow branch overfits easily.
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16x16 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.dropout = nn.Dropout(p=0.3)
        self.proj    = nn.Linear(128 * 4 * 4, out_dim)
        self.bn      = nn.BatchNorm1d(out_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        x = self.encoder(flow)       # (N, 128, 4, 4)
        x = x.flatten(1)             # (N, 2048)
        x = self.dropout(x)
        return self.bn(self.proj(x)) # (N, out_dim)


# ---------------------------------------------------------------------------
# GatedFusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Gated additive fusion — mirrors Sign2GPT's zero-gated cross-attention.

    Gate init: -4.0 so sigmoid(-4) ≈ 0.018
    → flow contributes ~2% at training start
    → gate opens gradually as flow proves useful

    THIS IS WHAT check_gate.py MONITORS:
      - gate_mean ~0.018  = flow not active yet (normal at epoch 0-2)
      - gate_mean > 0.05  = flow contributing (good signal)
      - gate_mean > 0.3   = flow strongly used (very good)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), -4.0))
        self.proj = nn.Linear(dim * 2, dim)
        self.bn   = nn.BatchNorm1d(dim)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        gate      = torch.sigmoid(self.gate)          # (dim,) in (0, 1)
        combined  = torch.cat([rgb, flow], dim=-1)    # (N, dim*2)
        projected = self.bn(self.proj(combined))      # (N, dim)
        return rgb + gate * projected                 # (N, dim)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    DINOv2 ViT-S frame encoder with optional precomputed flow fusion.

    Flow is NEVER computed here at runtime.
    It arrives as a precomputed tensor from the dataloader via list_of_flows.
    If list_of_flows is None or use_flow=False, runs as pure RGB baseline.
    """

    def __init__(
        self,
        ckpt_dir: str,
        trainable_names: Optional[List[str]] = None,
        adaptor_layers: Optional[List[int]] = None,
        adapt_params: Optional[Dict[str, Any]] = None,
        out_dim: Optional[int] = None,
        freeze: bool = False,
        img_size: int = 512,
        patch_size: int = 16,
        use_flow: bool = False,
        flow_params: Optional[Dict[str, Any]] = None,
        # NOTE: flow_ckpt_dir removed — precomputed flow needs no checkpoint
    ):
        super().__init__()

        trainable_names = trainable_names or []
        adaptor_layers  = adaptor_layers  or []
        adapt_params    = adapt_params    or {}
        flow_params     = flow_params     or {}

        if out_dim is None:
            raise ValueError("out_dim must be provided.")

        # ── DINOv2 ViT-S backbone ─────────────────────────────────────────────
        self.spatial_model = vit_small(
            img_size=img_size,
            patch_size=patch_size,
            layerscale_init=1.0,
            adaptor_layers=adaptor_layers,
            adapt_params=adapt_params,
            block_chunks=0,
        )
        num_features = self.spatial_model.num_features   # 384 for ViT-S
        self.lin = nn.Linear(num_features, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)

        # ── Optional flow branch ──────────────────────────────────────────────
        self.use_flow = use_flow
        if self.use_flow:
            self.flow_branch = FlowBranch(out_dim=out_dim)
            self.fusion      = GatedFusion(dim=out_dim)
            logger.info(
                "FlowBranch + GatedFusion enabled. "
                "Flow loaded from precomputed LMDB."
            )

        # ── Load DINOv2 checkpoint ────────────────────────────────────────────
        ckpt_path = self._resolve_checkpoint(ckpt_dir)
        logger.info(f"ckpt_dir: {ckpt_dir}")
        logger.info(f"use_flow: {self.use_flow}")

        missing = self.spatial_model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu"), strict=False
        )

        # ── Freeze strategy ───────────────────────────────────────────────────
        # Trainable: LoRA params (missing_keys) + anything in trainable_names
        # Frozen: everything else in the DINOv2 backbone
        # lin, bn, flow_branch, fusion are always trainable
        for name, param in self.spatial_model.named_parameters():
            if name in missing.missing_keys:
                param.requires_grad = True
            elif any(name.startswith(p) for p in trainable_names):
                param.requires_grad = True
            else:
                param.requires_grad = False

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.4f}%)"
        )

    def _resolve_checkpoint(self, ckpt_dir: str) -> str:
        if ckpt_dir.startswith("http://") or ckpt_dir.startswith("https://"):
            import requests
            local_path = "/tmp/dinov3_pretrain.pth"
            if idist.get_world_size() == 0 or idist.get_local_rank() == 0:
                if not os.path.isfile(local_path):
                    logger.info(f"Downloading checkpoint to {local_path} ...")
                    r = requests.get(ckpt_dir, timeout=120)
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(r.content)
            if idist.get_world_size() > 1:
                idist.barrier()
            return local_path
        return ckpt_dir

    def forward(
        self,
        list_of_frames: List[torch.Tensor],
        list_of_flows: Optional[List[torch.Tensor]] = None,
        max_len: Optional[int] = 1024,
    ):
        """
        Args:
            list_of_frames: list[B] of (Ti, 3, H, W)   — RGB frames
            list_of_flows:  list[B] of (Ti, 2, 64, 64) — precomputed flow
                            None = skip flow branch entirely
            max_len:        pad/truncate to this temporal length

        Returns:
            y_out:  (B, max_len, out_dim)
            mask:   (B, max_len) bool — True = real frame, False = padding
            extras: dict
        """
        lengths: List[int] = [len(x) for x in list_of_frames]
        B = len(lengths)

        # ── RGB path ──────────────────────────────────────────────────────────
        x     = torch.cat(list_of_frames, dim=0)              # (sum_T, 3, H, W)
        feats = self.spatial_model.forward_features(x)["x_norm_clstoken"]
        y     = self.bn(self.lin(feats.float()))               # (sum_T, out_dim)

        # ── Flow fusion ───────────────────────────────────────────────────────
        if self.use_flow and list_of_flows is not None:
            flow_x = torch.cat(list_of_flows, dim=0)          # (sum_T, 2, 64, 64)
            flow_x = flow_x.to(device=y.device, dtype=torch.float32)
            flow_y = self.flow_branch(flow_x)                  # (sum_T, out_dim)
            y      = self.fusion(y, flow_y)                    # (sum_T, out_dim)

        # ── Pad / truncate ────────────────────────────────────────────────────
        if max_len is None:
            max_len = max(lengths)

        offsets = [0]
        for l in lengths:
            offsets.append(offsets[-1] + l)

        y_padded = []
        for i in range(B):
            yi = y[offsets[i]: offsets[i + 1]]
            if yi.size(0) < max_len:
                pad = torch.zeros(
                    max_len - yi.size(0), yi.size(1),
                    device=yi.device, dtype=yi.dtype
                )
                yi = torch.cat([yi, pad], dim=0)
            elif yi.size(0) > max_len:
                yi = yi[:max_len]
            y_padded.append(yi)

        y_out = torch.stack(y_padded, dim=0)                   # (B, max_len, out_dim)

        mask = torch.zeros((B, max_len), device=y_out.device, dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, :min(l, max_len)] = True

        return y_out, mask, {"list_of_original_features": y}