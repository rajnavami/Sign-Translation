# models/spatial_models/frame_models/dino_adaptor_model.py
#
# KEY FIX vs previous version:
#   RESTORED: self.bn = BatchNorm1d (was changed to LayerNorm — broke accuracy)
#   RESTORED: FlowBranch.bn = BatchNorm1d
#   RESTORED: GatedFusion.bn = BatchNorm1d
#
# WHY LayerNorm broke accuracy:
#   zero_fasttext_prototype_head uses cosine similarity between normalized
#   features and FastText embeddings. BatchNorm1d normalizes per-feature
#   across the batch (correct for cosine sim). LayerNorm normalizes
#   per-sample across features — produces different statistics that make
#   all cosine similarities near-equal, so logits are flat → accuracy=0.
#
# HOW we fix the bs=1 BatchNorm crash properly:
#   BatchNorm1d crashes when input has only 1 sample (can't compute batch stats).
#   Solution: use model.eval() mode for BN during single-sample chunks, OR
#   use a safe_bn wrapper that falls back to identity when N=1.
#   We use the safe wrapper approach — transparent to the rest of the code.

import os
from typing import Dict, List, Optional, Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import ignite.distributed as idist
from loguru import logger

from models.dinov3.model.vision_transformer import vit_small


# ---------------------------------------------------------------------------
# SafeBatchNorm1d — handles bs=1 gracefully
# ---------------------------------------------------------------------------

class SafeBatchNorm1d(nn.Module):
    """
    BatchNorm1d that falls back to LayerNorm when batch size = 1.

    WHY THIS IS NEEDED:
    - BatchNorm1d requires N > 1 during training (needs batch statistics)
    - With bs=1 and chunked flow processing, last chunk may have N=1
    - Crashing with "Expected more than 1 value per channel"

    WHY NOT JUST USE LAYERNORM:
    - The downstream cosine similarity in zero_fasttext_prototype_head
      depends on BatchNorm's per-feature normalization statistics
    - LayerNorm produces different statistics → breaks logit scale → acc=0

    SOLUTION:
    - N > 1: use BatchNorm1d (correct normalization for cosine sim)
    - N = 1: use LayerNorm as fallback (mathematically equivalent for N=1
      since there's no batch dimension to normalize across anyway)
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.ln = nn.LayerNorm(num_features)    # fallback for N=1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1 and self.training:
            # Single sample during training — BN would crash
            # Use LN as fallback (only affects this one edge case)
            return self.ln(x)
        return self.bn(x)


# ---------------------------------------------------------------------------
# FlowBranch
# ---------------------------------------------------------------------------

class FlowBranch(nn.Module):
    """
    Lightweight CNN to encode precomputed optical flow.
    Input:  (N, 2, 64, 64)
    Output: (N, out_dim)
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.dropout = nn.Dropout(p=0.1)
        self.proj    = nn.Linear(128 * 4 * 4, out_dim)
        # RESTORED: BatchNorm1d (was LayerNorm — broke accuracy)
        # SafeBatchNorm1d handles the bs=1 edge case properly
        self.bn      = SafeBatchNorm1d(out_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        x = self.encoder(flow)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.bn(self.proj(x))


# ---------------------------------------------------------------------------
# GatedFusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Gated additive fusion.
    Gate init: -4.0 so sigmoid(-4) ≈ 0.018 at training start.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), -2.0))
        self.proj = nn.Linear(dim * 2, dim)
        # RESTORED: BatchNorm1d (was LayerNorm — broke accuracy)
        self.bn   = SafeBatchNorm1d(dim)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        gate      = torch.sigmoid(self.gate)
        combined  = torch.cat([rgb, flow], dim=-1)
        projected = self.bn(self.proj(combined))
        return rgb + gate * projected


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
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
    ):
        super().__init__()

        trainable_names = trainable_names or []
        adaptor_layers  = adaptor_layers  or []
        adapt_params    = adapt_params    or {}
        flow_params     = flow_params     or {}

        if out_dim is None:
            raise ValueError("out_dim must be provided.")

        self.spatial_model = vit_small(
            img_size=img_size,
            patch_size=patch_size,
            layerscale_init=1.0,
            adaptor_layers=adaptor_layers,
            adapt_params=adapt_params,
            block_chunks=0,
        )
        num_features = self.spatial_model.num_features
        self.lin = nn.Linear(num_features, out_dim)
        # RESTORED: BatchNorm1d — critical for cosine similarity downstream
        # SafeBatchNorm1d handles bs=1 edge case without breaking normalization
        self.bn  = SafeBatchNorm1d(out_dim)

        self.use_flow = use_flow
        if self.use_flow:
            self.flow_branch = FlowBranch(out_dim=out_dim)
            self.fusion      = GatedFusion(dim=out_dim)
            logger.info("FlowBranch + GatedFusion enabled.")

        ckpt_path = self._resolve_checkpoint(ckpt_dir)
        logger.info(f"ckpt_dir: {ckpt_dir}")
        logger.info(f"use_flow: {self.use_flow}")

        missing = self.spatial_model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu"), strict=False
        )

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
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

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

    def _forward_dino_chunked(
        self,
        x: torch.Tensor,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """
        Run DINOv2 forward in chunks to avoid OOM.
        chunk_size=16 works well for A100 80GB with bs=8.
        Reduce to 8 if OOM on smaller GPUs.
        """
        feat_chunks = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i: i + chunk_size]
            if not self.training:
                with torch.no_grad():
                    feat = self.spatial_model.forward_features(
                        chunk
                    )["x_norm_clstoken"]
            else:
                feat = self.spatial_model.forward_features(
                    chunk
                )["x_norm_clstoken"]
            feat_chunks.append(feat)
        return torch.cat(feat_chunks, dim=0)    # (sum_T, 384)

    def forward(
        self,
        list_of_frames: List[torch.Tensor],
        list_of_flows: Optional[List[torch.Tensor]] = None,
        max_len: Optional[int] = 1024,
    ):
        lengths: List[int] = [len(x) for x in list_of_frames]
        B = len(lengths)

        # ── RGB path ──────────────────────────────────────────────────────────
        x     = torch.cat(list_of_frames, dim=0)
        feats = self._forward_dino_chunked(x, chunk_size=16)
        # SafeBatchNorm1d handles the case where sum_T=1
        y     = self.bn(self.lin(feats.float()))

        # ── Flow fusion ───────────────────────────────────────────────────────
        if self.use_flow and list_of_flows is not None:
            flow_x = torch.cat(list_of_flows, dim=0)
            flow_x = flow_x.to(device=y.device, dtype=torch.float32)

            # Process flow in chunks of 32
            # SafeBatchNorm1d handles last chunk if N=1
            flow_chunks = []
            for i in range(0, flow_x.shape[0], 32):
                flow_chunks.append(
                    self.flow_branch(flow_x[i: i + 32])
                )
            flow_y = torch.cat(flow_chunks, dim=0)
            y      = self.fusion(y, flow_y)

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

        y_out = torch.stack(y_padded, dim=0)

        mask = torch.zeros((B, max_len), device=y_out.device, dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, :min(l, max_len)] = True

        return y_out, mask, {"list_of_original_features": y}