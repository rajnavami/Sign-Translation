# models/spatial_models/frame_models/dino_adaptor_model.py

import os
from typing import Dict, List, Optional, Any

import torch
from torch import nn
import ignite.distributed as idist
from loguru import logger

from models.dinov3.model.vision_transformer import vit_small


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
        self.dropout = nn.Dropout(p=0.3)
        self.proj    = nn.Linear(128 * 4 * 4, out_dim)
        self.bn      = nn.LayerNorm(out_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        x = self.encoder(flow)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.bn(self.proj(x))


class GatedFusion(nn.Module):
    """
    Gated additive fusion.
    Gate init: -4.0 so sigmoid(-4) ≈ 0.018 at training start.
    Flow contribution grows as gate learns.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), -4.0))
        self.proj = nn.Linear(dim * 2, dim)
        self.bn   = nn.LayerNorm(dim)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        gate      = torch.sigmoid(self.gate)
        combined  = torch.cat([rgb, flow], dim=-1)
        projected = self.bn(self.proj(combined))
        return rgb + gate * projected


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
        self.bn  = nn.LayerNorm(out_dim)

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

        WHY CHUNKING IS NEEDED:
        - x shape is (sum_T, 3, H, W) where sum_T = batch_size * frames_per_video
        - With bs=4 and ~69 frames per video: sum_T = 4 * 69 = 276 frames
        - DINOv2 attention is O(N^2) in sequence length
        - Processing all 276 frames at once = huge intermediate tensors = OOM
        - chunk_size=16 means we process 16 frames at a time instead
        - This reduces peak memory by ~17x at the cost of slightly more compute

        WHY chunk_size=16:
        - Each frame at 224x224 with patch_size=14 gives (224/14)^2 = 256 patches
        - 16 frames * 256 patches = 4096 tokens per chunk — fits in ~24GB VRAM
        - Reduce to 8 if still OOM, increase to 32 if you have headroom

        GRADIENT HANDLING:
        - During training (self.training=True): gradients flow through chunks
          normally. Each chunk.backward() accumulates gradients correctly.
        - During eval (self.training=False): torch.no_grad() saves memory.
        """
        feat_chunks = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i: i + chunk_size]            # (chunk_size, 3, H, W)
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
        return torch.cat(feat_chunks, dim=0)        # (sum_T, 384)

    def forward(
        self,
        list_of_frames: List[torch.Tensor],
        list_of_flows: Optional[List[torch.Tensor]] = None,
        max_len: Optional[int] = 1024,
    ):
        """
        Args:
            list_of_frames: list[B] of (Ti, 3, H, W)
            list_of_flows:  list[B] of (Ti, 2, 64, 64) — precomputed flow
                            None = skip flow branch
            max_len:        pad/truncate to this length
        """
        lengths: List[int] = [len(x) for x in list_of_frames]
        B = len(lengths)

        # ── RGB path: chunked to avoid OOM ───────────────────────────────────
        x     = torch.cat(list_of_frames, dim=0)    # (sum_T, 3, H, W)
        feats = self._forward_dino_chunked(x, chunk_size=16)  # (sum_T, 384)
        y     = self.bn(self.lin(feats.float()))     # (sum_T, out_dim)

        # ── Flow fusion: also chunked ─────────────────────────────────────────
        if self.use_flow and list_of_flows is not None:
            flow_x = torch.cat(list_of_flows, dim=0)          # (sum_T, 2, 64, 64)
            flow_x = flow_x.to(device=y.device, dtype=torch.float32)

            # Chunk the flow branch too — keeps memory flat
            flow_chunks = []
            for i in range(0, flow_x.shape[0], 32):
                flow_chunks.append(
                    self.flow_branch(flow_x[i: i + 32])
                )
            flow_y = torch.cat(flow_chunks, dim=0)             # (sum_T, out_dim)
            y      = self.fusion(y, flow_y)                    # (sum_T, out_dim)

        # ── Pad / truncate to max_len ─────────────────────────────────────────
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

        y_out = torch.stack(y_padded, dim=0)        # (B, max_len, out_dim)

        mask = torch.zeros((B, max_len), device=y_out.device, dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, :min(l, max_len)] = True

        return y_out, mask, {"list_of_original_features": y}