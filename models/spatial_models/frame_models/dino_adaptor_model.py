# sign2gpt/Sign2GPT/models/spatial_models/frame_models/dino_adaptor_model.py

import os
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import nn
import ignite.distributed as idist
from loguru import logger

from models.dinov3.model.vision_transformer import vit_small
from models.RAFT.core.raft import RAFT

# Add RAFT imports for optical flow
try:
    from torchvision.models.optical_flow import raft_small
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    print("Torchvision RAFT not available. Install torchvision>=0.13.0")


class Model(nn.Module):
    """
    DINOv3 frame encoder wrapper used by Sign2GPT.

    Updates vs your previous version:
    - FIX: no mutable default args (trainable_names/adaptor_layers/adapt_params)
    - FIX: robust checkpoint loading (local path or URL) with distributed barrier
    - FIX: correct bf16 casting (param.data = param.data.to(...))
    - IMPROVE: safer slicing/padding by using python lengths list + cumulative offsets
    - IMPROVE: cast backbone output to fp32 before Linear+BatchNorm (bn is often unstable in bf16)
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
        flow_ckpt_dir: Optional[str] = None,
        flow_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        trainable_names = trainable_names or []
        adaptor_layers = adaptor_layers or []
        adapt_params = adapt_params or {}
        flow_params = flow_params or {}

        if out_dim is None:
            raise ValueError("out_dim must be provided (e.g., dim_model=512).")

        # Build DINOv3 ViT-S (embed_dim=384, depth=12, heads=6)
        # NOTE: any extra kwargs not used by DinoVisionTransformer should be ignored by its **ignored_kwargs.
        self.spatial_model = vit_small(
            img_size=img_size,
            patch_size=patch_size,
            layerscale_init=1.0,
            adaptor_layers=adaptor_layers,
            adapt_params=adapt_params,
            block_chunks=0,  # harmless if ignored by your local dinov3; kept for backward compat
        )

        num_features = self.spatial_model.num_features
        self.lin = nn.Linear(num_features, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

        # Optical flow setup
        self.use_flow = use_flow
        if self.use_flow:
            if not RAFT_AVAILABLE:
                raise ImportError("Torchvision RAFT not available. Install torchvision>=0.13.0")
            self.flow_model = raft_small(pretrained=True)
            self.flow_model.eval()
            # Freeze flow model
            for param in self.flow_model.parameters():
                param.requires_grad = False
            # Separate DINO for flow features
            self.flow_dino = vit_small(
                img_size=img_size,
                patch_size=patch_size,
                layerscale_init=1.0,
                adaptor_layers=adaptor_layers,
                adapt_params=adapt_params,
                block_chunks=0,
            )
            # Load same checkpoint for flow DINO
            ckpt_path = self._resolve_checkpoint(ckpt_dir)
            self.flow_dino.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
            # Freeze flow DINO
            for param in self.flow_dino.parameters():
                param.requires_grad = False
            # Fusion layer: concat RGB + Flow features
            self.fusion_lin = nn.Linear(num_features * 2, out_dim)
            self.fusion_bn = nn.BatchNorm1d(out_dim)

        logger.info(f"ckpt_dir: {ckpt_dir}")
        logger.info(f"use_flow: {self.use_flow}")

        ckpt_path = self._resolve_checkpoint(ckpt_dir)

        # Load pretrained weights (strict=False so LoRA params remain missing_keys)
        missing = self.spatial_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

        # Freeze all backbone params except:
        # - newly introduced params (missing_keys) => LoRA modules, etc.
        # - anything in trainable_names prefixes (optional)
        for name, param in self.spatial_model.named_parameters():
            if name in missing.missing_keys:
                param.requires_grad = True
            elif any(name.startswith(prefix) for prefix in trainable_names):
                param.requires_grad = True
            else:
                param.requires_grad = False
                # Cast frozen params to bf16 (correctly) to save memory if supported
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and param.is_floating_point():
                    param.data = param.data.to(dtype=torch.bfloat16)

        if freeze:
            for _, param in self.named_parameters():
                param.requires_grad = False

        # Log trainables
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable params: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.4f}%)")

    def _resolve_checkpoint(self, ckpt_dir: str) -> str:
        """
        If ckpt_dir is a URL, download once on rank0 and barrier.
        If ckpt_dir is a local path, return as-is.
        """
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

            if idist.get_world_size() > 0:
                idist.barrier()

    def forward(self, list_of_frames: List[torch.Tensor], max_len: Optional[int] = 1024):
        """
        list_of_frames: list length B, each element is a tensor [Ti, 3, H, W]
        returns:
          y:    [B, max_len, out_dim]
          mask: [B, max_len] bool (True for valid frames)
        """
        # Python lengths list for safe slicing
        lengths: List[int] = [len(x_i) for x_i in list_of_frames]
        B = len(lengths)

        # Flatten frames to one big batch: [sum(Ti), 3, H, W]
        x = torch.cat(list_of_frames, dim=0)

        if self.use_flow and lengths[0] > 1:  # Need at least 2 frames for flow
            # Compute optical flow between consecutive frames
            flow_images = []
            for seq in list_of_frames:
                if len(seq) < 2:
                    # If only one frame, use zero flow
                    flow_img = torch.zeros_like(seq[0:1, :2, :, :])  # [1, 2, H, W]
                    flow_images.append(flow_img.repeat(len(seq), 1, 1, 1))
                else:
                    seq_flows = []
                    for i in range(len(seq) - 1):
                        img1 = seq[i]  # [3, H, W]
                        img2 = seq[i+1]  # [3, H, W]
                        with torch.no_grad():
                            predictions = self.flow_model(img1, img2)
                            flow = predictions[-1].unsqueeze(0)  # [1, 2, H, W]
                        seq_flows.append(flow.squeeze(0))
                    # For the last frame, use the previous flow
                    seq_flows.append(seq_flows[-1] if seq_flows else torch.zeros_like(seq[0][:2]))
                    flow_tensor = torch.stack(seq_flows, dim=0)  # [Ti, 2, H, W]
                    flow_images.append(flow_tensor)
            flow_x = torch.cat(flow_images, dim=0)  # [sum(Ti), 2, H, W]
            # Convert flow to 3-channel by duplicating or using magnitude
            flow_x = torch.cat([flow_x, flow_x.mean(dim=1, keepdim=True)], dim=1)  # [sum(Ti), 3, H, W]

            # Get flow features
            flow_feats = self.flow_dino.forward_features(flow_x)["x_norm_clstoken"]  # [sum(Ti), 384]
            flow_y = self.bn(self.lin(flow_feats.float()))  # [sum(Ti), out_dim]

            # Get RGB features
            rgb_feats = self.spatial_model.forward_features(x)["x_norm_clstoken"]  # [sum(Ti), 384]
            rgb_y = self.bn(self.lin(rgb_feats.float()))  # [sum(Ti), out_dim]

            # Fuse: concat and project
            fused_feats = torch.cat([rgb_y, flow_y], dim=-1)  # [sum(Ti), out_dim * 2]
            y = self.fusion_bn(self.fusion_lin(fused_feats))  # [sum(Ti), out_dim]
        else:
            # Original RGB-only path
            feats = self.spatial_model.forward_features(x)["x_norm_clstoken"]  # [sum(Ti), 384]
            y = self.bn(self.lin(feats.float()))  # [sum(Ti), out_dim]

        if max_len is None:
            max_len = max(lengths)

        # Build per-sample padded sequences
        offsets = [0]
        for l in lengths:
            offsets.append(offsets[-1] + l)

        y_padded = []
        for i in range(B):
            yi = y[offsets[i] : offsets[i + 1]]  # [Ti, out_dim]
            y_padded.append(self.pad(yi, max_len))

        y_out = torch.stack(y_padded, dim=0)  # [B, max_len, out_dim]

        # Mask
        mask = torch.zeros((B, max_len), device=y_out.device, dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, :l] = True

        return y_out, mask, {"list_of_original_features": y}

