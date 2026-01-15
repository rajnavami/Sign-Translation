# sign2gpt/Sign2GPT/models/spatial_models/frame_models/dino_adaptor_model.py

import os
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import nn
import ignite.distributed as idist
from loguru import logger

from models.dinov3.model.vision_transformer import vit_small


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
        img_size: int = 518,
        patch_size: int = 16,
    ):
        super().__init__()

        trainable_names = trainable_names or []
        adaptor_layers = adaptor_layers or []
        adapt_params = adapt_params or {}

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

        logger.info(f"ckpt_dir: {ckpt_dir}")

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

            return local_path

        # Local path
        if not os.path.isfile(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
        return ckpt_dir

    @staticmethod
    def pad(tensor: torch.Tensor, length: int) -> torch.Tensor:
        """
        tensor: [T, C]
        returns: [length, C] (zero padded)
        """
        if tensor.size(0) == length:
            return tensor
        if tensor.size(0) > length:
            return tensor[:length]
        return torch.cat(
            [tensor, tensor.new_zeros((length - tensor.size(0),) + tensor.size()[1:])],
            dim=0,
        )

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

        # DINOv3 features
        feats = self.spatial_model.forward_features(x)["x_norm_clstoken"]  # [sum(Ti), 384]
        list_of_original_features = feats

        # Safer for BN: cast to fp32 before linear+bn
        feats = feats.float()
        y = self.bn(self.lin(feats))  # [sum(Ti), out_dim]

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

        return y_out, mask, {"list_of_original_features": list_of_original_features}
