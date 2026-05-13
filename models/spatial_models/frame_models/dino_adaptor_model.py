import torch
from torch import nn
from models.dinov2.model.vision_transformer import *
import ignite.distributed as idist


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
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.ln = nn.LayerNorm(num_features, eps, elementwise_affine=affine)

    def forward(self, x):
        if x.size(0) == 1:
            # Use LayerNorm for single-sample batches
            return self.ln(x)
        else:
            # Use BatchNorm for multi-sample batches
            return self.bn(x)


# ---------------------------------------------------------------------------
# FlowBranch — processes precomputed optical flow
# ---------------------------------------------------------------------------

class FlowBranch(nn.Module):
    """
    Processes precomputed optical flow embeddings.

    Input:  [total_frames, H, W, 2] optical flow (dx, dy per pixel)
    Output: [total_frames, D] flow features
    """

    def __init__(self, in_channels=2, hidden_dim=384, out_dim=384):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pool over H,W
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.bn = SafeBatchNorm1d(out_dim)
        self.act = nn.GELU()

    def forward(self, flow):
        # flow: [total_frames, H, W, 2] → [total_frames, 2, H, W]
        flow = flow.permute(0, 3, 1, 2)  # [total_frames, 2, H, W]

        flow_features = []
        for i in range(flow.size(0)):
            f = flow[i:i+1]  # [1, 2, H, W]
            f = self.act(self.conv1(f))
            f = self.act(self.conv2(f))
            f = self.act(self.conv3(f))
            f = self.pool(f).squeeze(-1).squeeze(-1)  # [1, hidden]
            flow_features.append(f)

        flow_features = torch.cat(flow_features, dim=0)  # [total_frames, hidden]
        flow_features = self.act(self.bn(self.lin(flow_features)))  # [total_frames, out_dim]
        return flow_features


# ---------------------------------------------------------------------------
# GatedFusion — combines RGB and flow features
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Gated fusion of RGB and flow features.

    Uses a gate to weight RGB vs flow contributions.
    """

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.act = nn.Sigmoid()

    def forward(self, rgb_feat, flow_feat):
        # rgb_feat, flow_feat: [B, D]
        combined = torch.cat([rgb_feat, flow_feat], dim=-1)  # [B, 2*D]
        gate = self.act(self.gate(combined))  # [B, D]
        fused = gate * rgb_feat + (1 - gate) * flow_feat  # [B, D]
        return fused


class Model(nn.Module):
    def __init__(
        self,
        ckpt_dir,
        trainable_names=[],
        adaptor_layers=[],
        adapt_params={},
        out_dim=None,
        freeze=False,
        use_flow=False,
        flow_params={}
    ):
        super().__init__()
        self.use_flow = use_flow
        self.spatial_model = vit_small(
            img_size=518,
            init_values=1.0,
            patch_size=14,
            block_chunks=0,
            adaptor_layers=adaptor_layers,
            adapt_params=adapt_params,
        )
        import requests
        import os

        num_features = self.spatial_model.num_features
        self.lin = torch.nn.Linear(num_features, out_dim)
        self.bn = SafeBatchNorm1d(out_dim)

        if self.use_flow:
            self.flow_branch = FlowBranch(**flow_params)
            self.fusion = GatedFusion(out_dim)

        if idist.get_local_rank() == 0 or idist.get_world_size() == 0:
            if not os.path.isfile("/tmp/tmp.pth"):
                r = requests.get(ckpt_dir)
                open("/tmp/tmp.pth", "wb").write(r.content)

        if idist.get_world_size() > 0:
            idist.barrier()
        dict_additional = self.spatial_model.load_state_dict(
            torch.load("/tmp/tmp.pth", map_location="cpu"), strict=False
        )

        for name, param in self.spatial_model.named_parameters():
            if name in dict_additional.missing_keys:
                param.requires_grad = True
            elif any(name.startswith(s) for s in trainable_names):
                param.requires_grad = True
            else:
                param.requires_grad = False
                if torch.cuda.is_bf16_supported():
                    param.to(torch.bfloat16)

        if freeze:
            for name, param in self.named_parameters():
                param.requires_grad = False

        if idist.get_local_rank() == 0 or idist.get_world_size() == 0:
            if not os.path.isfile("/tmp/tmp.pth"):
                r = requests.get(ckpt_dir)
                open("/tmp/tmp.pth", "wb").write(r.content)

        if idist.get_world_size() > 0:
            idist.barrier()
        dict_additional = self.spatial_model.load_state_dict(
            torch.load("/tmp/tmp.pth", map_location="cpu"), strict=False
        )

        for name, param in self.spatial_model.named_parameters():
            if name in dict_additional.missing_keys:
                param.requires_grad = True
            elif any(name.startswith(s) for s in trainable_names):
                param.requires_grad = True
            else:
                param.requires_grad = False
                if torch.cuda.is_bf16_supported():
                    param.to(torch.bfloat16)

        if freeze:
            for name, param in self.named_parameters():
                param.requires_grad = False
                
    def pad(self, tensor, length):
        return torch.cat(
            [
                tensor,
                tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_(),
            ]
        )

    def forward(self, list_of_frames, flow_data=None, max_len=1024):
        lengths = torch.tensor([len(x_i) for x_i in list_of_frames])

        y = self.spatial_model.forward_features(torch.cat(list_of_frames, dim=0))[
            "x_norm_clstoken"
        ]
        list_of_original_features = y

        y = self.bn(self.lin(y))

        if self.use_flow and flow_data is not None:
            # Concatenate flow data across batch
            flow_concat = torch.cat(flow_data, dim=0)  # [total_frames, H, W, 2]

            # Process flow features
            flow_features = self.flow_branch(flow_concat)  # [total_frames, out_dim]

            # Fuse RGB and flow per frame
            fused_features = []
            start_idx = 0
            for length in lengths:
                rgb_seq = y[start_idx:start_idx + length]  # [length, out_dim]
                flow_seq = flow_features[start_idx:start_idx + length]  # [length, out_dim]
                fused_seq = torch.stack([self.fusion(rgb, flow) for rgb, flow in zip(rgb_seq, flow_seq)])  # [length, out_dim]
                fused_features.append(fused_seq)
                start_idx += length
            y = torch.cat(fused_features, dim=0)  # [total_frames, out_dim]

        if max_len is None:
            max_len = max(lengths)
        y = torch.cat(
            [
                self.pad(y[sum(lengths[:idx]) : sum(lengths[: idx + 1])], max_len)
                for idx, lgt in enumerate(lengths)
            ]
        )
        y = y.reshape(len(lengths), max_len, y.shape[1])

        mask = torch.zeros(
            y.shape[0],
            y.shape[1],
            device=y.device,
        )
        for i, l in enumerate(lengths):
            mask[i, :l] = 1
        mask = mask.bool()
        return y, mask, {"list_of_original_features": list_of_original_features}