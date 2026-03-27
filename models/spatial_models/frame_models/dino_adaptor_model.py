import torch
from torch import nn
from models.dinov2.model.vision_transformer import *
import ignite.distributed as idist
from models.spatial_models.frame_models.pose_encoder import PoseTemporalEncoder

class Model(nn.Module):
    def __init__(
        self,
        ckpt_dir,
        trainable_names=[],
        adaptor_layers=[],
        adapt_params={},
        out_dim=None,
        freeze=False,
        use_hand_emb=False,
        num_hands=2,
        hand_emb_dim=208,
        fusion_dim=256
    ):
        super().__init__()
        self.spatial_model = vit_small(
            img_size=518,
            init_values=1.0,
            patch_size=14,
            block_chunks=0,
            adaptor_layers=adaptor_layers,
            adapt_params=adapt_params,
        )
        self.use_hand_emb = use_hand_emb
        import requests
        import os

        num_features = self.spatial_model.num_features
        
        if use_hand_emb:
            # hand_emb_dim = num_hands * 63

            self.dino_proj = nn.Sequential(nn.Linear(num_features, fusion_dim),nn.LayerNorm(fusion_dim))
            # self.hand_proj = nn.Sequential(nn.Linear(hand_emb_dim, fusion_dim),nn.LayerNorm(fusion_dim))
            # combine dino-hand embeddings
            # self.hand_proj = nn.Sequential(nn.Linear(hand_emb_dim, fusion_dim),nn.LayerNorm(fusion_dim))
            self.fusion_lin = nn.Linear(fusion_dim * 2, out_dim)
            #added for gated fusion
            self.pose_encoder = PoseTemporalEncoder(
                input_dim=hand_emb_dim,   # 208
                d_model=fusion_dim,       # 256
                n_heads=4,
                n_layers=2,
                dropout=0.1,
            )
            # self.fusion_out = nn.Linear(fusion_dim, out_dim)
            # self.pose_gate  = nn.Parameter(torch.zeros(1))
        else:
            self.lin = nn.Linear(num_features, out_dim)
            
        self.bn = torch.nn.BatchNorm1d(out_dim)

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
    def forward(self, list_of_frames, max_len=1024, list_of_hand_landmarks=None):
        # Always compute lengths — needed for padding regardless of pose
        lengths = torch.tensor([len(x_i) for x_i in list_of_frames])
        
        # Always run DINO
        all_frames = torch.cat(list_of_frames, dim=0)
        dino_feat = self.spatial_model.forward_features(all_frames)["x_norm_clstoken"]
        list_of_original_features = dino_feat
    
        # # Debug logs — remove after confirming
        # if self.use_hand_emb:
        #     print(f"use_hand_emb: True")
        #     print(f"list_of_hand_landmarks is None: {list_of_hand_landmarks is None}")
        #     if list_of_hand_landmarks is not None:
        #         print(f"num samples with landmarks: {len(list_of_hand_landmarks)}")
        #         print(f"landmark shape[0]: {list_of_hand_landmarks[0].shape}")
        #         print(f"landmark dtype: {list_of_hand_landmarks[0].dtype}")
    
        # Fusion branch
        if self.use_hand_emb and list_of_hand_landmarks is not None:
            hand_feat = torch.cat(list_of_hand_landmarks, dim=0).to(
                device=dino_feat.device, dtype=dino_feat.dtype
            )
            # # Debug logs — remove after confirming
            # print(f"hand_feat shape: {hand_feat.shape}")
            # print(f"dino_feat shape: {dino_feat.shape}")
            # print(f"hand_feat dtype: {hand_feat.dtype}")
            # print(f"dino_feat dtype: {dino_feat.dtype}")
            dino_out  = self.dino_proj(dino_feat)          # sum_T × fusion_dim
            # hand_out  = self.hand_proj(hand_feat)          # sum_T × fusion_dim
            hand_out = self.pose_encoder(hand_feat, lengths)
            
            # gate = torch.sigmoid(self.pose_gate)      # scalar in (0, 1)
            # fused = dino_out + gate * hand_out         # gated residual addition
            # y = self.bn(self.fusion_out(fused))    # sum_T × out_dim
            y = torch.cat([dino_out, hand_out], dim=-1)        # sum_T x 512
            y = self.bn(self.fusion_lin(y))  
        else:
            y = self.bn(self.lin(dino_feat))
    
        if max_len is None:
            max_len = max(lengths)
    
        y = torch.cat(
            [
                self.pad(y[sum(lengths[:idx]): sum(lengths[:idx + 1])], max_len)
                for idx, lgt in enumerate(lengths)
            ]
        )
        y = y.reshape(len(lengths), max_len, y.shape[1])
    
        mask = torch.zeros(y.shape[0], y.shape[1], device=y.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1
        mask = mask.bool()
    
        return y, mask, {"list_of_original_features": list_of_original_features}

    # def forward(self, list_of_frames, max_len=1024, list_of_hand_landmarks=None):
    #     if self.use_hand_emb:
    #         print(f"use_hand_emb: True")
    #         print(f"list_of_hand_landmarks is None: {list_of_hand_landmarks is None}")
    #         if list_of_hand_landmarks is not None:
    #             print(f"num samples with landmarks: {len(list_of_hand_landmarks)}")
    #             print(f"landmark shape[0]: {list_of_hand_landmarks[0].shape}")
    #             print(f"landmark dtype: {list_of_hand_landmarks[0].dtype}")
    #         lengths = torch.tensor([len(x_i) for x_i in list_of_frames])

    #     all_frames = torch.cat(list_of_frames, dim=0)
    #     dino_feat = self.spatial_model.forward_features(all_frames)["x_norm_clstoken"]
    #     list_of_original_features = dino_feat
    #     if self.use_hand_emb and list_of_hand_landmarks is not None:
    #         hand_feat = torch.cat(list_of_hand_landmarks, dim=0).to(
    #             device=dino_feat.device, dtype=dino_feat.dtype
    #         )
    #         # ADD THIS
    #         print(f"hand_feat shape: {hand_feat.shape}")
    #         print(f"dino_feat shape: {dino_feat.shape}")
    #         print(f"hand_feat dtype: {hand_feat.dtype}")
    #         print(f"dino_feat dtype: {dino_feat.dtype}")
    #         # END DEBUG
        
    #     if self.use_hand_emb and list_of_hand_landmarks is not None:
    #         # hand_feat = torch.cat(list_of_hand_landmarks, dim=0).to(dino_feat.device)
    #         hand_feat = torch.cat(list_of_hand_landmarks, dim=0).to(device=dino_feat.device, dtype=dino_feat.dtype)
    #         y = torch.cat([self.dino_proj(dino_feat), self.hand_proj(hand_feat)], dim=-1)
    #         y = self.bn(self.fusion_lin(y))
    #     else:
    #         y = self.bn(self.lin(dino_feat))

    #     if max_len is None:
    #         max_len = max(lengths)
    #     y = torch.cat(
    #         [
    #             self.pad(y[sum(lengths[:idx]) : sum(lengths[: idx + 1])], max_len)
    #             for idx, lgt in enumerate(lengths)
    #         ]
    #     )
    #     y = y.reshape(len(lengths), max_len, y.shape[1])

    #     mask = torch.zeros(
    #         y.shape[0],
    #         y.shape[1],
    #         device=y.device,
    #     )
    #     for i, l in enumerate(lengths):
    #         mask[i, :l] = 1
    #     mask = mask.bool()
    #     return y, mask, {"list_of_original_features": list_of_original_features}