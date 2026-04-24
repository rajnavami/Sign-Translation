# models/model_sign_encoder/basic_sign_encoder.py
#
# CHANGES vs original:
#   CHANGED: forward() accepts list_of_flows and passes it to spatial_model
#   Everything else is identical

import torch
from torch import nn
import ignite.distributed as idist
import numpy as np
import torch.nn.functional as F
import importlib


class Model(nn.Module):
    def __init__(
        self,
        spatial_name,
        spatial_params,
        encoder_name,
        encoder_params,
    ):
        super().__init__()

        spatial_mod = importlib.import_module(spatial_name, package=None)
        spatial_params["out_dim"] = encoder_params["emb_params"]["in_dim"]
        self.spatial_model = spatial_mod.Model(**spatial_params)

        encoder_mod = importlib.import_module(encoder_name, package=None)
        self.encoder = encoder_mod.MetaFormer(**encoder_params)

    def forward(
        self,
        frame_features,
        max_len,
        list_of_flows=None,     # NEW — passed down from test_pretraining
                                #       None = flow disabled, no change in behaviour
    ):
        # CHANGE: pass list_of_flows to spatial_model
        # dino_adaptor_model.forward() accepts it as optional kwarg
        # When None, spatial model runs RGB-only (identical to original)
        x, mask, dict_feat = self.spatial_model(
            frame_features,
            max_len=max_len,
            list_of_flows=list_of_flows,    # NEW
        )

        enc_output = self.encoder(x, mask)

        return {
            "enc_output": enc_output,
            "dict_feat":  dict_feat,
        }