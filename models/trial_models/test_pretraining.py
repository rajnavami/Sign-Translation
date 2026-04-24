# models/trial_models/test_pretraining.py
#
# CHANGES vs original:
#   CHANGED: forward() accepts list_of_flows and passes it to sign_model
#   Everything else is identical

from torch import nn
import importlib


class Model(nn.Module):
    def __init__(
        self,
        sign_model_name,
        sign_model_params,
        post_name,
        post_params,
    ):
        super().__init__()

        mod = importlib.import_module(sign_model_name, package=None)
        self.sign_model = mod.Model(**sign_model_params)

        post_mod = importlib.import_module(post_name, package=None)
        self.post_model = post_mod.HeadModel(**post_params)

        self.dim = sign_model_params["encoder_params"]["emb_params"]["d_model"]

    def forward(
        self,
        frame_features,
        max_len,
        list_of_flows=None,     # NEW — received from trainer's prep_batch
                                #       passed straight down to sign_model
    ):
        # CHANGE: pass list_of_flows to sign_model
        # basic_sign_encoder.forward() passes it further to spatial_model
        # When None: identical behaviour to original
        dict_sign_output = self.sign_model(
            frame_features,
            max_len=max_len,
            list_of_flows=list_of_flows,    # NEW
        )

        post_features, post_mask = (
            dict_sign_output["enc_output"]["post_output"]["x"],
            dict_sign_output["enc_output"]["post_output"]["mask"],
        )

        if self.post_model is not None:
            dict_post_output = self.post_model(post_features, post_mask)
        else:
            dict_post_output = {}

        return {
            "dict_post_output": dict_post_output,
            **dict_sign_output,
        }