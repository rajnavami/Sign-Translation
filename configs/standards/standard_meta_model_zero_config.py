# sign2gpt/Sign2GPT/configs/standards/standard_meta_model_zero_config.py
#
# CHANGES vs current version:
#   REMOVED: "flow_ckpt_dir": None  — this parameter no longer exists in
#            dino_adaptor_model.py. Keeping it would cause a TypeError on init.
#   KEPT:    "use_flow": True
#   KEPT:    "flow_params": {}

import numpy as np


def get_sign_encoder():
    model_name = "models.trial_models.test_pretraining"
    dim_model  = 512
    dropout    = 0.1
    num_heads  = 8

    adaptor_layers = list(range(12))  # # ALL 12 layers
    adapt_params = {
        "w_lora":    True,
        "w_lora_ff": True,
        "lora_rank": 16,
        "lora_a":    16.0,
        "lora_drop": 0.05,
        "rng_init":  False,
    }

    sign_model_params = {
        "spatial_name": "models.spatial_models.frame_models.dino_adaptor_model",
        "spatial_params": {
            # DINOv2 ViT-S pretrained checkpoint (local path)
            "ckpt_dir": "/data/sign2gpt/Sign2GPT/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",

            # Empty — only LoRA missing_keys become trainable
            "trainable_names": [],

            # LoRA applied to top 3 blocks
            "adaptor_layers": adaptor_layers,
            "adapt_params":   adapt_params,

            # ── Optical flow ──────────────────────────────────────────────────
            # use_flow=True  : enables FlowBranch + GatedFusion
            #                  requires flow_lmdb_dir set in dataset config
            # use_flow=False : pure RGB baseline, no extra params
            "use_flow":    True,
            "flow_params": {},
            # NOTE: flow_ckpt_dir has been REMOVED — it no longer exists in
            # dino_adaptor_model.py. Flow is precomputed, no model needed.
        },
        "encoder_name": "models.metaformer.meta_model",
        "encoder_params": {
            "emb_name": "models.metaformer.emb.sine_pos",
            "emb_params": {
                "in_dim":     dim_model,
                "d_model":    dim_model,
                "pos_config": {"name": "my_sine", "dim_model": dim_model},
            },
            "net_name": "models.metaformer.net.downsampler_net",
            "net_params": {
                "drop_path_rate":         dropout,
                "use_layer_scale":        True,
                "layer_scale_init_value": 1e-5,
                "layer_norm_type":        "post",
                "layers":                 [2, 2],
                "downsamples":            [True],
                "embed_dims":             [dim_model, dim_model],
                "mixer_params": {
                    "residual_dropout":      dropout,
                    "num_heads":             num_heads,
                    "use_rotary_embeddings": True,
                },
                "attention_params": {
                    "name":        "local_mask",
                    "dropout":     dropout,
                    "window_size": 7,
                },
                "mlp_params": {
                    "name":                    "MLP",
                    "hidden_layer_multiplier": 4,
                    "activation":              "gelu",
                    "dropout":                 dropout,
                },
            },
            "post_name":   "models.metaformer.post.identity_head",
            "post_params": {"d_model": dim_model},
            "inits":       "xavier",
        },
    }
    return model_name, sign_model_params, dim_model


def get_proto_head_params(dim_model):
    post_name  = "models.metaformer.post.zero_fasttext_prototype_head"
    post_params = {
        "in_dim":                    dim_model,
        "hidden_dim":                300,
        "num_classes":               2533,
        "dropout":                   0.2,
        "class_temperature":         0.1,
        "time_temperature":          0.1,
        "dynamic_time_temperatures": False,
        "dynamic_class_temperatures": False,
        "emb_lang":                  "de",
        "emb_pkl_dir":               "data/phoenix2014t/processed_words.phx_pkl",
        "trainable_emb":             True,
    }
    return post_name, post_params


def get_decoder_adaptor_params():
    adaptor_params = {
        "adapt_layers": list(range(24)),
        "lora_layers":  list(range(24)),
        "w_lora_ff":    False,
        "lora_rank":    4,
        "lora_drop":    0.1,
        "gate_type":    "clamp",
        "lora_a":       4.0,
        "adapt_tokens": False,
    }
    return adaptor_params