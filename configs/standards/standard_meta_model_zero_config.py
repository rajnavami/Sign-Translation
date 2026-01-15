# sign2gpt/Sign2GPT/configs/standards/standard_meta_model_zero_config.py

import numpy as np


# -----------------------------
# DINOv3 + LoRA configuration
# -----------------------------
# Notes:
# - adaptor_layers controls WHICH transformer blocks get LoRA (0..11 for ViT-S depth=12).
# - adapt_params must include:
#     - w_lora      : enable LoRA in Attention (QKV + output proj)
#     - w_lora_ff   : enable LoRA in FFN (MLP/SwiGLU)
#     - lora_rank, lora_a, lora_drop
# - If you want "new style" LoRA (separate Q/K/V), add: "new": True
#   If you omit "new", it uses the "old style" LoRA (single low-rank update on combined QKV),
#   which matches the common DINOv2 setup you had before.


def get_sign_encoder():
    model_name = "models.trial_models.test_pretraining"
    dim_model = 512
    dropout = 0.1
    num_heads = 8

    # Choose which ViT blocks get LoRA (ViT-S has depth=12 => blocks 0..11)
    # adaptor_layers = list(range(12))  # all blocks
    # # adaptor_layers = [9, 10, 11]    # example: only last 3 blocks

    # adapt_params = {
    #     "w_lora": True,        # Attention LoRA ON
    #     "w_lora_ff": True,     # FFN LoRA ON
    #     "lora_rank": 16,
    #     "lora_drop": 0.05,
    #     "lora_a": 16.0,
    #     "rng_init": False,     # keep False for pretrained loading
    #     # "new": True,         # OPTIONAL: enable separate Q/K/V LoRA style
    # }

    # /////////////////
    
    adaptor_layers = [9, 10, 11]  # top 3 blocks for ViT-S depth=12
    adapt_params = {
        "w_lora": True,
        "w_lora_ff": True,
        "lora_rank": 4,
        "lora_a": 4.0,
        "lora_drop": 0.05,   # paper doesn’t pin this; 0.05–0.1 is typical
        "rng_init": False,
        # "new": True,  # optional; only if your DINOv3 LoRA code supports it
    }


    sign_model_params = {
        "spatial_name": "models.spatial_models.frame_models.dino_adaptor_model",
        "spatial_params": {
            # DINOv3 ViT-S16 pretrained checkpoint (local path)
            "ckpt_dir": "/data/sign2gpt/Sign2GPT/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",

            # Keep empty so only missing_keys (LoRA) become trainable via your loading logic
            "trainable_names": [],

            # LoRA applied to these blocks
            "adaptor_layers": adaptor_layers,

            # LoRA hyperparams
            "adapt_params": adapt_params,
        },
        "encoder_name": "models.metaformer.meta_model",
        "encoder_params": {
            "emb_name": "models.metaformer.emb.sine_pos",
            "emb_params": {
                "in_dim": dim_model,
                "d_model": dim_model,
                "pos_config": {"name": "my_sine", "dim_model": dim_model},
            },
            "net_name": "models.metaformer.net.downsampler_net",
            "net_params": {
                "drop_path_rate": dropout,
                "use_layer_scale": True,
                "layer_scale_init_value": 1e-5,
                "layer_norm_type": "post",
                "layers": [2, 2],
                "downsamples": [True],
                "embed_dims": [dim_model, dim_model],
                "mixer_params": {
                    "residual_dropout": dropout,
                    "num_heads": num_heads,
                    "use_rotary_embeddings": True,
                },
                "attention_params": {
                    "name": "local_mask",
                    "dropout": dropout,
                    "window_size": 7,
                },
                "mlp_params": {
                    "name": "MLP",
                    "hidden_layer_multiplier": 4,
                    "activation": "gelu",
                    "dropout": dropout,
                },
            },
            "post_name": "models.metaformer.post.identity_head",
            "post_params": {
                "d_model": dim_model,
            },
            "inits": "xavier",
        },
    }
    return model_name, sign_model_params, dim_model


def get_proto_head_params(dim_model):
    post_name = "models.metaformer.post.zero_fasttext_prototype_head"
    post_params = {
        "in_dim": dim_model,
        "hidden_dim": 300,
        "num_classes": 2533,
        "dropout": 0.2,
        "class_temperature": 0.1,
        "time_temperature": 0.1,
        "dynamic_time_temperatures": False,
        "dynamic_class_temperatures": False,
        "emb_lang": "de",
        "emb_pkl_dir": "data/phoenix2014t/processed_words.phx_pkl",
        "trainable_emb": True,
    }
    return post_name, post_params


def get_decoder_adaptor_params():
    adaptor_params = {
        "adapt_layers": list(range(24)),
        "lora_layers": list(range(24)),
        "w_lora_ff": False,
        "lora_rank": 4,
        "lora_drop": 0.1,
        "gate_type": "clamp",
        "lora_a": 4.0,
        "adapt_tokens": False,
    }
    return adaptor_params
