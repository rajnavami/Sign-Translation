# configs/PHX/PHX_example_s1_dyn_config.py
#
# CHANGES vs current version:
#   ADDED: "flow_lmdb_dir" inside ds_params for train, valid, AND test splits.
#          Without this, phoenix_video_dataset.py never loads flow, so
#          batch["flow"] is always absent and list_of_flows is always None.

from ml_collections import config_dict
from pathlib import Path
import os
import json
import numpy as np
import torch
from configs.base.base_utils import *
from loguru import logger


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.name      = Path(os.path.realpath(__file__)).stem
    base_name     = Path(os.path.realpath(__file__)).parent.name
    code_path     = str(Path(os.path.realpath(__file__)).resolve().parents[3])
    logger.info(f"code_path: {code_path}")

    cfg.log_name     = "dinov3_optical_flow_v2"
    ckpt_path        = get_checkpoint_path(base_name, cfg.name)
    lmdb_path        = get_lmdb_path()
    cfg.save_dir     = f"{ckpt_path}/{base_name}/{cfg.name}"
    logger.info(f"lmdb_path: {lmdb_path}")

    cfg.main_runner  = "trainer.psuedo_gloss_trainer"
    cfg.project_name = "final_phoenix2014t_pretrain"
    cfg.aug_name     = "augmentation.video.base_video_aug"

    cfg.aug_params = {
        "mean":         [0.485, 0.456, 0.406],
        "std":          [0.229, 0.224, 0.225],
        "strength":     0.2,
        "random_shift": 4,
        "stride":       2,
        "max_seq_len":  128,
    }

    base_bs     = 8
    cfg.bs      = int(4 * torch.cuda.device_count())
    cfg.accum   = 1
    cfg.num_workers = min(min(cfg.bs, int(10 * torch.cuda.device_count())), 10)

    cfg.gate_grad_multiplier = 1.0
    cfg.lr           = 3e-4
    cfg.lr_scheduler = "warmupwithcosine"
    cfg.lr_scheduler_params = config_dict.ConfigDict(
        {
            "lr_scale_factor":  0.01,
            "num_cycles":       1,
            "start_value_mult": 0.7,
            "end_value_mult":   0.7,
            "warmup_epochs":    5,
        }
    )

    cfg.optimizer_name   = "adamw"
    cfg.optimizer_params = config_dict.ConfigDict(
        {"lr": cfg.lr, "weight_decay": 0.001}
    )

    cfg.criterion_name   = "losses.base_loss"
    cfg.criterion_params = config_dict.ConfigDict(
        {
            "dict_of_loss_params": {
                "bce": {
                    "cls_name":    "losses.loss_functions.bce_loss",
                    "loss_params": {
                        "src_name": ("dict_post_output", "logits"),
                        "tgt_name": "pseudo_gloss_ids",
                    },
                    "weight": 10.0,
                },
            }
        }
    )

    cfg.max_epochs           = 100
    cfg.model_checkpoint_dir = ""

    cfg.train_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.valid_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.test_ds_name  = "dataloaders.phoenix_video_dataset"

    # ── Train split ───────────────────────────────────────────────────────────
    train_ds_params = {
        "csv_dir": f"{code_path}/data/phoenix2014t/PHOENIX-2014-T.train.corpus.csv",
        "pseudo_gloss_dir": f"{code_path}/data/phoenix2014t/processed_words.phx_pkl",
        "sep":  "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/phoenix2014t/lmdb_videos",
            "isValid": False,
            # NEW — tells phoenix_video_dataset.py to load flow alongside frames.
            # Must match the --flow_lmdb_dir you used in precompute_flow.py.
            # Remove this line (or set to None) to disable flow loading.
            "flow_lmdb_dir": f"{lmdb_path}/phoenix2014t/lmdb_flows",
        },
        "shuffle":     True,
        "num_workers": cfg.num_workers,
        "bs":          cfg.bs,
        "drop_last":   True,
    }
    cfg.train_ds_params = config_dict.ConfigDict(train_ds_params)

    # ── Validation split ──────────────────────────────────────────────────────
    valid_ds_params = {
        "csv_dir": f"{code_path}/data/phoenix2014t/PHOENIX-2014-T.dev.corpus.csv",
        "pseudo_gloss_dir": f"{code_path}/data/phoenix2014t/processed_words.phx_pkl",
        "sep":  "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/phoenix2014t/lmdb_videos",
            "isValid": True,
            # NEW — flow must be loaded for validation too so the model
            # behaves consistently between train and eval
            "flow_lmdb_dir": f"{lmdb_path}/phoenix2014t/lmdb_flows",
        },
        "shuffle":     False,
        "num_workers": cfg.num_workers,
        "bs":          cfg.bs,
        "drop_last":   False,
    }
    cfg.valid_ds_params = config_dict.ConfigDict(valid_ds_params)

    # ── Test split ────────────────────────────────────────────────────────────
    test_ds_params = {
        "csv_dir": f"{code_path}/data/phoenix2014t/PHOENIX-2014-T.test.corpus.csv",
        "pseudo_gloss_dir": f"{code_path}/data/phoenix2014t/processed_words.phx_pkl",
        "sep":  "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/phoenix2014t/lmdb_videos",
            "isValid": True,
            # NEW — same flow directory for test
            "flow_lmdb_dir": f"{lmdb_path}/phoenix2014t/lmdb_flows",
        },
        "shuffle":     False,
        "num_workers": cfg.num_workers,
        "bs":          cfg.bs,
        "drop_last":   False,
    }
    cfg.test_ds_params = config_dict.ConfigDict(test_ds_params)

    from configs.standards.standard_meta_model_zero_config import get_sign_encoder
    model_name, sign_model_params, dim_model = get_sign_encoder()

    cfg.model_name = model_name
    post_params = {
        "in_dim":                    dim_model,
        "hidden_dim":                300,
        "num_classes":               2330,
        "dropout":                   0.2,
        "class_temperature":         0.1,
        "time_temperature":          0.1,
        "dynamic_time_temperatures": True,
        "dynamic_class_temperatures": True,
        "emb_lang":                  "de",
        "emb_pkl_dir": f"{code_path}/data/phoenix2014t/processed_words.phx_pkl",
        "trainable_emb": False,
    }

    cfg.model_params = {
        "sign_model_name":   "models.model_sign_encoder.basic_sign_encoder",
        "sign_model_params": sign_model_params,
        "post_name":         "models.metaformer.post.zero_fasttext_prototype_head",
        "post_params":       post_params,
    }

    cfg.seed            = 1
    cfg.grad_clip_norm  = 1.0
    cfg.grad_clip_value = 1.0
    cfg.logger_name     = ["text", "tensorboard"]
    cfg.resume          = True
    cfg.train_length    = None
    cfg.val_length      = None
    cfg.log_every       = 100
    cfg.save_ckpt       = True
    cfg.score_factor    = 1
    cfg.score_name      = "valid/class_f1_score"
    cfg.bfloat16_only   = False
    cfg.mixup           = False

    return cfg