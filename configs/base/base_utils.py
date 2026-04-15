from pathlib import Path


def get_checkpoint_path(base_name, name):
    ckpt_path = "dinov3_optical_flow_rank16_alllayers" #_dinov2
    return ckpt_path


def get_lmdb_path():
    lmdb_path = "data"
    return lmdb_path
