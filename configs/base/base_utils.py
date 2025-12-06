from pathlib import Path


def get_checkpoint_path(base_name, name):
    ckpt_path = "checkpoint_dinov3_imgsize=512" #_dinov2
    return ckpt_path


def get_lmdb_path():
    lmdb_path = "data"
    return lmdb_path
