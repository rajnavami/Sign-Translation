from pathlib import Path


def get_checkpoint_path(base_name, name):
    ckpt_path = "ckpt_pose_transformer_concat_256"
    return ckpt_path


def get_lmdb_path():
    lmdb_path = "data"
    return lmdb_path
