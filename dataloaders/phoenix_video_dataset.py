# dataloaders/phoenix_video_dataset.py
#
# CHANGES vs original:
#   1. __init__: new optional parameter `flow_lmdb_dir` (default None = disabled)
#   2. __getitem__: loads flow using same frame selection as RGB, adds "flow" key
#   3. get_ds: passes flow_lmdb_dir from config to dataset constructor
#
# Everything else is IDENTICAL to the original. If flow_lmdb_dir is not set
# in the config, behaviour is exactly the same as before — no "flow" key in
# the batch, no FlowLMDBUtility created, zero overhead.

from torch.utils.data import Dataset
import lmdb
from PIL import Image
import io
import pickle
import numpy as np
import torch
from dataloaders.data_utils.my_concat_dataset import MyConcatDataset
from dataloaders.data_utils.lmdb_utils import LMDBUtility
from dataloaders.data_utils.flow_lmdb_utils import FlowLMDBUtility  # NEW
from tqdm import tqdm
import pandas as pd
import ignite.distributed as idist


class PhoenixVideoDataset(Dataset):
    def __init__(
        self,
        df,
        lmdb_video_dir,
        dict_gloss_to_id,
        transform,
        isValid,
        dict_sentence=None,
        dict_lem_to_id=None,
        dict_lem_counter=None,
        flow_lmdb_dir=None,     # NEW — path to precomputed flow LMDB directory
                                #       None means flow is disabled (default)
    ):
        self.items = df.to_dict("records")
        self.lmdb_dir = lmdb_video_dir
        self.transform = transform
        self.isValid = isValid
        self.lmdb_util_video = None
        self.dict_gloss_to_id = dict_gloss_to_id
        self.dict_sentence = dict_sentence
        self.dict_lem_to_id = dict_lem_to_id
        self.dict_lem_counter = dict_lem_counter

        # NEW — store flow directory; None means flow is not used
        self.flow_lmdb_dir = flow_lmdb_dir
        self.lmdb_util_flow = None   # lazy init (same pattern as video LMDB)

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        # Original collate_fn — unchanged.
        # It collects all values for each key into a list.
        # If "flow" is in the batch items (when flow is enabled), it will
        # automatically be collected as a list of tensors, which is exactly
        # what the trainer's prep_batch expects.
        key_set = {k for k in batch[0].keys()}
        val_list = lambda k: [d.get(k) for d in batch if d.get(k) is not None]
        return {k: val_list(k) for k in key_set}

    def __getitem__(self, idx):
        item = self.items[idx]
        file_name = item["name"]

        # ── Original video loading (UNCHANGED) ──────────────────────────────
        if not self.lmdb_util_video:
            self.lmdb_util_video = LMDBUtility(
                f"{self.lmdb_dir}/{file_name}",
            )
            self.num_frames = self.lmdb_util_video.details["num_frames"]

        if self.isValid:
            start_frame = 0
            end_frame = self.num_frames
        else:
            start_frame = np.random.randint(0, self.transform.random_shift)
            end_frame = np.random.randint(
                self.num_frames - self.transform.random_shift, self.num_frames + 1
            )

        selection = np.arange(start_frame, end_frame, self.transform.stride)
        selection = selection.astype(int)

        if len(selection) > self.transform.max_seq_len:
            selection = np.random.choice(
                selection, size=self.transform.max_seq_len, replace=False
            )
            selection = np.sort(selection)

        frames = self.lmdb_util_video.get_frames(selection)

        if self.transform:
            frames = self.transform.aug_video(frames, self.isValid)
        else:
            frames = torch.tensor(np.stack(frames)).float()
        # ── End of original video loading ────────────────────────────────────

        # ── NEW: flow loading ─────────────────────────────────────────────────
        # Only runs when flow_lmdb_dir was set in the config.
        # Uses the EXACT same `selection` array as RGB frame loading above,
        # so RGB frame i and flow frame i always correspond to the same moment
        # in the video. This temporal alignment is critical.
        #
        # Why lazy init here (not in __init__)?
        # - Each PhoenixVideoDataset instance is for ONE video (one file_name).
        # - DataLoader workers are forked AFTER __init__, so LMDB handles
        #   opened in __init__ would be inherited by all workers and cause
        #   conflicts. Opening on first __getitem__ call is safe.
        flow_tensor = None
        if self.flow_lmdb_dir is not None:
            if self.lmdb_util_flow is None:
                self.lmdb_util_flow = FlowLMDBUtility(
                    f"{self.flow_lmdb_dir}/{file_name}"
                )

            # Load flow frames for the same selection as RGB
            # Returns list of (64, 64, 2) float16 numpy arrays
            flows = self.lmdb_util_flow.get_flows(selection)

            # Stack into tensor and reorder dims:
            #   np.stack(flows) shape: (T, 64, 64, 2)
            #   .permute(0,3,1,2) -> (T, 2, 64, 64)  [standard PyTorch NCHW]
            #   .float() converts float16 -> float32 for autocast compatibility
            flow_tensor = torch.from_numpy(
                np.stack(flows).astype(np.float32)   # float16 -> float32
            ).permute(0, 3, 1, 2)
            # Final shape: (T, 2, 64, 64)
            # T = number of selected frames (same as len(frames))
            # Channel 0 = dx, Channel 1 = dy, both in [-1, 1]
        # ── End of flow loading ───────────────────────────────────────────────

        # ── Original gloss/sentence logic (UNCHANGED) ────────────────────────
        sentence = item["translation"]
        glosses = item["orth"].split(" ")
        pseudo_gloss_ids = []
        if self.dict_sentence is not None:
            if sentence in self.dict_sentence:
                lems = self.dict_sentence[sentence]
                pseudo_gloss_ids = [
                    self.dict_lem_to_id[lem]
                    for lem in lems
                    if self.dict_lem_counter[lem] / len(self.dict_sentence) < 0.4
                ]

        if self.dict_gloss_to_id:
            gloss_ids = [self.dict_gloss_to_id[gloss] for gloss in glosses]

        return {
            "index": torch.tensor(idx).long(),
            "frames": frames,
            "sentence": sentence,
            "file_name": file_name,
            **(
                {"gloss_ids": torch.tensor(gloss_ids).long()}
                if self.dict_gloss_to_id
                else {}
            ),
            **(
                {"pseudo_gloss_ids": torch.tensor(pseudo_gloss_ids).long()}
                if self.dict_lem_to_id
                else {}
            ),
            # NEW — only included in the batch when flow is enabled.
            # If flow_lmdb_dir is None, this key is absent from the batch
            # and all downstream code treats it as disabled (via .get("flow", None))
            **(
                {"flow": flow_tensor}
                if flow_tensor is not None
                else {}
            ),
        }


def get_ds(ds_params, transform):
    from dataloaders.data_utils.file_utils import read_json, read_pickle

    df = pd.read_csv(ds_params["csv_dir"], sep=ds_params["sep"])
    if "gloss_dir" in ds_params:
        dict_gloss_to_id = read_json(ds_params["gloss_dir"])
    else:
        dict_gloss_to_id = None

    if "pseudo_gloss_dir" in ds_params:
        dict_processed_words = read_pickle(ds_params["pseudo_gloss_dir"])
        dict_sentence = dict_processed_words["dict_sentence"]
        dict_lem_to_id = dict_processed_words["dict_lem_to_id"]
        dict_lem_counter = dict_processed_words["dict_lem_counter"]
    else:
        dict_sentence = None
        dict_lem_to_id = None
        dict_lem_counter = None

    list_of_ds = []
    length = 0
    for gp, d in tqdm(df.groupby("name")):
        ds = PhoenixVideoDataset(
            d,
            **ds_params["ds_params"],
            dict_gloss_to_id=dict_gloss_to_id,
            dict_sentence=dict_sentence,
            dict_lem_to_id=dict_lem_to_id,
            dict_lem_counter=dict_lem_counter,
            transform=transform,
            # NEW — read from config; None if not present (flow disabled)
            # Set this in your config ds_params to enable flow:
              # "flow_lmdb_dir": "/path/to/phoenix2014t/lmdb_flows"
            flow_lmdb_dir=ds_params.get("flow_lmdb_dir", None),
        )
        list_of_ds.append(ds)
        collate_fn = ds.collate_fn
        length += len(ds)

    ds = MyConcatDataset(list_of_ds)
    dl = idist.auto_dataloader(
        ds,
        shuffle=ds_params["shuffle"],
        sampler=None,
        num_workers=ds_params["num_workers"],
        batch_size=ds_params["bs"],
        drop_last=ds_params["drop_last"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dl, {"length": length, "dict_lem_to_id": dict_lem_to_id}