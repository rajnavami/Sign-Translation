# dataloaders/phoenix_video_dataset.py
#
# FIX: "got multiple values for keyword argument 'flow_lmdb_dir'"
#
# ROOT CAUSE:
#   In get_ds(), the call was:
#       ds = PhoenixVideoDataset(
#           d,
#           **ds_params["ds_params"],          # unpacks flow_lmdb_dir from here
#           ...
#           flow_lmdb_dir=ds_params.get(...),  # AND passes it again explicitly
#       )
#
#   Since flow_lmdb_dir is inside ds_params["ds_params"] in the config,
#   **ds_params["ds_params"] already unpacks it. Passing it again explicitly
#   causes the "multiple values" TypeError.
#
# FIX:
#   Remove the explicit flow_lmdb_dir= line from the get_ds() call.
#   flow_lmdb_dir arrives correctly via **ds_params["ds_params"].
#   PhoenixVideoDataset.__init__ already has flow_lmdb_dir=None as default,
#   so if it's absent from ds_params["ds_params"], it defaults to None safely.

from torch.utils.data import Dataset
import lmdb
from PIL import Image
import io
import pickle
import numpy as np
import torch
from dataloaders.data_utils.my_concat_dataset import MyConcatDataset
from dataloaders.data_utils.lmdb_utils import LMDBUtility
from dataloaders.data_utils.flow_lmdb_utils import FlowLMDBUtility
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
        flow_lmdb_dir=None,     # None = flow disabled, no change in behaviour
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
        self.flow_lmdb_dir = flow_lmdb_dir
        self.lmdb_util_flow = None   # lazy init — same pattern as video LMDB

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        key_set = {k for k in batch[0].keys()}
        val_list = lambda k: [d.get(k) for d in batch if d.get(k) is not None]
        return {k: val_list(k) for k in key_set}

    def __getitem__(self, idx):
        item = self.items[idx]
        file_name = item["name"]

        # ── Original video loading (UNCHANGED) ───────────────────────────────
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
                self.num_frames - self.transform.random_shift,
                self.num_frames + 1
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
        # ── End original video loading ────────────────────────────────────────

        # ── Flow loading ──────────────────────────────────────────────────────
        # Uses the EXACT same selection as RGB — temporal alignment guaranteed.
        # Lazy init mirrors the video LMDB pattern.
        flow_tensor = None
        if self.flow_lmdb_dir is not None:
            if self.lmdb_util_flow is None:
                self.lmdb_util_flow = FlowLMDBUtility(
                    f"{self.flow_lmdb_dir}/{file_name}"
                )
            flows = self.lmdb_util_flow.get_flows(selection)
            # (T, 64, 64, 2) -> (T, 2, 64, 64), float32
            flow_tensor = torch.from_numpy(
                np.stack(flows).astype(np.float32)
            ).permute(0, 3, 1, 2)
        # ── End flow loading ──────────────────────────────────────────────────

        # ── Original gloss/sentence logic (UNCHANGED) ─────────────────────────
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
        dict_sentence  = dict_processed_words["dict_sentence"]
        dict_lem_to_id = dict_processed_words["dict_lem_to_id"]
        dict_lem_counter = dict_processed_words["dict_lem_counter"]
    else:
        dict_sentence  = None
        dict_lem_to_id = None
        dict_lem_counter = None

    list_of_ds = []
    length = 0
    for gp, d in tqdm(df.groupby("name")):
        ds = PhoenixVideoDataset(
            d,
            # FIX: **ds_params["ds_params"] already unpacks flow_lmdb_dir
            # because it is defined inside ds_params["ds_params"] in the config.
            # Do NOT pass flow_lmdb_dir again explicitly — that caused the
            # "multiple values for keyword argument" TypeError.
            **ds_params["ds_params"],
            dict_gloss_to_id=dict_gloss_to_id,
            dict_sentence=dict_sentence,
            dict_lem_to_id=dict_lem_to_id,
            dict_lem_counter=dict_lem_counter,
            transform=transform,
            # flow_lmdb_dir is NOT passed here — it comes from **ds_params["ds_params"]
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