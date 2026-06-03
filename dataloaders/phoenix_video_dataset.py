from torch.utils.data import Dataset
import lmdb
from PIL import Image
import io
import pickle
import numpy as np
import torch
from dataloaders.data_utils.my_concat_dataset import MyConcatDataset
from dataloaders.data_utils.lmdb_utils import LMDBUtility
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
        flow_lmdb_dir=None,
    ):
        self.items = df.to_dict("records")
        self.lmdb_dir = lmdb_video_dir
        self.flow_lmdb_dir = flow_lmdb_dir
        self.transform = transform
        self.isValid = isValid
        self.lmdb_util_video = None
        # NOTE: lmdb_util_flow removed — flow is NOT stored as images,
        # it is stored as pickled numpy arrays. We open it directly with lmdb.
        self.flow_env = None
        self.dict_gloss_to_id = dict_gloss_to_id
        self.dict_sentence = dict_sentence
        self.dict_lem_to_id = dict_lem_to_id
        self.dict_lem_counter = dict_lem_counter

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        key_set = {k for k in batch[0].keys()}
        val_list = lambda k: [d.get(k) for d in batch if d.get(k) is not None]
        return {k: val_list(k) for k in key_set}

    def _read_flow_frames(self, file_name, selection):
        """
        Read optical flow frames from LMDB.

        Flow is stored as pickled numpy arrays (NOT PIL images).
        optical_flow_embed.py saves: txn.put(f"{i:06d}".encode(), pickle.dumps(flow))

        This is fundamentally different from the video LMDB which stores JPEG frames.
        Using LMDBUtility.get_frames() would crash with PIL.UnidentifiedImageError.
        """
        flow_path = f"{self.flow_lmdb_dir}/{file_name}"
        # Open a new env per call (worker-safe, no shared state)
        env = lmdb.open(flow_path, readonly=True, lock=False, readahead=False)
        flows = []
        with env.begin() as txn:
            for i in selection:
                raw = txn.get(f"{i:06d}".encode())
                if raw is None:
                    # Fallback: some videos may have one fewer flow frame
                    # (flow has N-1 frames for N video frames, first is zeros)
                    raw = txn.get(f"{max(0, i-1):06d}".encode())
                if raw is None:
                    # Zero flow as last resort
                    flows.append(np.zeros((64, 64, 2), dtype=np.float32))
                else:
                    flows.append(pickle.loads(raw))
        env.close()
        return flows

    def __getitem__(self, idx):
        item = self.items[idx]
        file_name = item["name"]

        if not self.lmdb_util_video:
            self.lmdb_util_video = LMDBUtility(f"{self.lmdb_dir}/{file_name}")
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

        # FIX: read flow with pickle, NOT with PIL image reader
        flow_data = None
        if self.flow_lmdb_dir:
            flow_frames = self._read_flow_frames(file_name, selection)
            flow_data = torch.tensor(np.stack(flow_frames)).float()  # [T, H, W, 2]

        if self.transform:
            frames = self.transform.aug_video(frames, self.isValid)
        else:
            frames = torch.tensor(np.stack(frames)).float()

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

        result = {
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
        }
        if flow_data is not None:
            result["flow_data"] = flow_data
        return result


def get_ds(ds_params, transform):
    from dataloaders.data_utils.file_utils import read_json, read_pickle

    df = pd.read_csv(ds_params["csv_dir"], sep=ds_params["sep"])
    if "gloss_dir" in ds_params:
        dict_gloss_to_id = read_json(ds_params["gloss_dir"])
    else:
        dict_gloss_to_id = None

    if "pseudo_gloss_dir" in ds_params:
        dict_processed_words = read_pickle(ds_params["pseudo_gloss_dir"])
        dict_sentence   = dict_processed_words["dict_sentence"]
        dict_lem_to_id  = dict_processed_words["dict_lem_to_id"]
        dict_lem_counter = dict_processed_words["dict_lem_counter"]
    else:
        dict_sentence    = None
        dict_lem_to_id   = None
        dict_lem_counter = None

    list_of_ds = []
    length = 0
    for gp, d in tqdm(df.groupby("name")):
        ds = PhoenixVideoDataset(
            d,
            lmdb_video_dir=ds_params["ds_params"]["lmdb_video_dir"],
            flow_lmdb_dir=ds_params["ds_params"].get("flow_lmdb_dir"),
            dict_gloss_to_id=dict_gloss_to_id,
            dict_sentence=dict_sentence,
            dict_lem_to_id=dict_lem_to_id,
            dict_lem_counter=dict_lem_counter,
            transform=transform,
            isValid=ds_params["ds_params"]["isValid"],
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