import lmdb
from PIL import Image
import numpy as np
import io
import pickle
from collections import defaultdict
import torch
import os

class LMDBUtility(object):
    def __init__(self, lmdb_dir, landmark_keys={}):
        self.lmdb_dir = lmdb_dir

        with lmdb.open(
            path=f"{lmdb_dir}",
            readonly=True,
            readahead=False,
            lock=False,
            meminit=False,
        ).begin(write=False) as txn:
            self.details = pickle.loads(txn.get(key=f"details".encode("ascii")))
        self.landmark_keys = landmark_keys
        self._hand_landmarks = None

    def get_frames(self, selection):
        with lmdb.open(
            path=f"{self.lmdb_dir}",
            readonly=True,
            readahead=False,
            lock=False,
            meminit=False,
        ).begin(write=False) as txn:
            frames = [
                np.array(Image.open(io.BytesIO(txn.get(key=f"{idx}".encode("ascii")))))
                for idx in selection
            ]
        return frames

    def get_hand_landmarks(self, selection):
        """
        Load pre-extracted hand landmarks for the given frame indices.
        """
        if self._hand_landmarks is None:
            video_name = os.path.basename(self.lmdb_dir)
            parent_dir = os.path.dirname(self.lmdb_dir)
            npy_path = os.path.join(parent_dir, f"{video_name}_hand_landmarks.npy")
            if not os.path.exists(npy_path):
                return None
            self._hand_landmarks = np.load(npy_path)

        return self._hand_landmarks[np.array(selection)]