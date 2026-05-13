import argparse
import os
import sys
import lmdb
try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is required for optical flow processing. Install it with `pip install opencv-python` or `pip install opencv-python-headless`."
    ) from e
import numpy as np
import pickle
import io
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloaders.data_utils.lmdb_utils import LMDBUtility

def compute_flow_for_video(args):

    video_lmdb_dir, flow_lmdb_dir, video_name = args
    video_path = os.path.join(video_lmdb_dir, video_name)
    output_path = os.path.join(flow_lmdb_dir, video_name)

    # Skip if already computed
    if Path(output_path).exists():
        return f"skip:{video_name}"

    try:

        util = LMDBUtility(video_path)

        num_frames = util.details["num_frames"]

        if num_frames < 2:
            return f"too_few_frames:{video_name}"

        all_indices = list(range(num_frames))

        frames = util.get_frames(all_indices)

        # Convert frames to grayscale
        gray_frames = [

            cv2.cvtColor(
                np.array(f),
                cv2.COLOR_RGB2GRAY
            )

            for f in frames

        ]

        # Resize to 64x64
        gray_frames = [cv2.resize(f, (64, 64)) for f in gray_frames]

        # Compute optical flow
        flow_frames = []
        prev_frame = gray_frames[0]
        for frame in gray_frames[1:]:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_frames.append(flow)
            prev_frame = frame

        # First frame has no flow, so duplicate the first flow or use zeros
        if flow_frames:
            flow_frames.insert(0, np.zeros_like(flow_frames[0]))
        else:
            flow_frames = [np.zeros((64, 64, 2), dtype=np.float32)]

        # Save to LMDB
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        env = lmdb.open(output_path, map_size=1e9)
        with env.begin(write=True) as txn:
            for i, flow in enumerate(flow_frames):
                flow_bytes = pickle.dumps(flow.astype(np.float32))
                txn.put(f"{i:06d}".encode(), flow_bytes)
            txn.put(b"num_frames", str(num_frames).encode())
        env.close()

        return f"success:{video_name}"

    except Exception as e:
        return f"error:{video_name}:{str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_lmdb_dir", required=True, help="Directory containing video LMDBs")
    parser.add_argument("--flow_lmdb_dir", required=True, help="Output directory for flow LMDBs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.flow_lmdb_dir, exist_ok=True)

    video_names = [d for d in os.listdir(args.video_lmdb_dir) if os.path.isdir(os.path.join(args.video_lmdb_dir, d))]

    tasks = [(args.video_lmdb_dir, args.flow_lmdb_dir, vn) for vn in video_names]

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(compute_flow_for_video, tasks), total=len(tasks)))

    for r in results:
        print(r)