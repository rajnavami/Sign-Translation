import argparse
import os
import lmdb
import cv2
import numpy as np
import pickle
import io
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

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
        gray_frames = [

            cv2.resize(g,(64,64))

            for g in gray_frames

        ]


        flows = []

        # Compute flow between consecutive frames
        for i in range(len(gray_frames)-1):

            flow = cv2.calcOpticalFlowFarneback(

                gray_frames[i],
                gray_frames[i+1],
                None,

                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0

            )

            # Normalize motion
            flow = np.clip(flow,-20,20)/20.0

            flow = flow.astype(np.float16)

            flows.append(flow)


        # Estimate LMDB map size
        map_size = num_frames * 64 * 64 * 2 * 2 * 2


        env = lmdb.open(

            output_path,

            map_size=map_size

        )


        with env.begin(write=True) as txn:

            details = {

                "num_frames":len(flows)

            }

            txn.put(

                "details".encode("ascii"),

                pickle.dumps(details)

            )


            for idx,flow in enumerate(flows):

                buffer = io.BytesIO()

                np.save(buffer,flow)

                txn.put(

                    f"{idx}".encode("ascii"),

                    buffer.getvalue()

                )


        env.close()

        return f"done:{video_name}"


    except Exception as e:

        return f"error:{video_name}:{str(e)}"



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_lmdb_dir",
        required=True
    )

    parser.add_argument(
        "--flow_lmdb_dir",
        required=True
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )


    args = parser.parse_args()


    Path(args.flow_lmdb_dir).mkdir(

        parents=True,
        exist_ok=True

    )


    video_dirs = sorted([

        d for d in os.listdir(args.video_lmdb_dir)

        if os.path.isdir(
            os.path.join(args.video_lmdb_dir,d)
        )

    ])


    print(f"Found {len(video_dirs)} videos")


    tasks = [

        (
            args.video_lmdb_dir,
            args.flow_lmdb_dir,
            v
        )

        for v in video_dirs

    ]


    with Pool(args.num_workers) as pool:

        for result in tqdm(

            pool.imap_unordered(
                compute_flow_for_video,
                tasks
            ),

            total=len(tasks),
            desc="Computing optical flow"

        ):

            if result.startswith("error"):

                print(result)


    print("Flow preprocessing finished")


if __name__=="__main__":

    main()