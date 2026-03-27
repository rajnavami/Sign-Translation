import argparse
import os
import io
import pickle
import numpy as np
import mediapipe as mp
from PIL import Image
from tqdm import tqdm
import lmdb
import multiprocessing as python_mp

mp_hands = mp.solutions.hands


def extract_landmarks_for_video(lmdb_path, num_frames, num_hands):
    # Each worker creates its own MediaPipe instance (not picklable, so can't share)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarks_out = np.zeros((num_frames, num_hands * 63), dtype=np.float32)

    with lmdb.open(
        path=lmdb_path,
        readonly=True,
        readahead=False,
        lock=False,
        meminit=False,
    ).begin(write=False) as txn:
        for idx in range(num_frames):
            raw = txn.get(f"{idx}".encode("ascii"))
            if raw is None:
                continue
            frame_np = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            results = hands.process(frame_np)

            # ── OLD CODE (no handedness awareness, incorrect slot assignment) ──
            # embeddings = []
            # if results.multi_hand_landmarks:
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         embedding = []
            #         for lm in hand_landmarks.landmark:
            #             embedding.extend([lm.x, lm.y, lm.z])
            #         embeddings.append(np.array(embedding))  # 63-dim
            # while len(embeddings) < num_hands:
            #     embeddings.append(np.zeros(63, dtype=np.float32))
            # landmarks_out[idx] = np.concatenate(embeddings[:num_hands])
            # ── END OLD CODE ──────────────────────────────────────────────────

            # ── NEW CODE (correct left/right assignment via MediaPipe handedness) ──
            # Phoenix14T is a broadcast camera — NOT mirrored
            # MediaPipe "Left"  = signer's left  → slot 0 (first  63 values)
            # MediaPipe "Right" = signer's right → slot 1 (second 63 values)
            left_hand  = np.zeros(63, dtype=np.float32)
            right_hand = np.zeros(63, dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    label = handedness.classification[0].label
                    coords = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in hand_lm.landmark
                    ]).flatten()  # 63-dim

                    if label == "Left":
                        right_hand = coords   # MediaPipe "Left" = signer's right
                    else:
                        left_hand  = coords   # MediaPipe "Right" = signer's left

            landmarks_out[idx] = np.concatenate([left_hand, right_hand])
            # ── END NEW CODE ──────────────────────────────────────────────────

    hands.close()
    return landmarks_out


def process_video(args):
    """Worker function — called once per video by the multiprocessing pool."""
    lmdb_dir, video_name, num_hands = args
    lmdb_path = os.path.join(lmdb_dir, video_name)
    out_path = os.path.join(lmdb_dir, f"{video_name}_hand_landmarks.npy")

    if os.path.exists(out_path):
        return f"skip:{video_name}"

    try:
        with lmdb.open(
            path=lmdb_path, readonly=True, readahead=False, lock=False, meminit=False
        ).begin(write=False) as txn:
            details = pickle.loads(txn.get(b"details"))
        num_frames = details["num_frames"]
    except Exception as e:
        return f"error:{video_name}:{e}"

    landmarks = extract_landmarks_for_video(lmdb_path, num_frames, num_hands)
    np.save(out_path, landmarks)
    return f"done:{video_name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_dir", required=True, help="Directory containing per-video LMDB folders")
    parser.add_argument("--num_hands", type=int, default=2, help="Max hands to track per frame")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Parallel worker processes (default: number of CPU cores)")
    args = parser.parse_args()

    num_workers = args.num_workers or python_mp.cpu_count()
    print(f"Using {num_workers} worker processes")

    video_dirs = sorted(
        [d for d in os.listdir(args.lmdb_dir)
         if os.path.isdir(os.path.join(args.lmdb_dir, d))]
    )
    print(f"Found {len(video_dirs)} video LMDBs in {args.lmdb_dir}")

    tasks = [(args.lmdb_dir, v, args.num_hands) for v in video_dirs]

    with python_mp.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_video, tasks),
                           total=len(tasks), desc="Extracting hand landmarks"):
            if result.startswith("error:"):
                print(f"\n  ⚠️  {result}")

    print("Done. Landmark files saved as <video_name>_hand_landmarks.npy")


if __name__ == "__main__":
    main()
