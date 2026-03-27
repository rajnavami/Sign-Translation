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

mp_holistic = mp.solutions.holistic

# ── Keypoint indices ──────────────────────────────────────────────────────────
# Upper body only (no legs) — 25 joints from MediaPipe pose (0-24)
BODY_JOINTS = list(range(25))

# 37 key face landmarks — subset of MediaPipe 468 face mesh
# Covers eyes, eyebrows, nose, mouth, face contour
FACE_JOINTS = [
    # Face contour
    10, 234, 454, 152,
    # Left eye
    33, 133, 159, 145,
    # Right eye
    362, 263, 386, 374,
    # Eyebrows
    70, 63, 105, 66, 107,
    300, 293, 334, 296, 336,
    # Nose
    1, 2, 5, 4, 6,
    # Mouth outer
    61, 291, 39, 181, 269, 405,
    # Mouth inner
    13, 14, 78, 308,
]
assert len(FACE_JOINTS) == 37, f"Expected 37 face joints, got {len(FACE_JOINTS)}"

# Total keypoints: 25 body + 21 left hand + 21 right hand + 37 face = 104
# Using x,y only (not z) → 104 × 2 = 208 dims per frame
TOTAL_DIMS = (len(BODY_JOINTS) + 21 + 21 + len(FACE_JOINTS)) * 2  # 208


def _local_normalize(kps_2d):
    """
    Local normalization for hands and face — scale each to [-0.9, 0.9]
    maintaining aspect ratio with 10% border (Paper 1 SignSpace method).

    kps_2d: N × 2  (landmarks for one hand or face, one frame)
    Returns: N × 2 normalized, or zeros if all zeros (not detected)
    """
    if kps_2d.sum() == 0:
        return kps_2d  # not detected — leave as zeros

    mins = kps_2d.min(axis=0)   # 2
    maxs = kps_2d.max(axis=0)   # 2
    ranges = np.maximum(maxs - mins, 1e-6)

    # Scale to [0, 1] then to [-0.9, 0.9] (10% border each side)
    normalized = (kps_2d - mins) / ranges   # [0, 1]
    normalized = normalized * 1.8 - 0.9     # [-0.9, 0.9]
    return normalized


def _signspace_normalize(body_kps, left_hand_kps, right_hand_kps, face_kps):
    """
    Paper 1 SignSpace two-level normalization:
    - Body: global normalization relative to shoulder center and width
    - Hands + Face: local normalization independently to [-0.9, 0.9]

    body_kps:       25 × 2
    left_hand_kps:  21 × 2
    right_hand_kps: 21 × 2
    face_kps:       37 × 2

    Returns: (body_norm, lh_norm, rh_norm, face_norm) all same shapes
    """
    # ── Global body normalization ──────────────────────────────────────────
    # MediaPipe pose: joint 11 = left shoulder, joint 12 = right shoulder
    left_shoulder  = body_kps[11]   # 2
    right_shoulder = body_kps[12]   # 2

    shoulder_center = (left_shoulder + right_shoulder) / 2   # 2
    shoulder_width  = np.linalg.norm(left_shoulder - right_shoulder)
    shoulder_width  = max(shoulder_width, 1e-6)

    # Signing space = 3× shoulder width, centered on shoulder midpoint
    signing_space = 3 * shoulder_width

    body_norm = (body_kps - shoulder_center) / signing_space   # 25 × 2

    # ── Local hand and face normalization ─────────────────────────────────
    lh_norm   = _local_normalize(left_hand_kps)    # 21 × 2
    rh_norm   = _local_normalize(right_hand_kps)   # 21 × 2
    face_norm = _local_normalize(face_kps)          # 37 × 2

    return body_norm, lh_norm, rh_norm, face_norm


def _interpolate_missing(landmarks, max_gap=5):
    """
    Linear interpolation for missing keypoints (all zeros).
    Only interpolates when gap between detections <= max_gap frames.
    Paper 1 finding: interpolation significantly improves robustness.

    landmarks: T × D
    Returns:   T × D with interpolated values
    """
    T, D = landmarks.shape
    result = landmarks.copy()

    # Missing = entire frame is zeros (no detection at all)
    missing = (landmarks.sum(axis=1) == 0)

    if not missing.any():
        return result

    i = 0
    while i < T:
        if missing[i]:
            # Find end of this missing run
            j = i
            while j < T and missing[j]:
                j += 1

            gap = j - i

            # Only interpolate if gap is small and we have both endpoints
            if gap <= max_gap and i > 0 and j < T:
                for k in range(gap):
                    alpha = (k + 1) / (gap + 1)
                    result[i + k] = (1 - alpha) * landmarks[i - 1] + \
                                     alpha       * landmarks[j]
            i = j
        else:
            i += 1

    return result


def extract_holistic_landmarks(lmdb_path, num_frames):
    """
    Extract 208-dim holistic pose features per frame using MediaPipe Holistic.
    Applies SignSpace normalization and missing keypoint interpolation.

    Output shape: T × 208
    Layout: [body(50) | left_hand(42) | right_hand(42) | face(74)]
    """
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,              # 0=lite, 1=full, 2=heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=False,     # not needed for 468-mesh subset
    )

    landmarks_out = np.zeros((num_frames, TOTAL_DIMS), dtype=np.float32)

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
            results  = holistic.process(frame_np)

            # ── Body pose — 25 upper body joints × 2 (x, y only) ─────────
            body = np.zeros((25, 2), dtype=np.float32)
            if results.pose_landmarks:
                for i, j in enumerate(BODY_JOINTS):
                    lm = results.pose_landmarks.landmark[j]
                    body[i, 0] = lm.x
                    body[i, 1] = lm.y

            # ── Left hand — 21 joints × 2 ────────────────────────────────
            # Phoenix14T: broadcast camera, NOT mirrored
            # MediaPipe "left_hand_landmarks" = signer's left hand → slot 0
            left_hand = np.zeros((21, 2), dtype=np.float32)
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    left_hand[i, 0] = lm.x
                    left_hand[i, 1] = lm.y

            # ── Right hand — 21 joints × 2 ───────────────────────────────
            # MediaPipe "right_hand_landmarks" = signer's right hand → slot 1
            right_hand = np.zeros((21, 2), dtype=np.float32)
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    right_hand[i, 0] = lm.x
                    right_hand[i, 1] = lm.y

            # ── Face — 37 key landmarks × 2 ──────────────────────────────
            face = np.zeros((37, 2), dtype=np.float32)
            if results.face_landmarks:
                for i, j in enumerate(FACE_JOINTS):
                    lm = results.face_landmarks.landmark[j]
                    face[i, 0] = lm.x
                    face[i, 1] = lm.y

            # ── SignSpace normalization ───────────────────────────────────
            # Only normalize if body detected (need shoulders for signing space)
            if results.pose_landmarks:
                body_n, lh_n, rh_n, face_n = _signspace_normalize(
                    body, left_hand, right_hand, face
                )
            else:
                # No body detected — use local normalization only
                body_n  = body
                lh_n    = _local_normalize(left_hand)
                rh_n    = _local_normalize(right_hand)
                face_n  = _local_normalize(face)

            # ── Concatenate: [body | left_hand | right_hand | face] ───────
            landmarks_out[idx] = np.concatenate([
                body_n.flatten(),    # 50
                lh_n.flatten(),      # 42
                rh_n.flatten(),      # 42
                face_n.flatten(),    # 74
            ])                       # total: 208

    holistic.close()

    # ── Interpolate missing frames ────────────────────────────────────────
    landmarks_out = _interpolate_missing(landmarks_out, max_gap=5)

    return landmarks_out


def process_video(args):
    """Worker function — called once per video by the multiprocessing pool."""
    lmdb_dir, video_name = args
    lmdb_path = os.path.join(lmdb_dir, video_name)
    out_path  = os.path.join(lmdb_dir, f"{video_name}_hand_landmarks.npy")

    if os.path.exists(out_path):
        return f"skip:{video_name}"

    try:
        with lmdb.open(
            path=lmdb_path, readonly=True,
            readahead=False, lock=False, meminit=False,
        ).begin(write=False) as txn:
            details = pickle.loads(txn.get(b"details"))
        num_frames = details["num_frames"]
    except Exception as e:
        return f"error:{video_name}:{e}"

    landmarks = extract_holistic_landmarks(lmdb_path, num_frames)
    np.save(out_path, landmarks)
    return f"done:{video_name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lmdb_dir", required=True,
        help="Directory containing per-video LMDB folders"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Parallel worker processes (default: number of CPU cores)"
    )
    args = parser.parse_args()

    num_workers = args.num_workers or python_mp.cpu_count()
    print(f"Using {num_workers} worker processes")
    print(f"Output: {TOTAL_DIMS}-dim holistic features per frame")
    print(f"  Body:       {len(BODY_JOINTS) * 2} dims (25 joints × xy)")
    print(f"  Left hand:  {21 * 2} dims (21 joints × xy)")
    print(f"  Right hand: {21 * 2} dims (21 joints × xy)")
    print(f"  Face:       {len(FACE_JOINTS) * 2} dims (37 joints × xy)")

    video_dirs = sorted([
        d for d in os.listdir(args.lmdb_dir)
        if os.path.isdir(os.path.join(args.lmdb_dir, d))
    ])
    print(f"Found {len(video_dirs)} video LMDBs in {args.lmdb_dir}")

    tasks = [(args.lmdb_dir, v) for v in video_dirs]

    with python_mp.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_video, tasks),
            total=len(tasks),
            desc="Extracting holistic landmarks"
        ):
            if result.startswith("error:"):
                print(f"\n  ⚠️  {result}")

    print(f"Done. Files saved as <video_name>_hand_landmarks.npy  shape: T × {TOTAL_DIMS}")


if __name__ == "__main__":
    main()