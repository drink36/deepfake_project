import os
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from decord import VideoReader, cpu

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(x, hi))


def make_window_indices(
    total_frames: int,
    center: int,
    window_len: int,
    frame_interval: int,
) -> List[int]:
    window_size = (window_len - 1) * frame_interval + 1
    half = window_size // 2

    center = clamp(center, 0, total_frames - 1)
    start_idx = center - half
    start_idx = clamp(start_idx, 0, max(0, total_frames - window_size))

    idxs = [start_idx + i * frame_interval for i in range(window_len)]
    return [clamp(i, 0, total_frames - 1) for i in idxs]


def choose_real_centers(total_frames: int, k: int) -> List[int]:
    """
    real video 取 k 個 center（預設均勻取 25%, 50%, 75%...）
    """
    if k <= 1:
        return [total_frames // 2]

    # 均勻分佈在 (0,1) 上，不取 0%/100% 以免太靠邊
    centers = []
    for i in range(1, k + 1):
        frac = i / (k + 1)  # e.g. k=3 -> 0.25,0.5,0.75
        centers.append(int(frac * (total_frames - 1)))
    return centers


def save_frames_as_images(frames: np.ndarray, out_dir: Path):
    """
    frames: (T, H, W, C) uint8 RGB
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PIL:
        # 沒 PIL 就存 npy（不建議，但至少不中斷）
        for t in range(frames.shape[0]):
            np.save(out_dir / f"{t:03d}.npy", frames[t])
        return

    for t in range(frames.shape[0]):
        Image.fromarray(frames[t]).save(out_dir / f"{t:03d}.jpg", quality=95)


def export(
    data_root: str,
    in_metadata_json: str,
    out_clips_root: str,
    out_metadata_json: str,
    subset: Optional[str] = None,          # 例如 "train"；None=不過濾 split
    window_len: int = 32,
    frame_interval: int = 1,
    real_windows_per_video: int = 3,       # real 每支存幾個 window
    take_num: Optional[int] = None,
):
    data_root = Path(data_root)
    out_clips_root = Path(out_clips_root)
    out_clips_root.mkdir(parents=True, exist_ok=True)

    with open(in_metadata_json, "r") as f:
        meta = json.load(f)

    if subset is not None:
        meta = [m for m in meta if m.get("split") == subset]

    if take_num is not None:
        meta = meta[:take_num]

    new_meta: List[Dict[str, Any]] = []

    for idx, m in enumerate(meta):
        rel_video_path = m["file"]
        video_path = data_root / rel_video_path

        if not video_path.exists():
            print(f"[WARN] missing video: {video_path}")
            continue

        # 你們資料：visual_fake_segments 代表 fake periods（秒）
        fake_periods = m.get("visual_fake_segments", []) or []

        # fps：如果 json 沒有，就用 decord 讀；再不行就 fallback 25
        fps = m.get("fps", None)

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            if total_frames <= 0:
                continue

            if fps is None:
                try:
                    fps = float(vr.get_avg_fps())
                except Exception:
                    fps = 25.0
            else:
                fps = float(fps)

            # 用 video path 當 key，避免撞名
            video_key = rel_video_path.replace("/", "_").replace("\\", "_")

            # -------------------------
            # Fake: 每個 fake_period 存一個 window
            # -------------------------
            if len(fake_periods) > 0:
                for seg_id, seg in enumerate(fake_periods):
                    start_sec, end_sec = float(seg[0]), float(seg[1])
                    s_idx = int(start_sec * fps)
                    e_idx = int(end_sec * fps)
                    center = (s_idx + e_idx) // 2

                    window_indices = make_window_indices(
                        total_frames=total_frames,
                        center=center,
                        window_len=window_len,
                        frame_interval=frame_interval,
                    )

                    frames = vr.get_batch(window_indices).asnumpy()  # (32,H,W,3)

                    # 輸出路徑：clips_root/<subset or split>/<video_key>/seg_XXX/
                    split_name = subset if subset is not None else (m.get("split") or "unknown")
                    out_dir = out_clips_root / split_name / video_key / f"seg_{seg_id:03d}"
                    save_frames_as_images(frames, out_dir)

                    new_meta.append({
                        "split": split_name,
                        "label": 1,
                        "clip_dir": str(out_dir.relative_to(out_clips_root)).replace("\\", "/"),
                        "source_video": rel_video_path,
                        "seg_id": seg_id,
                        "fake_start_sec": start_sec,
                        "fake_end_sec": end_sec,
                        "fps": fps,
                        "window_len": window_len,
                        "frame_interval": frame_interval,
                    })

            # -------------------------
            # Real: 每支影片存 K 個 window（均勻取 center）
            # -------------------------
            else:
                split_name = subset if subset is not None else (m.get("split") or "unknown")
                centers = choose_real_centers(total_frames, real_windows_per_video)

                for k_id, center in enumerate(centers):
                    window_indices = make_window_indices(
                        total_frames=total_frames,
                        center=center,
                        window_len=window_len,
                        frame_interval=frame_interval,
                    )
                    frames = vr.get_batch(window_indices).asnumpy()

                    out_dir = out_clips_root / split_name / video_key / f"real_{k_id:03d}"
                    save_frames_as_images(frames, out_dir)

                    new_meta.append({
                        "split": split_name,
                        "label": 0,
                        "clip_dir": str(out_dir.relative_to(out_clips_root)).replace("\\", "/"),
                        "source_video": rel_video_path,
                        "seg_id": None,
                        "fake_start_sec": None,
                        "fake_end_sec": None,
                        "fps": fps,
                        "window_len": window_len,
                        "frame_interval": frame_interval,
                    })

            if (idx + 1) % 200 == 0:
                print(f"[INFO] processed {idx+1}/{len(meta)} videos, windows so far={len(new_meta)}")

        except Exception as e:
            print(f"[ERR] {video_path}: {e}")
            continue

    with open(out_metadata_json, "w") as f:
        json.dump(new_meta, f, indent=2)

    print(f"[DONE] windows={len(new_meta)} saved to {out_clips_root}")
    print(f"[DONE] new metadata -> {out_metadata_json}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--in_metadata", type=str, required=True)
    ap.add_argument("--out_clips_root", type=str, required=True)
    ap.add_argument("--out_metadata", type=str, required=True)

    ap.add_argument("--subset", type=str, default=None)  # e.g. train/val/test
    ap.add_argument("--window_len", type=int, default=32)
    ap.add_argument("--frame_interval", type=int, default=1)
    ap.add_argument("--real_k", type=int, default=3)
    ap.add_argument("--take_num", type=int, default=None)

    args = ap.parse_args()

    export(
        data_root=args.data_root,
        in_metadata_json=args.in_metadata,
        out_clips_root=args.out_clips_root,
        out_metadata_json=args.out_metadata,
        subset=args.subset,
        window_len=args.window_len,
        frame_interval=args.frame_interval,
        real_windows_per_video=args.real_k,
        take_num=args.take_num,
    )
