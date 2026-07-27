from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple


def get_video_decode_backend() -> str:
    """
    Decode backend selector (best-effort).

    Env:
      - VLM_VIDEO_DECODE_BACKEND: 'auto' (default) | 'decord' | 'pyav'
    """
    return os.environ.get("VLM_VIDEO_DECODE_BACKEND", "auto").strip().lower()


def ffprobe_video_info(path: str) -> Tuple[int, float, float]:
    """
    Return (n_frames, fps, duration) via ffprobe.
    This is more robust than relying on random-access decoders for long videos.
    """
    cmd1 = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,avg_frame_rate",
        "-of",
        "json",
        path,
    ]
    data = subprocess.check_output(cmd1)
    info = json.loads(data)["streams"][0]
    n_frames = int(info["nb_read_packets"])
    num, den = info["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    cmd2 = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    duration = float(subprocess.check_output(cmd2).decode().strip() or 0.0)
    return n_frames, fps, duration


def save_frames_by_indices_pyav(
    vid_path: str,
    indices: List[int],
    frame_paths: List[str],
    *,
    total_frames: Optional[int] = None,
    desc: Optional[str] = None,
) -> None:
    """
    Robust frame extraction by sequential decoding (PyAV), saving only selected indices.
    This avoids decord/ffmpeg random-seek issues on very long or problematic videos.
    """
    assert len(indices) == len(frame_paths)
    if len(indices) == 0:
        return

    # Map index -> list of output paths (handle duplicates).
    idx2paths: Dict[int, List[str]] = {}
    for idx, p in zip(indices, frame_paths):
        idx = int(idx)
        idx2paths.setdefault(idx, []).append(p)

    needed = set(idx2paths.keys())
    max_needed = max(needed) if needed else -1

    import av  # local import to avoid hard dependency for non-video users
    from tqdm import tqdm

    with av.open(vid_path) as container:
        stream = container.streams.video[0]
        iterator: Iterable = container.decode(stream)
        if total_frames is not None and total_frames > 0:
            iterator = tqdm(iterator, total=total_frames, desc=desc or f"Extracting (pyav): {os.path.basename(vid_path)}")
        elif desc:
            iterator = tqdm(iterator, desc=desc)

        for i, frame in enumerate(iterator):
            if i > max_needed and not needed:
                break
            if i not in idx2paths:
                continue

            img = frame.to_image()
            for pth in idx2paths[i]:
                if not os.path.exists(pth):
                    os.makedirs(os.path.dirname(pth), exist_ok=True)
                    img.save(pth)
            needed.discard(i)
            if not needed:
                break


