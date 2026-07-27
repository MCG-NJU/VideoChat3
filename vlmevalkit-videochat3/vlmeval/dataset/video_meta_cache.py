import json
import hashlib
import os
import time
from typing import Callable, Dict, Iterable, Optional

from .utils.video_pyav import ffprobe_video_info


META_VERSION = 1


def _atomic_dump_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        os.replace(tmp, path)
    except OSError:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        try:
            os.remove(tmp)
        except OSError:
            pass


def load_video_meta(meta_path: Optional[str]) -> Dict[str, dict]:
    if not meta_path or not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        videos = obj.get('videos', obj)
        if isinstance(videos, dict):
            return {str(k): v for k, v in videos.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def probe_video_meta(video_path: str, prefer_pyav: bool = False) -> dict:
    backend = 'decord'
    if not prefer_pyav:
        try:
            import decord

            vr = decord.VideoReader(video_path, num_threads=1)
            n_frames = int(len(vr))
            fps = float(vr.get_avg_fps())
            duration = float(n_frames / fps) if fps > 0 else 0.0
            stat = os.stat(video_path)
            return {
                'path': video_path,
                'duration': duration,
                'fps': fps,
                'n_frames': n_frames,
                'backend': backend,
                'size': int(stat.st_size),
                'mtime': float(stat.st_mtime),
            }
        except Exception:
            pass

    n_frames, fps, duration = ffprobe_video_info(video_path)
    stat = os.stat(video_path)
    return {
        'path': video_path,
        'duration': float(duration),
        'fps': float(fps),
        'n_frames': int(n_frames),
        'backend': 'pyav',
        'size': int(stat.st_size),
        'mtime': float(stat.st_mtime),
    }


def _meta_is_usable(meta: dict) -> bool:
    try:
        return (
            float(meta.get('duration', 0.0)) > 0
            and float(meta.get('fps', 0.0)) > 0
            and int(meta.get('n_frames', 0)) > 0
        )
    except Exception:
        return False


def ensure_video_meta_json(
    meta_path: str,
    video_ids: Iterable[str],
    resolve_video_path: Callable[[str], str],
    prefer_pyav: bool = False,
) -> Dict[str, dict]:
    video_ids = sorted({str(x) for x in video_ids if str(x)})
    existing = load_video_meta(meta_path)
    if all(_meta_is_usable(existing.get(video_id, {})) for video_id in video_ids):
        return existing

    lock_dir = os.environ.get('VLMEVAL_VIDEO_META_LOCK_DIR', '/tmp/vlmeval_video_meta_locks')
    os.makedirs(lock_dir, exist_ok=True)
    lock_key = hashlib.md5(meta_path.encode('utf-8')).hexdigest()
    lock_path = os.path.join(lock_dir, f'{lock_key}.lock')
    try:
        import portalocker

        lock_ctx = portalocker.Lock(lock_path, 'w', timeout=1800)
    except Exception:
        lock_ctx = None

    if lock_ctx is None:
        return _build_missing_video_meta(meta_path, video_ids, resolve_video_path, existing, prefer_pyav)

    try:
        with lock_ctx:
            existing = load_video_meta(meta_path)
            if all(_meta_is_usable(existing.get(video_id, {})) for video_id in video_ids):
                return existing
            return _build_missing_video_meta(meta_path, video_ids, resolve_video_path, existing, prefer_pyav)
    except Exception:
        return _build_missing_video_meta(meta_path, video_ids, resolve_video_path, existing, prefer_pyav)


def _build_missing_video_meta(
    meta_path: str,
    video_ids: list[str],
    resolve_video_path: Callable[[str], str],
    existing: Dict[str, dict],
    prefer_pyav: bool,
) -> Dict[str, dict]:
    videos = dict(existing)
    errors = {}
    for video_id in video_ids:
        if _meta_is_usable(videos.get(video_id, {})):
            continue
        try:
            videos[video_id] = probe_video_meta(resolve_video_path(video_id), prefer_pyav=prefer_pyav)
        except Exception as exc:
            errors[video_id] = repr(exc)

    obj = {
        '__version__': META_VERSION,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'meta_path': meta_path,
        'videos': videos,
        'errors': errors,
    }
    _atomic_dump_json(obj, meta_path)
    return videos


def sample_frame_count_from_meta(meta: dict, fps: float, nframe: int, frames_limit: int) -> int:
    total = int(meta.get('n_frames', 0))
    video_fps = float(meta.get('fps', 0.0))
    duration = float(meta.get('duration', 0.0))
    if nframe > 0 and fps < 0:
        return int(nframe)
    if fps > 0:
        required_frames = int(duration * fps) if duration > 0 else 0
        if required_frames > frames_limit:
            return int(frames_limit)
        return max(1, int(required_frames))
    raise ValueError('fps and nframe should be set at least one valid value')


def sample_indices_from_meta(meta: dict, fps: float, nframe: int, frames_limit: int) -> list[int]:
    total = int(meta.get('n_frames', 0))
    video_fps = float(meta.get('fps', 0.0))
    duration = float(meta.get('duration', 0.0))
    if total <= 0 or video_fps <= 0:
        return []
    if nframe > 0 and fps < 0:
        step_size = total / (nframe + 1)
        indices = [int(i * step_size) for i in range(1, nframe + 1)]
    elif fps > 0:
        required_frames = int(duration * fps) if duration > 0 else 0
        if required_frames > frames_limit:
            step_size = total / (frames_limit + 1)
            indices = [int(i * step_size) for i in range(1, frames_limit + 1)]
        else:
            step_size = video_fps / fps
            indices = [int(i * step_size) for i in range(max(1, required_frames))]
    else:
        raise ValueError('fps and nframe should be set at least one valid value')
    return [min(max(0, idx), total - 1) for idx in indices]


def sample_fps_from_meta(meta: dict, sampled_frames: int, fps: float, frames_limit: int) -> float:
    duration = float(meta.get('duration', 0.0))
    required_frames = int(duration * fps) if duration > 0 and fps > 0 else 0
    if required_frames > frames_limit and sampled_frames == frames_limit and duration > 0:
        return float(frames_limit / duration)
    return float(fps)
