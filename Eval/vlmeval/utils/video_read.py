import math
import os
import random
import re
from typing import Optional

import numpy as np


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpeg', '.mpg', '.m4v', '.gif'}
SUPPORTED_VIDEO_READ_TYPES = {'av', 'decord', 'gif', 'img', 'frame'}


def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    if pts == math.inf:
        return math.inf
    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader) -> float:
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time,
    )
    return float(video_duration)


def infer_video_read_type(video_path, video_read_type: Optional[str] = None) -> str:
    if video_read_type is not None:
        normalized = str(video_read_type).strip().lower()
        if normalized == 'frame':
            normalized = 'img'
        if normalized not in SUPPORTED_VIDEO_READ_TYPES:
            raise ValueError(
                f'Unsupported video_read_type={video_read_type!r}. '
                f'Expected one of {sorted(SUPPORTED_VIDEO_READ_TYPES)}.'
            )
        return normalized

    video_path = os.fspath(video_path)
    if os.path.isdir(video_path):
        return 'img'

    suffix = os.path.splitext(video_path)[1].lower()
    if suffix == '.gif':
        return 'gif'
    if suffix == '.avi':
        return 'av'
    if suffix in VIDEO_EXTENSIONS:
        return 'decord'

    raise ValueError(
        f'Unable to infer video reader for path={video_path!r}. '
        f'Supported local media are directories and {sorted(VIDEO_EXTENSIONS)}.'
    )


def _normalize_min_num_frames(vlen, min_num_frames, local_num_frames, dynamic_fps=None, mode='video'):
    if min_num_frames <= vlen:
        return int(min_num_frames)

    if dynamic_fps is not None:
        if mode == 'img':
            return int(math.ceil(float(vlen) / local_num_frames) * local_num_frames)
        return int((int(float(vlen)) // local_num_frames) * local_num_frames)

    return int(vlen)


def _resolve_dynamic_num_frames(duration, min_num_frames, max_num_frames, local_num_frames, dynamic_fps, mode):
    scaled_duration = float(duration) * float(dynamic_fps)
    if mode == 'img':
        num_segments = math.ceil((scaled_duration + 1) / local_num_frames)
        num_frames = local_num_frames if num_segments == 0 else local_num_frames * num_segments
        if max_num_frames > 0:
            num_frames = min(num_frames, max_num_frames)
        num_frames = max(min_num_frames, num_frames)
        return max(2, int(num_frames))

    num_segments = int((scaled_duration + 1) // local_num_frames)
    num_frames = local_num_frames if num_segments == 0 else local_num_frames * num_segments
    if max_num_frames > 0:
        num_frames = min(num_frames, max_num_frames)
    return int(max(min_num_frames, num_frames))


def get_frame_indices(
    num_frames,
    vlen,
    sample='middle',
    fix_start=None,
    input_fps=1,
    min_num_frames=1,
    max_num_frames=-1,
    local_num_frames=8,
    dynamic_fps: Optional[float] = None,
    mode='video',
):
    if dynamic_fps is not None and dynamic_fps <= 0:
        raise ValueError(f'dynamic_fps must be positive, got {dynamic_fps}')

    min_num_frames = _normalize_min_num_frames(
        vlen=vlen,
        min_num_frames=min_num_frames,
        local_num_frames=local_num_frames,
        dynamic_fps=dynamic_fps,
        mode=mode,
    )

    if dynamic_fps is not None:
        duration = float(vlen) / float(input_fps)
        num_frames = _resolve_dynamic_num_frames(
            duration=duration,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames,
            local_num_frames=local_num_frames,
            dynamic_fps=dynamic_fps,
            mode=mode,
        )
        sample = 'middle'
    else:
        num_frames = max(min_num_frames, int(num_frames))

    vlen = int(vlen)
    if sample in ['rand', 'middle']:
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = [(interv, intervals[idx + 1] - 1) for idx, interv in enumerate(intervals[:-1])]

        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        else:
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]

        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
        return frame_indices

    if isinstance(sample, str) and sample.startswith('fps'):
        output_fps = float(sample[3:])
        duration = float(vlen) / float(input_fps)
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
        return frame_indices

    raise ValueError(f'Not support sample type: {sample}')


def read_frames_av(
    video_path,
    num_frames,
    sample='rand',
    fix_start=None,
    min_num_frames=1,
    max_num_frames=-1,
    clip=None,
    local_num_frames=8,
    dynamic_fps: Optional[float] = None,
):
    if clip is not None:
        raise NotImplementedError('The av reader does not support clip sampling.')

    import av

    reader = av.open(video_path)
    try:
        frames = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
        vlen = len(frames)
        duration = get_pyav_video_duration(reader)
        fps = vlen / float(duration)
        frame_indices = get_frame_indices(
            num_frames=num_frames,
            vlen=vlen,
            sample=sample,
            fix_start=fix_start,
            input_fps=fps,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames,
            local_num_frames=local_num_frames,
            dynamic_fps=dynamic_fps,
            mode='video',
        )
        frames = np.stack([frames[idx] for idx in frame_indices])
        return frames, frame_indices, float(fps), duration
    finally:
        reader.close()


def read_frames_gif(
    video_path,
    num_frames,
    sample='rand',
    fix_start=None,
    min_num_frames=1,
    max_num_frames=-1,
    clip=None,
    local_num_frames=8,
    dynamic_fps: Optional[float] = None,
):
    if clip is not None:
        raise NotImplementedError('The gif reader does not support clip sampling.')

    import cv2
    import imageio

    gif = imageio.get_reader(video_path)
    try:
        vlen = len(gif)
        fps = 1.0
        duration = vlen / fps
        frame_indices = get_frame_indices(
            num_frames=num_frames,
            vlen=vlen,
            sample=sample,
            fix_start=fix_start,
            input_fps=fps,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames,
            local_num_frames=local_num_frames,
            dynamic_fps=dynamic_fps,
            mode='video',
        )
        frames = []
        real_frame_indices = []
        min_h = min_w = 100000
        hw_set = set()
        for index, frame in enumerate(gif):
            if index not in frame_indices:
                continue
            real_frame_indices.append(index)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = frame.astype(np.uint8)
            frames.append(frame)
            hw_set.add(frame.shape)
            min_h = min(min_h, frame.shape[0])
            min_w = min(min_w, frame.shape[1])

        if len(hw_set) > 1:
            frames = [frame[:min_h, :min_w] for frame in frames]
        return np.stack(frames), real_frame_indices, float(fps), duration
    finally:
        gif.close()


def read_frames_decord(
    video_path,
    num_frames,
    sample='rand',
    fix_start=None,
    min_num_frames=1,
    max_num_frames=-1,
    clip=None,
    local_num_frames=8,
    dynamic_fps: Optional[float] = None,
):
    video_path = os.fspath(video_path)
    if video_path.lower().endswith('.avi'):
        return read_frames_av(
            video_path=video_path,
            num_frames=num_frames,
            sample=sample,
            fix_start=fix_start,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames,
            clip=clip,
            local_num_frames=local_num_frames,
            dynamic_fps=dynamic_fps,
        )

    import decord

    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if clip is not None:
        start, end = float(clip[0]), float(clip[1])
        start = max(0.0, start)
        end = min(max(duration - 0.1, 0.0), end)
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)
    else:
        start_index = 0

    frame_indices = get_frame_indices(
        num_frames=num_frames,
        vlen=vlen,
        sample=sample,
        fix_start=fix_start,
        input_fps=fps,
        min_num_frames=min_num_frames,
        max_num_frames=max_num_frames,
        local_num_frames=local_num_frames,
        dynamic_fps=dynamic_fps,
        mode='video',
    )
    if clip is not None:
        frame_indices = [frame_index + start_index for frame_index in frame_indices]

    frames = video_reader.get_batch(frame_indices).asnumpy()
    video_reader.seek(0)
    return frames, frame_indices, float(fps), duration


def _filter_image_files(frame_names):
    return [name for name in frame_names if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS]


def _extract_frame_number(filename):
    if '_' not in filename:
        match = re.search(r'(\d+)', filename)
    elif filename.endswith('.jpg'):
        match = re.search(r'_(\d+).jpg$', filename)
    elif filename.endswith('.jpeg'):
        match = re.search(r'_(\d+).jpeg$', filename)
    elif filename.endswith('.png'):
        match = re.search(r'_(\d+).png$', filename)
    else:
        raise NotImplementedError(f'Unsupported frame filename: {filename}')
    return int(match.group(1)) if match else -1


def _sort_frame_names(frame_names):
    frame_names = _filter_image_files(frame_names)
    return sorted(frame_names, key=lambda x: _extract_frame_number(os.path.basename(x)))


def _infer_dir_fps(video_path, fps=None):
    if fps is not None:
        return float(fps)
    lowered = str(video_path).lower()
    if 'tvqa' in lowered:
        return 3.0
    if 'got-10k' in lowered:
        return 10.0
    if 'lasot' in lowered:
        return 30.0
    return 1.0


def _get_img_clip_indices(start_sec, end_sec, num_segments, fps, max_frame):
    start_idx = max(0, round(start_sec * fps))
    end_idx = min(round(end_sec * fps), max_frame)
    seg_size = float(end_idx - start_idx) / (num_segments - 1)
    return np.array([start_idx + int(np.round(seg_size * idx)) for idx in range(num_segments)])


def read_frames_img(
    video_path,
    num_frames,
    sample='rand',
    fix_start=None,
    min_num_frames=1,
    max_num_frames=-1,
    clip=None,
    local_num_frames=8,
    fps=None,
    dynamic_fps: Optional[float] = None,
):
    import cv2

    img_list = _sort_frame_names(list(os.listdir(video_path)))
    fps = _infer_dir_fps(video_path, fps=fps)

    if clip is not None:
        start = max(0.0, float(clip[0]))
        end = min((len(img_list) - 1) / fps, float(clip[1]))
        vlen = (end - start) * fps + 1
    else:
        vlen = len(img_list)

    duration = (vlen - 1) / fps
    min_num_frames = _normalize_min_num_frames(
        vlen=vlen,
        min_num_frames=min_num_frames,
        local_num_frames=local_num_frames,
        dynamic_fps=dynamic_fps,
        mode='img',
    )

    if dynamic_fps is not None:
        num_frames = _resolve_dynamic_num_frames(
            duration=duration,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames,
            local_num_frames=local_num_frames,
            dynamic_fps=dynamic_fps,
            mode='img',
        )
    else:
        num_frames = max(min_num_frames, int(num_frames))

    num_frames = int(num_frames)
    if clip is not None:
        frame_indices = _get_img_clip_indices(
            start_sec=float(clip[0]),
            end_sec=float(clip[1]),
            num_segments=num_frames,
            fps=fps,
            max_frame=len(img_list) - 1,
        )
    else:
        frame_indices = get_frame_indices(
            num_frames=num_frames,
            vlen=vlen,
            sample=sample,
            fix_start=fix_start,
            input_fps=fps,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames,
            local_num_frames=local_num_frames,
            dynamic_fps=dynamic_fps,
            mode='img',
        )

    imgs = []
    for idx in frame_indices:
        frame_path = os.path.join(video_path, img_list[int(idx)])
        with open(frame_path, 'rb') as file:
            img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)

    return np.array(imgs, dtype=np.uint8), list(map(int, frame_indices)), float(fps), float(duration)


def read_frames_auto(
    video_path,
    num_frames,
    video_read_type: Optional[str] = None,
    sample='middle',
    fix_start=None,
    min_num_frames=1,
    max_num_frames=-1,
    clip=None,
    local_num_frames=1,
    fps=None,
    dynamic_fps: Optional[float] = None,
):
    reader_type = infer_video_read_type(video_path, video_read_type=video_read_type)
    reader = VIDEO_READER_FUNCS[reader_type]
    reader_kwargs = dict(
        video_path=video_path,
        num_frames=num_frames,
        sample=sample,
        fix_start=fix_start,
        min_num_frames=min_num_frames,
        max_num_frames=max_num_frames,
        clip=clip,
        local_num_frames=local_num_frames,
        dynamic_fps=dynamic_fps,
    )
    if reader_type in {'img', 'frame'} and fps is not None:
        reader_kwargs['fps'] = fps
    return reader(**reader_kwargs)


VIDEO_READER_FUNCS = {
    'av': read_frames_av,
    'decord': read_frames_decord,
    'gif': read_frames_gif,
    'img': read_frames_img,
    'frame': read_frames_img,
}


__all__ = [
    'IMAGE_EXTENSIONS',
    'VIDEO_EXTENSIONS',
    'SUPPORTED_VIDEO_READ_TYPES',
    'VIDEO_READER_FUNCS',
    'pts_to_secs',
    'get_pyav_video_duration',
    'infer_video_read_type',
    'get_frame_indices',
    'read_frames_av',
    'read_frames_decord',
    'read_frames_gif',
    'read_frames_img',
    'read_frames_auto',
]
