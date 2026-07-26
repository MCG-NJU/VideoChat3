import io
import os
import random
import re
import imageio
import av
import cv2
import numpy as np
import math

from dataclasses import dataclass, fields
from typing import Optional, Mapping
from PIL import Image
from transformers.video_utils import VideoMetadata
from xtuner.v1.utils import get_logger

try:
    from decord import VideoReader
except Exception:
    VideoReader = None
    
try:
    import av
except Exception:
    av = None 

logger = get_logger()

# 支持的图像格式（小写，用于大小写不敏感匹配）
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif']

def is_image_file(filename):
    """检查文件是否为支持的图像格式（大小写不敏感）"""
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)

@dataclass
class VideoChat3VideoMetadata(Mapping):
    total_num_frames: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    video_backend: Optional[str] = None
    frames_indices: Optional[list[int]] = None
    video_start_time: float = 0.0 # The start time of the video, in seconds
    clip_start_time: Optional[float] = None # The start time of the video clip to be extracted, in seconds
    clip_end_time: Optional[float] = None # The end time of the video clip to be extracted, in seconds
    specified_wh: Optional[tuple[int, int]] = None # The specified width and height of the video

    def __post_init__(self):
        if self.fps is not None and self.duration is not None:
            expected_frames = self.fps * self.duration
            if abs(expected_frames - self.total_num_frames) > 1e-6:
                raise ValueError(f"fps * duration must be equal to total_num_frames, but got {expected_frames} != {self.total_num_frames}")
            
        # if self.video_start_time < 0 or (self.duration is not None and self.video_start_time >= self.duration):
        #     raise ValueError(f"video_start_time must be greater than or equal to 0 and less than duration, but got {self.video_start_time}")

        if (self.clip_start_time is None) != (self.clip_end_time is None):
            raise ValueError("clip_start_time and clip_end_time must both be None or both be not None.")
        if self.clip_start_time is not None and self.clip_end_time is not None and self.clip_end_time <= self.clip_start_time:
            raise ValueError(f"clip_end_time must be greater than clip_start_time, but got {self.clip_end_time} <= {self.clip_start_time}")
        if self.clip_start_time is not None and self.clip_start_time < 0:
            raise ValueError(f"clip_start_time must be greater than 0, but got {self.clip_start_time}")
        if self.clip_end_time is not None and (self.clip_end_time < 0 or self.clip_end_time > self.duration):
            raise ValueError(f"clip_end_time must be greater than 0 and less than duration, but got {self.clip_end_time}")

    def __iter__(self):
        return (f.name for f in fields(self))

    def __len__(self):
        return len(fields(self))

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    @property
    def timestamps(self) -> float:
        "Timestamps of the sampled frames in seconds."
        if self.fps is None:
            raise ValueError("Cannot infer video `timestamps` when `fps` is None.")
        elif self.frames_indices is None:
            raise ValueError("Cannot infer video `timestamps` when `frames_indices` is None.")
            
        return [self.video_start_time + frame_idx / self.fps for frame_idx in self.frames_indices]

    def update(self, dictionary):
        for key, value in dictionary.items():
            if hasattr(self, key):
                setattr(self, key, value)



def read_frames_gif(
    video_path,
    frame_sample_indices,
    client=None,
    video_meta=None,
):
    byteio = None

    assert VideoReader is not None, "Please install decord: pip install decord"
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        video_reader = imageio.get_reader(byteio)
    else:
        video_reader = imageio.get_reader(video_path)
    
    vlen = len(video_reader)
    new_frame_sample_indices = []
    for idx in frame_sample_indices:
        if idx < vlen:
            new_frame_sample_indices.append(idx)
        else:
            logger.warning(
                f"WARNING: {idx} is out of range for video {video_path} (vlen = {vlen}), use the last frame index {vlen - 1} instead."
            )
            new_frame_sample_indices.append(vlen - 1)


    frames = []
    min_h = min_w = 100000
    hw_set = set()
    for index, frame in enumerate(video_reader):
        # for index in frame_idxs:
        if index in new_frame_sample_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = frame.astype(np.uint8)
            # # (H x W x C) to (C x H x W)
            # frame = frame.permute(2, 0, 1)
            frames.append(frame) # (H x W x C)
            hw_set.add(frame.shape)
            if frame.shape[0] < min_h:
                min_h = frame.shape[0]
            if frame.shape[1] < min_w:
                min_w = frame.shape[1]
    # print(hw_set, min_h, min_w)
    if len(hw_set) > 1:
        frames = [i[:min_h, :min_w] for i in frames]

    frames = [Image.fromarray(frames[i]) for i in range(len(frames))]

    if byteio != None:
        byteio.close()

    return frames


def read_frames_decord(
    video_path,
    frame_sample_indices,
    client=None,
    video_meta=None,
):
    byteio = None
    decord_video_threads = int(os.getenv("XTUNER_DECORD_VIDEO_THREADS", 1))
    if video_path.endswith('.avi'):
        return read_frames_av(video_path, frame_sample_indices, client, video_meta=video_meta)
    assert VideoReader is not None, "Please install decord: pip install decord"
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        video_reader = VideoReader(byteio, num_threads=decord_video_threads)
    else:
        video_reader = VideoReader(video_path, num_threads=decord_video_threads)
    
    vlen = len(video_reader)
    new_frame_sample_indices = []
    for idx in frame_sample_indices:
        if idx < vlen:
            new_frame_sample_indices.append(idx)
        else:
            logger.warning(
                f"WARNING: {idx} is out of range for video {video_path} (vlen = {vlen}), use the last frame index {vlen - 1} instead."
            )
            new_frame_sample_indices.append(vlen - 1)

    frames = video_reader.get_batch(new_frame_sample_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]

    video_reader.seek(0)
    if byteio != None:
        byteio.close()

    return frames


def read_frames_av(
    video_path,
    frame_sample_indices,
    client=None,
    video_meta=None,
):
    assert av is not None, "Please install av: pip install av"
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        byteio.seek(0)
        reader = av.open(byteio)
    else:
        byteio = None
        reader = av.open(video_path)

        reader = reader.streams.video[0]
    ori_frames = [f.to_rgb().to_image() for f in reader.decode(video=0)]
    frames = []
    for idx in frame_sample_indices:
        if idx < len(ori_frames):
            frames.append(ori_frames[idx])
        else:
            logger.warning(
                f"WARNING: {idx} is out of range for video {video_path} (len(ori_frames) = {len(ori_frames)}), use the last frame instead."
            )
            frames.append(ori_frames[-1])

    if byteio != None:
        byteio.close()
        
    reader.close()

    return frames


def read_frames_dir(
    video_path,
    frame_sample_indices,
    client=None,
    video_meta=None,
):
    def extract_frame_number(filename):
        # Extract the numeric part from the filename using regular expressions
        filename_lower = filename.lower()
        if filename_lower.endswith('.jpg') or filename_lower.endswith('.jpeg'):
            match = re.search(r'_(\d+)\.(jpg|jpeg)$', filename_lower)
        elif filename_lower.endswith('.png'):
            match = re.search(r'_(\d+)\.png$', filename_lower)
        elif filename_lower.endswith('.bmp'):
            match = re.search(r'_(\d+)\.bmp$', filename_lower)
        elif filename_lower.endswith('.gif'):
            match = re.search(r'_(\d+)\.gif$', filename_lower)
        elif filename_lower.endswith('.webp'):
            match = re.search(r'_(\d+)\.webp$', filename_lower)
        elif filename_lower.endswith('.tiff') or filename_lower.endswith('.tif'):
            match = re.search(r'_(\d+)\.(tiff|tif)$', filename_lower)
        else:
            raise NotImplementedError(f"Unsupported image format: {filename}")

        return int(match.group(1)) if match else -1

    def sort_frames(frame_paths):
        # Extract filenames from each path and sort by their numeric part
        frame_paths = [x for x in frame_paths if is_image_file(x)]
        return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))

    try:
        if "s3://" in video_path:
            if video_path[-1] != '/':
                video_path = video_path + '/'
            img_list = sort_frames(client.list(video_path))
        else:
            img_list = sort_frames(list(os.listdir(video_path)))
    except Exception as e:
        raise ValueError(f"Meet Error at sort_frames for {video_path} {e}!!!")
    frames = []
    for idx in frame_sample_indices:
        if idx < len(img_list):
            frame_fname = img_list[idx]
        else:
            logger.warning(
                f"WARNING: {idx} is out of range for video {video_path} (len(img_list) = {len(img_list)}), use the last frame instead."
            )
            frame_fname = img_list[-1]
        try:
            if "s3://" in video_path:
                s3_prefix = video_path.split("s3://")[0]
                if '/' in frame_fname or frame_fname.startswith(video_path.split("s3://")[1]):
                    s3_bucket = video_path.split("s3://")[1].split("/")[0]
                else:
                    s3_bucket = video_path.split("s3://")[1]
                    
                frame_fname = os.path.join(s3_prefix+"s3://", s3_bucket, frame_fname)
                img_bytes = client.get(frame_fname)
                frames.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            else:
                frame_fname = os.path.join(video_path, frame_fname)
                frames.append(Image.open(frame_fname).convert("RGB"))
                
        except Exception as e:
            raise ValueError(f"Meet Error at read frames for {video_path}: {frame_fname}!!!")
    return frames


def read_frames_dir2(
    video_path,
    frame_sample_indices,
    client=None,
    video_meta=None,
):

    def sort_frames2(frame_paths):
        # Sort by filename alphabetically (works for fixed-format names like "00865_50.jpg")
        frame_paths = [x for x in frame_paths if is_image_file(x)]
        return sorted(frame_paths, key=lambda x: os.path.basename(x))

    try:
        if "s3://" in video_path:
            if video_path[-1] != '/':
                video_path = video_path + '/'
            img_list = sort_frames2(client.list(video_path))
        else:
            img_list = sort_frames2(list(os.listdir(video_path)))
    except Exception as e:
        raise ValueError(f"Meet Error at sort_frames for {video_path}!!!")
    frames = []
    for idx in frame_sample_indices:
        if idx < len(img_list):
            frame_fname = img_list[idx]
        else:
            logger.warning(
                f"WARNING: {idx} is out of range for video {video_path} (len(img_list) = {len(img_list)}), use the last frame instead."
            )
            frame_fname = img_list[-1]
        try:
            if "s3://" in video_path:
                s3_prefix = video_path.split("s3://")[0]
                if '/' in frame_fname or frame_fname.startswith(video_path.split("s3://")[1]):
                    s3_bucket = video_path.split("s3://")[1].split("/")[0]
                else:
                    s3_bucket = video_path.split("s3://")[1]
                    
                frame_fname = os.path.join(s3_prefix+"s3://", s3_bucket, frame_fname)
                img_bytes = client.get(frame_fname)
                frames.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            else:
                frame_fname = os.path.join(video_path, frame_fname)
                frames.append(Image.open(frame_fname).convert("RGB"))
                
        except Exception as e:
            raise ValueError(f"Meet Error at read frames for {video_path}: {frame_fname}!!!")
    return frames


def read_frames_indexed_jpg(
    video_path,
    frame_sample_indices,
    client=None,
    video_meta=None,
):
    if video_meta is None:
        raise ValueError("indexed_jpg reader requires video_meta.")

    total_num_frames = int(video_meta.total_num_frames)
    if total_num_frames <= 0:
        raise ValueError(f"Invalid total_num_frames for {video_path}: {total_num_frames}")

    if "s3://" in video_path:
        if client is None:
            raise RuntimeError("Ceph client is required for s3 video_path.")
        base_path = video_path.rstrip("/") + "/"
    else:
        base_path = video_path

    frames = []
    for idx in frame_sample_indices:
        idx = int(idx)
        if idx < 0:
            logger.warning(
                f"WARNING: {idx} is negative for video {video_path}, use the first frame instead."
            )
            idx = 0
        elif idx >= total_num_frames:
            logger.warning(
                f"WARNING: {idx} is out of range for video {video_path} "
                f"(total_num_frames = {total_num_frames}), use the last frame instead."
            )
            idx = total_num_frames - 1

        frame_fname = f"{idx + 1:08d}.jpg"
        try:
            if "s3://" in video_path:
                frame_path = base_path + frame_fname
                img_bytes = client.get(frame_path)
                frames.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            else:
                frame_path = os.path.join(base_path, frame_fname)
                frames.append(Image.open(frame_path).convert("RGB"))
        except Exception as e:
            raise ValueError(
                f"Meet Error at read indexed_jpg frame for {frame_path}!!! "
                f"Original error: {type(e).__name__}: {e!r}"
            ) from e

    return frames


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_decord_old(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    client=None,
    clip=None,
    min_num_frames=4,
    random_frame_num=None,
):
    assert VideoReader is not None, "Please install decord: pip install decord"
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames



VIDEO_READER_MAP = {
    "decord": read_frames_decord,
    "gif": read_frames_gif,
    "img": read_frames_dir2,
    "img2": read_frames_dir2,
    "frame": read_frames_dir2,
    "frame2": read_frames_dir2,
    "indexed_jpg": read_frames_indexed_jpg,
    "av": read_frames_av,
}