from __future__ import annotations

import math
import torch
from PIL import Image
import os
from typing import Iterable

# from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    force_resize: bool = False,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    If force_resize is True, skip min/default rounding and always scale so that
    the result uses as many pixels as allowed under max_pixels (same rule as the
    shrink-to-fit branch: area <= max_pixels, aspect ratio preserved, factor-aligned).

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    if force_resize:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
        return h_bar, w_bar
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


VLLM_MAX_IMAGE_INPUT_NUM = 24

# 与 `tokenizer.encode(..., add_special_tokens=False)` 一致，用于生成循环内后缀匹配
STREAM_OUTPUT_TRIGGER_TAGS: tuple[str, ...] = ("</Standby>", "</Silence>")


def output_ids_endswith_suffix(output_ids: list[int], suffix: list[int]) -> bool:
    m, n = len(suffix), len(output_ids)
    if m == 0 or n < m:
        return False
    return output_ids[-m:] == suffix


def strip_file_url(path: str) -> str:
    if path.startswith("file://"):
        return path[len("file://") :]
    return path


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model."""
    path_parts = model_path.split("/")
    non_moe_patterns = ["2B", "4B", "8B", "32B"]
    for part in path_parts:
        if any(pattern in part for pattern in non_moe_patterns):
            return False
    return True


def _ensure_url(path: str, prefixes: Iterable[str], data_name: str) -> str:
    if any(path.startswith(prefix) for prefix in prefixes):
        return path
    if os.path.exists(path):
        return "file://" + path
    raise ValueError(f"Invalid {data_name}: {path}")


def ensure_image_url(image: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:image"]
    return _ensure_url(image, prefixes=prefixes, data_name="image")


def ensure_video_url(video: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:video"]
    return _ensure_url(video, prefixes=prefixes, data_name="video")


def smart_video_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 1,
    factor: int = 28,
    frame_min_pixels: int = 16 * 28 * 28 * 4,
    frame_max_pixels: int = 1024 * 28 * 28 * 4,
    force_resize: bool = False,
    # video_max_total_pixels: int = 5000 * 28 * 28 * 4,
):
    assert temporal_factor == 1, "temporal_factor must be 1 for videochat3!"
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar, w_bar = smart_resize(
        height, width, factor, frame_min_pixels, frame_max_pixels, force_resize=force_resize
    )
    # t_bar = round(num_frames / temporal_factor) * temporal_factor

    # if t_bar * h_bar * w_bar > video_max_total_pixels:
    #     beta = math.sqrt((num_frames * height * width) / video_max_total_pixels)
    #     h_bar = max(factor, math.floor(height / beta / factor) * factor)
    #     w_bar = max(factor, math.floor(width / beta / factor) * factor)

    return h_bar, w_bar

