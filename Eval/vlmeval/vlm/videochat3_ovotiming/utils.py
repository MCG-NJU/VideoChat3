from __future__ import annotations

import math
import torch
from PIL import Image
import os
from typing import Iterable

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

class _VideoChat3InferenceFastEngine:
    """与 `inference_fast.Qwen3VLStreamEngine` 相同的 infer 接口，复用已加载的 model/processor。"""

    _END_TOKENS = ('<|redacted_im_end|>', '<|endoftext|>')

    def __init__(self, vlm: 'VideoChat3') -> None:
        self.processor = vlm.processor
        self.model = vlm.model
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _to_chat_messages(messages: list[dict], images: list[Image.Image]) -> list[dict]:
        image_iter = iter(images)
        converted = []
        for m in messages:
            role, content = m['role'], m['content']
            if '<image>' not in content:
                converted.append({'role': role, 'content': [{'type': 'text', 'text': content}]})
                continue
            segments = content.split('<image>')
            parts: list[dict] = []
            for idx, seg in enumerate(segments):
                seg = seg.strip()
                if seg:
                    parts.append({'type': 'text', 'text': seg})
                if idx < len(segments) - 1:
                    parts.append({'type': 'image', 'image': next(image_iter)})
            converted.append({'role': role, 'content': parts})
        return converted

    def _strip_end_tokens(self, text: str) -> str:
        for end_tok in self._END_TOKENS:
            while text.rstrip().endswith(end_tok):
                text = text.rstrip()[: -len(end_tok)]
        return text.strip()

    def infer(self, data: dict, max_tokens: int = 128, temperature: float = 0.0) -> str:
        chat = self._to_chat_messages(data['messages'], data['images'])

        # print(f"chat: {chat}", flush=True)

        # print("chat messages:\n ", chat, flush=True)

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs['temperature'] = temperature

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs['input_ids'].shape[-1]
        # print(f'prompt_len: {prompt_len}', flush=True)
        new_tokens = out[:, prompt_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return self._strip_end_tokens(text)

    @staticmethod
    def _to_chat_messages_video(messages: list[dict], videos: list) -> list[dict]:
        video_iter = iter(videos)
        converted = []
        for m in messages:
            role, content = m['role'], m['content']
            if '<video>' not in content:
                converted.append({'role': role, 'content': [{'type': 'text', 'text': content}]})
                continue
            segments = content.split('<video>')
            parts: list[dict] = []
            for idx, seg in enumerate(segments):
                seg = seg.strip()
                if seg:
                    parts.append({'type': 'text', 'text': seg})
                if idx < len(segments) - 1:
                    parts.append({'type': 'video', 'video': next(video_iter)})
            converted.append({'role': role, 'content': parts})
        return converted

    @staticmethod
    def pil_frames_to_video_tensor(frames: list[Image.Image]) -> torch.Tensor:
        """PIL 帧列表 -> (T, C, H, W) uint8 tensor，供 video_processor 使用。"""
        from torchvision.transforms.functional import pil_to_tensor

        return torch.stack([pil_to_tensor(f.convert('RGB')) for f in frames], dim=0)

    @staticmethod
    def build_clip_metadata(
        frames: list[Image.Image],
        t_lo: float,
        t_hi: float,
        clip_fps: float | None = None,
    ) -> dict:
        """构造 VideoChat3VideoMetadata 字段；时间戳由 processor 根据 fps 与 video_start_time 生成。

        ``video_start_time`` 为原视频时间轴上的 clip 起点（秒）；``duration`` 不设为 clip 长度，
        否则在 t_lo 较大时会触发 ``video_start_time < duration`` 校验失败。
        """
        n = len(frames)
        w, h = frames[0].size
        clip_span = max(float(t_hi) - float(t_lo), 1e-6)
        fps = float(clip_fps) if clip_fps is not None else (n / clip_span)
        return {
            'total_num_frames': n,
            'fps': fps,
            'frames_indices': list(range(n)),
            'video_start_time': float(t_lo),
            'width': int(w),
            'height': int(h),
            'duration': None,
        }

    def infer_video(self, data: dict, max_tokens: int = 128, temperature: float = 0.0) -> str:
        """流式在线 v3：多段短视频 + metadata，关闭 processor 内 resize/sample。"""
        chat = self._to_chat_messages_video(data['messages'], data['videos'])

        text = self.processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # print("[INFO] text: ", text, flush=True)
        # print("[INFO] data['videos']: ", data['videos'], flush=True)
        # print("[INFO] data['video_metadata']: ", data['video_metadata'], flush=True)

        inputs = self.processor(
            text=[text],
            videos=data['videos'],
            video_metadata=data['video_metadata'],
            do_resize=False,
            do_sample_frames=False,
            return_tensors='pt',
        )
        try:
            inputs = inputs.to(self.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            # do_sample=False,
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs['temperature'] = temperature

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs['input_ids'].shape[-1]
        new_tokens = out[:, prompt_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return self._strip_end_tokens(text)




def extract_ovo_online_meta(message: list) -> dict | None:
    for item in message:
        if isinstance(item, dict) and item.get('ovo_online_meta'):
            return item['ovo_online_meta']
    return None


def max_num_frames_from_ovo_meta(meta: dict) -> int:
    return max(1, int(meta.get('max_num_frames', meta.get('max_rounds', 32))))
