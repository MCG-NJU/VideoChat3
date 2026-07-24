"""VideoChat3 online streaming model for ProactiveVideoQA."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .streaming_video_base import (
    StreamingVideoModel,
    _format_time_tag,
    _round_span,
)
from .utils import smart_video_resize


def _make_proactive_session_class(StreamingSession):
    class ProactiveStreamingSession(StreamingSession):
        def __init__(self, *args, extra_turns=None, target_fps=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self._extra_turns = list(extra_turns or [])
            self._target_fps = float(target_fps)

        def _user_content(
            self,
            frame,
            round_idx: int,
            include_question: bool,
            frame_max_pixels: int | None = None,
        ):
            base = super()._user_content(
                frame, round_idx, include_question, frame_max_pixels
            )
            old_tag = f"<{round_idx}s-{round_idx + 1}s>"
            new_tag = _format_time_tag(round_idx, self._target_fps)
            start, end = _round_span(round_idx, self._target_fps)
            hits = [
                t["content"]
                for t in self._extra_turns
                if start < float(t["time"]) <= end and t.get("content")
            ]
            if not hits:
                return [
                    {
                        **item,
                        "text": item["text"].replace(old_tag, new_tag, 1),
                    }
                    if item.get("type") == "text"
                    else item
                    for item in base
                ]
            merged = list(base)
            for idx, item in enumerate(merged):
                if item.get("type") == "text":
                    item = dict(item)
                    item["text"] = item["text"].replace(old_tag, new_tag, 1)
                    item["text"] = "\n".join(hits) + "\n" + item["text"]
                    merged[idx] = item
                    break
            return merged

    return ProactiveStreamingSession


class VideoChat3ProactiveVQA(StreamingVideoModel):
    INSTALL_REQ = False

    def __init__(
        self,
        model_path: str = None,
        max_rounds: int = 32,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 20,
        target_fps: float = 1.0,
        global_question: bool = True,
        attn_implementation: str = "flash_attention_2",
        device: str = "auto",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        encode_frame_size: tuple[int, int] | None = None,
        encode_frame_size_standby: tuple[int, int] | None = None,
        standby_high_res_frames: int = 1,
        force_resize: bool = True,
        debug: bool = False,
        debug_log_path: str | None = None,
        debug_max_text_chars: int = 4000,
        **kwargs,  # absorb any extra VLMEvalKit kwargs; NOT forwarded to the engine
    ):
        super().__init__()
        self.model_path = model_path
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.target_fps = target_fps
        self.global_question = bool(global_question)
        self.standby_high_res_frames = int(standby_high_res_frames)
        self.force_resize = bool(force_resize)
        self._standby_high_res_remaining = 0

        _default_max_pixels = int(os.environ.get("MAX_PIXELS", 100352))
        if encode_frame_size is not None:
            self._frame_max_pixels = int(np.prod(encode_frame_size))
        else:
            self._frame_max_pixels = max_pixels or _default_max_pixels

        if encode_frame_size_standby is not None:
            self._frame_max_pixels_standby = int(np.prod(encode_frame_size_standby))
        else:
            # 放大一倍：线性尺寸 ×2 → 像素数 ×4
            self._frame_max_pixels_standby = self._frame_max_pixels * 4

        print(
            f"[VideoChat3ProactiveVQA] frame_max_pixels={self._frame_max_pixels}, "
            f"frame_max_pixels_standby={self._frame_max_pixels_standby}, "
            f"standby_high_res_frames={self.standby_high_res_frames}",
            flush=True,
        )

        from .inference_fast import (
            SYSTEM,
            StreamingSession,
            VideoChat3StreamEngine,
            VideoFrameExtractor,
        )

        self._SYSTEM = SYSTEM
        self._VideoFrameExtractor = VideoFrameExtractor
        self._ProactiveStreamingSession = _make_proactive_session_class(StreamingSession)
        self._session = None

        print(
            f"[VideoChat3ProactiveVQA] Loading engine from: {model_path}",
            flush=True,
        )
        # Only pass engine-known params; extra VLMEvalKit kwargs are intentionally dropped.
        # Processor global cap must cover standby frames.
        engine_max_pixels = max(self._frame_max_pixels, self._frame_max_pixels_standby)
        if max_pixels is not None:
            engine_max_pixels = max(engine_max_pixels, max_pixels)

        self._engine = VideoChat3StreamEngine(
            model_path,
            device=device,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=engine_max_pixels,
            debug=debug,
            debug_log_path=debug_log_path,
            debug_max_text_chars=debug_max_text_chars,
        )

    def _resize_frame(self, frame: Image.Image, high_res: bool) -> Image.Image:
        if not self.force_resize:
            return frame
        w, h = frame.size
        max_px = (
            self._frame_max_pixels_standby if high_res else self._frame_max_pixels
        )
        h_bar, w_bar = smart_video_resize(
            num_frames=1,
            height=h,
            width=w,
            frame_min_pixels=28 * 28,
            frame_max_pixels=max_px,
            force_resize=True,
        )
        return frame.resize((w_bar, h_bar))

    def reset_session(self, question: str, extra_turns: list):
        self._standby_high_res_remaining = 0
        self._session = self._ProactiveStreamingSession(
            engine=self._engine,
            system=self._SYSTEM,
            question=question,
            question_time=0,
            max_rounds=self.max_rounds,
            global_question=self.global_question,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            extra_turns=extra_turns,
            target_fps=self.target_fps,
        )

    def step(self, frame, round_idx: int) -> str:
        assert self._session is not None, "reset_session must be called before step"

        is_after_standby = self._standby_high_res_remaining > 0
        if self._standby_high_res_remaining > 0:
            self._standby_high_res_remaining -= 1

        frame_max_pixels = (
            self._frame_max_pixels_standby
            if is_after_standby
            else self._frame_max_pixels
        )
        frame = self._resize_frame(frame, is_after_standby)

        print(f"[Budget INFO]is_after_standby: {is_after_standby}, frame_max_pixels: {frame_max_pixels}")

        raw = self._session.step(
            frame, round_idx=round_idx, frame_max_pixels=frame_max_pixels
        )

        if "</Standby>" in raw:
            self._standby_high_res_remaining = self.standby_high_res_frames

        return raw

    def _open_extractor(self, video_path: str):
        return self._VideoFrameExtractor(video_path, target_fps=self.target_fps)

