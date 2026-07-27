"""Abstract base class for online streaming video models in VLMEvalKit.

These models run a stateful streaming loop inside generate_inner, returning a
JSON-encoded record list as the prediction string.  VLMEvalKit's standard
task scheduling, per-rank pkl resume, and result aggregation all work
unchanged.

Dataset (ProactiveVideoQA) puts all streaming metadata into the message as a
text item with a special prefix; this module extracts it.
"""
from __future__ import annotations

import json
import math
from abc import abstractmethod

from ..base import BaseModel

# Must match the prefix used in ProactiveVideoQA.build_prompt
STREAM_META_PREFIX = '__PROACTIVE_STREAM_META__'
RESPONSE_TAG = '</Response>'


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _normalise_time(value: float):
    value = round(float(value), 6)
    nearest_int = round(value)
    if abs(value - nearest_int) < 1e-9:
        return int(nearest_int)
    return value


def _format_time(value: float) -> str:
    value = _normalise_time(value)
    if isinstance(value, int):
        return str(value)
    return f'{value:.6f}'.rstrip('0').rstrip('.')


def _round_span(round_idx: int, fps: float):
    start = round_idx / fps
    end = (round_idx + 1) / fps
    return _normalise_time(start), _normalise_time(end)


def _format_time_tag(round_idx: int, fps: float) -> str:
    start, end = _round_span(round_idx, fps)
    return f'<{_format_time(start)}s-{_format_time(end)}s>'


def _parse_raw_answer(raw: str):
    """Map a raw tagged answer to (answerable, model_response).

    Returns ('Yes', content) for </Response>..., ('No', None) otherwise.
    """
    s = (raw or '').strip()
    if s.startswith(RESPONSE_TAG):
        content = s[len(RESPONSE_TAG):].lstrip(' \t:\n')
        return 'Yes', content
    return 'No', None


def _extract_stream_meta(message: list) -> dict:
    """Find and decode the stream_meta dict embedded in a message list."""
    for item in message:
        if item.get('type') == 'text' and isinstance(item.get('value'), str):
            if item['value'].startswith(STREAM_META_PREFIX):
                return json.loads(item['value'][len(STREAM_META_PREFIX):])
    raise ValueError(
        'No stream_meta found in message — was this dataset built with ProactiveVideoQA?'
    )


def _effective_fps(model, meta: dict) -> float:
    fps = getattr(model, 'target_fps', None)
    if fps is None:
        fps = meta.get('target_fps', 1.0)
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f'target_fps must be positive, got {fps}')
    return fps


def _compute_stream_rounds(meta: dict, fps: float) -> int:
    """Compute how many streaming steps to run at the requested fps."""
    answer = meta.get('answer') or []
    duration = meta.get('duration')
    if duration is None or not answer:
        return max(1, int(meta['target_rounds']))

    duration = float(duration)
    last_reply_end = float(answer[-1]['reply_timespan'][1])
    by_frames = int(duration * fps)
    by_reply = int(math.ceil(min(duration, last_reply_end) * fps))
    return max(1, min(by_frames, by_reply))


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class StreamingVideoModel(BaseModel):
    """Abstract base for online streaming video models.

    Sub-classes must implement:
        reset_session(question, extra_turns)  — initialise a fresh session
        step(frame, round_idx) -> str         — process one frame, return raw tag
        _open_extractor(video_path)           — return VideoFrameExtractor-compatible obj

    The full streaming loop runs inside generate_inner; VLMEvalKit's scheduling,
    per-rank pkl resume, and result aggregation are not affected.

    Prompt convention
    -----------------
    The ProactiveVideoQA dataset builds a two-element message:
        [{'type': 'video', 'value': video_path},
         {'type': 'text',  'value': '__PROACTIVE_STREAM_META__<json>'}]
    The ``video`` entry is only a progress-tracker placeholder; frames are read
    inside ``generate_inner`` via ``meta['video_path']``. Do not run
    ``BaseModel.preproc_content`` on it (missing files on S3 mount would fail
    the ``mime is None => type must be text`` assert).
    """

    VIDEO_LLM = True
    INTERLEAVE = True

    def __init__(self):
        super().__init__()

    def generate(self, message, dataset=None):
        """Skip file-based preproc; streaming loop opens the video itself."""
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], (
            f'Invalid input type: {message}'
        )
        if self.check_content(message) == 'str':
            message = [dict(type='text', value=message)]
        elif self.check_content(message) == 'dict':
            message = [message]
        elif self.check_content(message) == 'liststr':
            message = [dict(type='text', value=s) for s in message]
        for item in message:
            assert item.get('type') in self.allowed_types, f'Invalid input type: {item.get("type")}'
            assert 'value' in item
        return self.generate_inner(message, dataset)

    def generate_inner(self, message: list, dataset=None) -> str:
        """Run the streaming loop; return JSON string of per-step records."""
        meta = _extract_stream_meta(message)
        fps = _effective_fps(self, meta)
        target_rounds = _compute_stream_rounds(meta, fps)

        self.reset_session(
            question=meta['question'],
            extra_turns=meta['extra_turns'],
        )

        records = []
        extractor = self._open_extractor(meta['video_path'])
        try:
            for r in range(target_rounds):
                try:
                    frame = extractor.get_frame_at_round(r)
                except StopIteration:
                    break
                raw = self.step(frame, round_idx=r)
                answerable, content = _parse_raw_answer(raw)
                records.append({
                    'video_span': list(_round_span(r, fps)),
                    'answerable': answerable,
                    'model_response': content,
                    'raw_answer': raw,
                })
        finally:
            extractor.close()

        return json.dumps({
            'question_id': meta['qid'],
            'question': meta['question'],
            'records': records,
        }, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def reset_session(self, question: str, extra_turns: list):
        """Initialise / reset the streaming session for a new video.

        Args:
            question: The main user question string.
            extra_turns: List of {'time': float, 'content': str} dicts for
                additional user turns (e.g. TV subtitles).
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, frame, round_idx: int) -> str:
        """Process one frame and return a raw tagged answer string.

        Returns one of:
            '</Silence>'
            '</Standby>'
            '</Response><text content>'
        """
        raise NotImplementedError

    @abstractmethod
    def _open_extractor(self, video_path: str):
        """Return a VideoFrameExtractor-compatible object.

        The returned object must support:
            extractor.get_frame_at_round(r)  -> frame (raises StopIteration on end)
            extractor.close()
        """
        raise NotImplementedError
