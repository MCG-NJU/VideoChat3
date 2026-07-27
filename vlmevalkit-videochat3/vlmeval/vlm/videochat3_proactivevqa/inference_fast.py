"""VideoChat3 streaming frame-by-frame inference engine.

Usage:
    engine = VideoChat3StreamEngine(model_path)
    session = StreamingSession(
        engine, system=SYSTEM, question=QUESTION, question_time=0, max_rounds=32
    )
    extractor = VideoFrameExtractor(video_path, target_fps=1.0)
    for t in range(extractor.get_total_rounds()):
        answer = session.step(extractor.get_frame_at_round(t), round_idx=t)

Debug mode:
    Pass debug=True to VideoChat3StreamEngine and/or StreamingSession to get
    verbose per-round prompt / output logging (stderr or file via debug_log_path).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from queue import Queue
from threading import Thread
from typing import List, Optional, TextIO

import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM = """
You are a helpful assistant specializing in streaming video analysis.
You will receive input frame by frame, each labeled with absolute time intervals
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
   - No relevant event has started, OR
   - The current input is irrelevant to the given question.

2. Use </Standby> when:
   - An event is in progress but has not yet completed, OR
   - The current input is relevant but the question cannot yet be answered.

3. Use </Response> only when:
   - An event has fully concluded, OR
   - The available information is sufficient to fully answer the question.
   Provide a complete description at this point.

Do not provide partial answers or speculate beyond the given information.
Whenever you deliver an answer, begin with </Response>.
""".strip()

_TAG_RE = re.compile(r"^\s*(</Silence>|</Standby>|</Response>)")


# ---------------------------------------------------------------------------
# Video frame extractor
# ---------------------------------------------------------------------------
class VideoFrameExtractor:
    """Extract video frames at target fps in decode order."""

    def __init__(self, video_path: str, target_fps: float = 1.0, prefetch: bool = True):
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.original_fps if self.original_fps > 0 else 0
        self.frame_interval = max(int(round(self.original_fps / target_fps)), 1)
        self.num_extracted_frames = int(self.duration * target_fps)

        self._prefetch = prefetch
        if prefetch:
            self._queue: "Queue[Optional[Image.Image]]" = Queue(maxsize=4)
            Thread(target=self._producer, daemon=True).start()
        else:
            self._iter = self._sequential_iter()

    def _sequential_iter(self):
        idx, extracted = 0, 0
        while extracted < self.num_extracted_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % self.frame_interval == 0:
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                extracted += 1
            idx += 1

    def _producer(self):
        for img in self._sequential_iter():
            self._queue.put(img)
        self._queue.put(None)

    def get_frame_at_round(self, round_num: int) -> Image.Image:
        img = self._queue.get() if self._prefetch else next(self._iter, None)
        if img is None:
            raise StopIteration(f"No more frames at round {round_num}")
        return img

    def get_total_rounds(self) -> int:
        return self.num_extracted_frames

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# VideoChat3 inference engine
# ---------------------------------------------------------------------------
class VideoChat3StreamEngine:
    """HF transformers based streaming engine for VideoChat3."""

    _END_TOKENS = ("<|im_end|>", "<|endoftext|>")

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        attn_implementation: str = "flash_attention_2",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        debug: bool = False,
        debug_log_path: Optional[str] = None,
        debug_max_text_chars: int = 4000,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        # VideoChat3 uses Qwen2VLImageProcessorFast which supports min/max_pixels.
        # Without max_pixels the processor has no upper bound (longest_edge=16M),
        # causing thousands of visual tokens per frame and severe slowdowns.
        min_pixels = min_pixels or int(os.environ.get("MIN_PIXELS", 3136))
        max_pixels = max_pixels or int(os.environ.get("MAX_PIXELS", 100352))
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.torch = torch

        self.debug = debug
        self.debug_max_text_chars = debug_max_text_chars
        self._debug_step: int = 0
        self._debug_fp: Optional[TextIO] = None
        if debug and debug_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(debug_log_path)), exist_ok=True)
            self._debug_fp = open(debug_log_path, "a", buffering=1)

    def _dbg(self, *args) -> None:
        if not self.debug:
            return
        msg = " ".join(str(a) for a in args)
        try:
            if self._debug_fp is not None:
                self._debug_fp.write(msg + "\n")
            else:
                print(msg, file=sys.stderr, flush=True)
        except Exception:
            pass

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + f"\n... <truncated {len(text) - max_chars} chars> ...\n" + text[-half:]

    @staticmethod
    def _strip_end_tokens(text: str) -> str:
        for end_tok in VideoChat3StreamEngine._END_TOKENS:
            while text.rstrip().endswith(end_tok):
                text = text.rstrip()[: -len(end_tok)]
        return text.strip()

    def infer(self, messages: List[dict], max_tokens: int = 128, temperature: float = 0.0,
              top_p: float = 1.0, top_k: int = 0) -> str:
        if self.debug:
            self._debug_step += 1
            self._dbg(
                f"\n{'=' * 80}\n[DEBUG step={self._debug_step}] messages ({len(messages)} turns)\n{'=' * 80}"
            )
            for i, m in enumerate(messages):
                role = m["role"]
                content = m["content"]
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if c.get("type") == "image":
                            parts.append("<<IMAGE>>")
                        elif c.get("type") == "text":
                            parts.append(repr(c["text"]))
                        else:
                            parts.append(repr(c))
                    self._dbg(f"  [{i:02d}] {role}: {' | '.join(parts)}")
                else:
                    self._dbg(f"  [{i:02d}] {role}: {repr(content)}")

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        if self.debug:
            try:
                input_ids = inputs["input_ids"]
                self._dbg(f"[DEBUG step={self._debug_step}] input_ids.shape = {tuple(input_ids.shape)}")
            except Exception as e:
                self._dbg(f"[DEBUG] tensor info failed: {e}")

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            use_cache=True,
        )
        if temperature > 0.0:
            # Sampling mode: use caller-supplied values.
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        else:
            # Greedy mode: neutralize model's generation_config presets to
            # silence the "generation flags not valid" warning.
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 0

        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        cleaned = self._strip_end_tokens(text)
        if self.debug:
            self._dbg(f"[DEBUG step={self._debug_step}] raw={repr(text)}  cleaned={repr(cleaned)}")
        return cleaned


# ---------------------------------------------------------------------------
# Streaming session with sliding window
# ---------------------------------------------------------------------------
class StreamingSession:
    """Stateful per-video session with sliding window."""

    def __init__(
        self,
        engine: VideoChat3StreamEngine,
        system: Optional[str] = SYSTEM,
        question: Optional[str] = None,
        question_time: int = 0,
        max_rounds: int = 32,
        global_question: bool = True,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        debug: bool = False,
    ):
        self.engine = engine
        self.system = system
        self.question = question
        self.question_time = question_time
        self.max_rounds = max_rounds
        self.global_question = global_question
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.debug = debug

        self._messages: List[dict] = []
        self._last_answer: Optional[str] = None
        # Tracks the round_idx of the first user message currently in the window.
        self._window_start_round: int = 0

    def _user_content(
        self,
        frame: Image.Image,
        round_idx: int,
        include_question: bool,
        frame_max_pixels: Optional[int] = None,
    ) -> List[dict]:
        """Build user message content for a single frame."""
        time_tag = f"<{round_idx}s-{round_idx + 1}s>"
        text = time_tag
        if include_question and self.question:
            text = f"{self.question}\n{text}"
        image_item: dict = {"type": "image", "image": frame}
        if frame_max_pixels is not None:
            image_item["min_pixels"] = 28 * 28
            image_item["max_pixels"] = frame_max_pixels
        return [
            image_item,
            {"type": "text", "text": text},
        ]

    def _inject_question(self, msg: dict) -> None:
        """Prepend the question into the text part of a user message (in-place)."""
        if not self.question:
            return
        content = msg.get("content", [])
        for i, item in enumerate(content):
            if item.get("type") == "text":
                if self.question not in item.get("text", ""):
                    new_item = dict(item)
                    new_item["text"] = self.question + "\n" + new_item["text"]
                    msg["content"] = list(content)
                    msg["content"][i] = new_item
                break

    @staticmethod
    def _text_only(text: str) -> List[dict]:
        """Wrap plain text as a single-item content list (required by VideoChat3 processor)."""
        return [{"type": "text", "text": text}]

    def _append_turn(
        self,
        frame: Image.Image,
        round_idx: int,
        frame_max_pixels: Optional[int] = None,
    ) -> None:
        if not self._messages:
            # First frame: include question if global_question or round matches question_time.
            # VideoChat3's processor requires content to always be List[dict], never bare str.
            if self.system:
                self._messages.append(
                    {"role": "system", "content": self._text_only(self.system)}
                )
            include_q = self.global_question or (round_idx == self.question_time)
            self._messages.append(
                {
                    "role": "user",
                    "content": self._user_content(
                        frame, round_idx, include_q, frame_max_pixels
                    ),
                }
            )
            self._window_start_round = round_idx
            return

        # Subsequent frames: append last assistant answer then new user turn.
        if self._last_answer is not None:
            self._messages.append(
                {"role": "assistant", "content": self._text_only(self._last_answer)}
            )

        # FIX 1: do NOT re-inject question on every turn when global_question=True.
        # The question was already placed in the first message; only re-include it
        # if this specific round_idx is question_time.
        include_q = (round_idx == self.question_time)
        self._messages.append(
            {
                "role": "user",
                "content": self._user_content(
                    frame, round_idx, include_q, frame_max_pixels
                ),
            }
        )

        # Sliding window: keep at most max_rounds user turns.
        user_count = sum(1 for m in self._messages if m.get("role") == "user")
        if user_count > self.max_rounds:
            rounds_to_remove = user_count - self.max_rounds
            sys_offset = 1 if self._messages and self._messages[0].get("role") == "system" else 0

            for _ in range(rounds_to_remove):
                # Remove oldest user message.
                if (
                    sys_offset < len(self._messages)
                    and self._messages[sys_offset].get("role") == "user"
                ):
                    del self._messages[sys_offset]
                # Remove the paired assistant message (if present).
                if (
                    sys_offset < len(self._messages)
                    and self._messages[sys_offset].get("role") == "assistant"
                ):
                    del self._messages[sys_offset]

            self._window_start_round += rounds_to_remove

            # FIX 2: Re-inject question into the new first user message if it was
            # lost due to truncation.
            # Mirrors the original Qwen3-VL logic:
            #   include_q_start = global_question OR (question_time == new window start)
            new_first_idx = next(
                (i for i, m in enumerate(self._messages) if m.get("role") == "user"), None
            )
            if new_first_idx is not None:
                include_q_start = self.global_question or (
                    self.question_time == self._window_start_round
                )
                if include_q_start:
                    self._inject_question(self._messages[new_first_idx])

    def step(
        self,
        frame: Image.Image,
        round_idx: int,
        frame_max_pixels: Optional[int] = None,
    ) -> str:
        """Feed one frame and return the model answer for this round."""
        self._append_turn(frame, round_idx, frame_max_pixels=frame_max_pixels)

        if self.debug:
            msgs = self._messages
            user_turns = sum(1 for m in msgs if m.get("role") == "user")
            self.engine._dbg(
                f"\n###### StreamingSession.step round_idx={round_idx} "
                f"question_time={self.question_time} global_question={self.global_question} "
                f"| turns={len(msgs)} user_turns={user_turns} "
                f"window_start={self._window_start_round} ######"
            )

        if round_idx < self.question_time:
            answer = "</Silence>"
            if self.debug:
                self.engine._dbg(
                    f"  [round {round_idx}] short-circuit '</Silence>' "
                    f"(round_idx < question_time={self.question_time})"
                )
        else:
            answer = self.engine.infer(
                self._messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        self._last_answer = answer
        return answer

    def reset(self) -> None:
        self._messages = []
        self._last_answer = None
        self._window_start_round = 0


# ---------------------------------------------------------------------------
# Convenience: run a full video
# ---------------------------------------------------------------------------
def run_video(
    engine: VideoChat3StreamEngine,
    video_path: str,
    question: str,
    target_fps: float = 1.0,
    system: str = SYSTEM,
    question_time: int = 0,
    max_rounds: int = 32,
    global_question: bool = True,
    max_tokens: int = 128,
    temperature: float = 0.0,
    verbose: bool = True,
) -> dict:
    extractor = VideoFrameExtractor(video_path, target_fps=target_fps)
    session = StreamingSession(
        engine=engine,
        system=system,
        question=question,
        question_time=question_time,
        max_rounds=max_rounds,
        global_question=global_question,
        max_tokens=max_tokens,
        temperature=temperature,
        debug=getattr(engine, "debug", False),
    )

    outputs = {}
    try:
        for i in range(extractor.get_total_rounds()):
            frame = extractor.get_frame_at_round(i)
            answer = session.step(frame, round_idx=i)
            outputs[f"Round {i}"] = answer
            if verbose:
                print(f"[{i}s-{i + 1}s] {answer}")
    finally:
        extractor.close()

    return outputs


# ---------------------------------------------------------------------------
# Debug helper: run one video with full verbose logging
# ---------------------------------------------------------------------------
def debug_single_video(
    model_path: str,
    video_path: str,
    question: str,
    *,
    target_fps: float = 1.0,
    system: str = SYSTEM,
    question_time: int = 0,
    max_rounds: int = 32,
    global_question: bool = True,
    max_tokens: int = 128,
    temperature: float = 0.0,
    log_path: Optional[str] = None,
    max_text_chars: int = 4000,
) -> dict:
    """Run one video+question with full debug output (stderr or file)."""
    engine = VideoChat3StreamEngine(
        model_path,
        debug=True,
        debug_log_path=log_path,
        debug_max_text_chars=max_text_chars,
    )
    engine._dbg(f"\n############ DEBUG SESSION ############")
    engine._dbg(f"model_path      = {model_path}")
    engine._dbg(f"video_path      = {video_path}")
    engine._dbg(f"question        = {repr(question)}")
    engine._dbg(f"target_fps      = {target_fps}")
    engine._dbg(f"question_time   = {question_time}")
    engine._dbg(f"max_rounds      = {max_rounds}")
    engine._dbg(f"global_question = {global_question}")
    engine._dbg(f"system (head)   = {repr(system[:200])}")
    engine._dbg(f"#######################################\n")

    return run_video(
        engine,
        video_path=video_path,
        question=question,
        target_fps=target_fps,
        system=system,
        question_time=question_time,
        max_rounds=max_rounds,
        global_question=global_question,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# CLI / demo entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    MODEL_PATH = (
        "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/"
        "stage4_online/VideoChat3_4B_train_stage4_online_v8/20260422105827/hf-latest"
    )
    VIDEO_PATH = (
        "/mnt/petrelfs/zengxiangyu/Research_yyd/A_Datasets/online_eval/"
        "ProactiveVideoQA/WEB/videos/8nr6Np3HGWw.6.mp4"
    )
    QUESTION = "What activity is taking place with the sacks?"

    # -------------------------------------------------------------------
    # Two modes:
    #   DEBUG=0  (default): normal inference, results saved to JSONL
    #   DEBUG=1            : verbose per-round prompt/output logging
    # Set DEBUG_LOG=<path> to redirect debug output to a file instead of stderr.
    # -------------------------------------------------------------------
    DEBUG = bool(int(os.environ.get("DEBUG", "0")))
    DEBUG_LOG = os.environ.get("DEBUG_LOG") or None

    if DEBUG:
        debug_single_video(
            model_path=MODEL_PATH,
            video_path=VIDEO_PATH,
            question=QUESTION,
            target_fps=1.0,
            question_time=0,
            max_rounds=32,
            global_question=True,
            log_path=DEBUG_LOG,
        )
    else:
        engine = VideoChat3StreamEngine(MODEL_PATH)

        t0 = time.time()
        outputs = run_video(
            engine,
            video_path=VIDEO_PATH,
            question=QUESTION,
            target_fps=1.0,
            max_rounds=32,
            global_question=True,
        )
        print(f"Total time: {time.time() - t0:.2f}s")

        with open("./test_sample_video_vc3.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "video_path": VIDEO_PATH,
                        "target_fps": 1.0,
                        "output": outputs,
                    }
                )
                + "\n"
            )
