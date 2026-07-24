"""Qwen3-VL 流式逐帧推理 (对外接口版).

用法:
    engine = Qwen3VLStreamEngine(model_path)
    session = StreamingSession(engine, system=SYSTEM, question=QUESTION,
                               question_time=0, max_num_frames=120, global_question=True)

    for frame, t in iter_video_frames(video_path, target_fps=1.0):
        answer = session.step(frame, round_idx=t)
        print(f"[{t}s] {answer}")
"""

from __future__ import annotations

import os
import json
import time
from queue import Queue
from threading import Thread
from typing import List, Optional, Sequence, Tuple

import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# 默认 system prompt
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
"""


# ---------------------------------------------------------------------------
# 视频帧抽取
# ---------------------------------------------------------------------------
class VideoFrameExtractor:
    """按指定 fps 从视频中抽帧, 顺序解码避免 seek 开销."""

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
# Qwen3-VL 推理引擎
# ---------------------------------------------------------------------------
class Qwen3VLStreamEngine:
    """基于 HuggingFace transformers 的 Qwen3-VL 推理引擎."""

    _END_TOKENS = ("<|im_end|>", "<|endoftext|>")

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        device: str = "cuda",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        min_pixels = min_pixels or int(os.environ.get("MIN_PIXELS", 3136))
        max_pixels = max_pixels or int(os.environ.get("MAX_PIXELS", 100352))

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=trust_remote_code,
        )
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _to_chat_messages(
        messages: List[dict], images: List[Image.Image]
    ) -> List[dict]:
        """把内部 messages (含 <image> 占位符) 转成 Qwen-VL 多模态格式.

        - 纯文本 role (system/assistant) 包成 [{"type":"text",...}].
        - 含 <image> 的 user message：按原文顺序交替「<image> 前的一段 text」与「image」,
          与 ``...<tag>\\n<image>...`` 中时间戳与帧一一对应.
        """
        image_iter = iter(images)
        converted = []
        for m in messages:
            role, content = m["role"], m["content"]
            if "<image>" not in content:
                converted.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue
            segments = content.split("<image>")
            parts: List[dict] = []
            for idx, seg in enumerate(segments):
                seg = seg.strip()
                if seg:
                    parts.append({"type": "text", "text": seg})
                if idx < len(segments) - 1:
                    parts.append({"type": "image", "image": next(image_iter)})
            converted.append({"role": role, "content": parts})
        return converted

    def _strip_end_tokens(self, text: str) -> str:
        for end_tok in self._END_TOKENS:
            while text.rstrip().endswith(end_tok):
                text = text.rstrip()[: -len(end_tok)]
        return text.strip()

    def infer(self, data: dict, max_tokens: int = 128, temperature: float = 0.0) -> str:
        import torch

        chat = self._to_chat_messages(data["messages"], data["images"])

        # print("chat messages:\n ", chat, flush=True)

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id
                or self.processor.tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        print(f"prompt_len: {prompt_len}",flush=True)
        new_tokens = out[:, prompt_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=False)[0]
        return self._strip_end_tokens(text)


# ---------------------------------------------------------------------------
# 流式会话 (对外主要接口)
# ---------------------------------------------------------------------------
class StreamingSession:
    """对单条视频做逐帧推理的有状态会话对象, 内置滑动窗口.

    ``max_num_frames``: 上下文中保留的 **图像张数** 上限（多帧一批时按张数计，不是对话轮数）。
    """

    def __init__(
        self,
        engine: Qwen3VLStreamEngine,
        system: Optional[str] = SYSTEM,
        question: Optional[str] = None,
        question_messages: Optional[List[dict]] = None,
        question_time: int = 0,
        max_num_frames: int = 120,
        global_question: bool = True,
        max_tokens: int = 128,
        temperature: float = 0.0,
        print_freq: int = 50,
    ):
        self.engine = engine
        self.system = system
        self.question = question
        self.question_messages = question_messages
        self.question_time = question_time
        self.max_num_frames = max_num_frames
        self.global_question = global_question
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.print_freq = print_freq
        self.print_count = 0

        self._data: Optional[dict] = None
        self._last_answer: Optional[str] = None
        # round_idx -> 该步时间窗 (秒); 未设置则用整数标签 <k>s-(k+1)s>
        self._interval_by_step: dict[int, Optional[Tuple[float, float]]] = {}
        # 与 messages 中每个 user 块对应: 该轮含几张图 (用于滑动窗口按轮裁剪)
        self._turn_image_counts: List[int] = []
        # 与每个 user 块对应: 重建首条 user 内容用
        self._turn_meta: List[dict] = []
        # 避免重复注入同一条/同一组 question（但允许不同 question 多次注入）
        self._last_injected_question_sig: Optional[tuple] = None

    def _question_signature(self) -> Optional[tuple]:
        """Build a stable signature for the current question content."""
        if self.question_messages:
            items = []
            for m in self.question_messages:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", ""))
                content = str(m.get("content", ""))
                items.append((role, content))
            return ("messages", tuple(items)) if items else None
        if self.question:
            return ("text", str(self.question))
        return None

    def _resize_stream_frame(self, frame: Image.Image) -> Image.Image:
        print(frame.size, flush=True)
        out = frame.resize((336, 224))
        print(out.size, flush=True)
        return out

    def _user_content(
        self,
        round_idx: int,
        time_sec: Optional[Tuple[float, float]] = None,
    ) -> str:
        if time_sec is not None:
            lo, hi = time_sec
            time_tag = f"<{lo:.3f}s-{hi:.3f}s>\n<image>"
        else:
            time_tag = f"<{round_idx}s-{round_idx + 1}s>\n<image>"
        return time_tag

    def _user_content_multi(
        self,
        round_idx: int,
        time_secs: Sequence[Tuple[float, float]],
    ) -> str:
        """单条 user 内多段时间标签 + 多 <image>, 与 images 顺序一一对应 (不含 question, question 单独一条 message)."""
        parts: List[str] = []
        for lo, hi in time_secs:
            parts.append(f"<{lo:.3f}s-{hi:.3f}s>\n<image>")
        return "\n".join(parts)

    def _append_turn(
        self,
        frame: Image.Image,
        round_idx: int,
        time_sec: Optional[Tuple[float, float]] = None,
    ) -> None:
        if time_sec is None:
            ts_list = [(float(round_idx), float(round_idx + 1))]
        else:
            ts_list = [time_sec]
        self._append_turn_batch([frame], ts_list, round_idx)

    def _append_turn_batch(
        self,
        frames: Sequence[Image.Image],
        time_secs: Sequence[Tuple[float, float]],
        round_idx: int,
    ) -> None:
        if len(frames) != len(time_secs) or not frames:
            raise ValueError("frames and time_secs must be non-empty and the same length")

        self._interval_by_step[round_idx] = (time_secs[0][0], time_secs[-1][1])
        # resize 的管理放在外部
        # resized = [self._resize_stream_frame(f) for f in frames]
        resized = frames

        if self._data is None:
            messages = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            include_q = self.global_question or (round_idx == self.question_time)
            if include_q:
                sig = self._question_signature()
                if sig is not None and sig != self._last_injected_question_sig:
                    if self.question_messages:
                        messages.extend(self.question_messages)
                    elif self.question:
                        messages.append({"role": "user", "content": self.question})
                    self._last_injected_question_sig = sig
            messages.append(
                {
                    "role": "user",
                    "content": self._user_content_multi(round_idx, time_secs),
                }
            )
            self._data = {"images": list(resized), "messages": messages}
            self._turn_image_counts = [len(resized)]
            self._turn_meta = [{"round_idx": round_idx, "time_secs": [tuple(t) for t in time_secs]}]
        else:
            messages = self._data["messages"]
            if self._last_answer is not None:
                messages.append({"role": "assistant", "content": self._last_answer})
            include_q = (not self.global_question) and (round_idx == self.question_time)
            if include_q:
                sig = self._question_signature()
                if sig is not None and sig != self._last_injected_question_sig:
                    if self.question_messages:
                        messages.extend(self.question_messages)
                    elif self.question:
                        messages.append({"role": "user", "content": self.question})
                    self._last_injected_question_sig = sig
            messages.append(
                {
                    "role": "user",
                    "content": self._user_content_multi(round_idx, time_secs),
                }
            )
            self._data["images"].extend(resized)
            self._turn_image_counts.append(len(resized))
            self._turn_meta.append({"round_idx": round_idx, "time_secs": [tuple(t) for t in time_secs]})

        # if include_q:
            # print("question messages:\n ", self.question_messages, flush=True)
            # print("message:\n ", messages, flush=True)

        self._trim_sliding_window()

    def _first_frame_user_message_index(self) -> int:
        """Index of the earliest user message that contains frames (<image>)."""
        m = self._data["messages"]
        for i, msg in enumerate(m):
            if msg.get("role") == "user" and "<image>" in str(msg.get("content", "")):
                return i
        return len(m)

    def _trim_sliding_window(self) -> None:
        """按「含帧 user + assistant」整轮裁剪, 使 images 总数不超过 max_num_frames.

        不删除 system 与独立 question 消息, 只删除带 ``<image>`` 的 user 及其后 assistant.
        """
        if self._data is None:
            return
        images = self._data["images"]
        messages = self._data["messages"]

        # if len(images) > self.max_num_frames:
        #     self.print_count += 1
        #     if self.print_count % self.print_freq == 0:
        #         print(
        #             f"images length: {len(images)} > max_num_frames: {self.max_num_frames}, trimming...",
        #             flush=True,
        #         )

        while len(images) > self.max_num_frames:
            first_u = self._first_frame_user_message_index()
            if first_u + 1 >= len(messages):
                break
            n0 = self._turn_image_counts[0]
            rid = self._turn_meta[0]["round_idx"]
            self._interval_by_step.pop(rid, None)
            del messages[first_u : first_u + 2]
            self._data["images"] = images[n0:]
            images = self._data["images"]
            self._turn_image_counts.pop(0)
            self._turn_meta.pop(0)
            first_u = self._first_frame_user_message_index()
            if first_u < len(messages) and messages[first_u].get("role") == "user":
                tm0 = self._turn_meta[0]
                r = tm0["round_idx"]
                ts = tm0["time_secs"]
                messages[first_u]["content"] = self._user_content_multi(r, ts)

    def step(
        self,
        frame: Image.Image,
        round_idx: int,
        time_sec: Optional[Tuple[float, float]] = None,
    ) -> Optional[str]:
        """喂一帧, 返回该 round 的回答 (自动更新会话状态).

        time_sec: 若给定 (t_lo, t_hi), 时间标签为 ``<t_lo秒-t_hi秒>`` (三位小数),
        用于非 1Hz 输入 (如 fps=2 时步长约 0.5s). 省略则仍为 ``<round>s-(round+1)s>``.
        """
        self._append_turn(frame, round_idx, time_sec)

        answer = self.engine.infer(
            self._data, max_tokens=self.max_tokens, temperature=self.temperature,
        )
        self._last_answer = answer
        return answer

    def step_frames(
        self,
        frames: Sequence[Image.Image],
        time_secs: Sequence[Tuple[float, float]],
        round_idx: int,
    ) -> Optional[str]:
        """同一推理步内喂多帧: 一条 user 消息内含多个时间窗与 <image>, 仅在本步末尾 infer 一次."""
        self._append_turn_batch(list(frames), time_secs, round_idx)

        if round_idx < self.question_time:
            # 未插入问题时不产生 assistant 回复，避免把 </Silence> 写进历史干扰后续上下文
            answer = None
        else:
            answer = self.engine.infer(
                self._data, max_tokens=self.max_tokens, temperature=self.temperature,
            )
        self._last_answer = answer
        return answer

    @property
    def question_message_index(self) -> Optional[int]:
        """独立 question 在 ``self._data['messages']`` 中的下标; 无会话或未插入时为 None."""
        return None

    def reset(self) -> None:
        self._data = None
        self._last_answer = None
        self._interval_by_step = {}
        self._turn_image_counts = []
        self._turn_meta = []
        self._last_injected_question_sig = None


class StreamingSessionV3:
    """流式在线 v3：每步喂一段聚合短视频（``<video>``），时间戳由 processor 根据 metadata 生成。

    ``max_num_frames``: 上下文中保留的 **视频片段数** 上限（与 v2 参数名兼容）。
    """

    def __init__(
        self,
        engine,
        system: Optional[str] = SYSTEM,
        question: Optional[str] = None,
        question_messages: Optional[List[dict]] = None,
        question_time: int = 0,
        max_num_frames: int = 120,
        global_question: bool = True,
        max_tokens: int = 128,
        temperature: float = 0.0,
        print_freq: int = 50,
    ):
        self.engine = engine
        self.system = system
        self.question = question
        self.question_messages = question_messages
        self.question_time = question_time
        self.max_num_frames = max_num_frames
        self.global_question = global_question
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.print_freq = print_freq
        self.print_count = 0

        self._data: Optional[dict] = None
        self._last_answer: Optional[str] = None
        self._interval_by_step: dict[int, Optional[Tuple[float, float]]] = {}
        self._turn_video_counts: List[int] = []
        self._turn_meta: List[dict] = []
        self._last_injected_question_sig: Optional[tuple] = None

    def _question_signature(self) -> Optional[tuple]:
        if self.question_messages:
            items = []
            for m in self.question_messages:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", ""))
                content = str(m.get("content", ""))
                items.append((role, content))
            return ("messages", tuple(items)) if items else None
        if self.question:
            return ("text", str(self.question))
        return None

    @staticmethod
    def _user_content_video() -> str:
        return "<video>"

    def _append_turn_video(
        self,
        video_tensor,
        video_metadata: dict,
        round_idx: int,
        time_sec: Tuple[float, float],
    ) -> None:
        self._interval_by_step[round_idx] = time_sec

        if self._data is None:
            messages = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            include_q = self.global_question or (round_idx == self.question_time)
            if include_q:
                sig = self._question_signature()
                if sig is not None and sig != self._last_injected_question_sig:
                    if self.question_messages:
                        messages.extend(self.question_messages)
                    elif self.question:
                        messages.append({"role": "user", "content": self.question})
                    self._last_injected_question_sig = sig
            messages.append({"role": "user", "content": self._user_content_video()})
            self._data = {
                "videos": [video_tensor],
                "video_metadata": [video_metadata],
                "messages": messages,
            }
            self._turn_video_counts = [1]
            self._turn_meta = [{"round_idx": round_idx, "time_sec": tuple(time_sec)}]
        else:
            messages = self._data["messages"]
            if self._last_answer is not None:
                messages.append({"role": "assistant", "content": self._last_answer})
            include_q = (not self.global_question) and (round_idx == self.question_time)
            if include_q:
                sig = self._question_signature()
                if sig is not None and sig != self._last_injected_question_sig:
                    if self.question_messages:
                        messages.extend(self.question_messages)
                    elif self.question:
                        messages.append({"role": "user", "content": self.question})
                    self._last_injected_question_sig = sig
            messages.append({"role": "user", "content": self._user_content_video()})
            self._data["videos"].append(video_tensor)
            self._data["video_metadata"].append(video_metadata)
            self._turn_video_counts.append(1)
            self._turn_meta.append({"round_idx": round_idx, "time_sec": tuple(time_sec)})

        self._trim_sliding_window()

    def _first_video_user_message_index(self) -> int:
        m = self._data["messages"]
        for i, msg in enumerate(m):
            if msg.get("role") == "user" and "<video>" in str(msg.get("content", "")):
                return i
        return len(m)

    def _trim_sliding_window(self) -> None:
        if self._data is None:
            return
        videos = self._data["videos"]
        video_metadata = self._data["video_metadata"]
        messages = self._data["messages"]

        while len(videos) > self.max_num_frames:
            first_u = self._first_video_user_message_index()
            if first_u + 1 >= len(messages):
                break
            n0 = self._turn_video_counts[0]
            rid = self._turn_meta[0]["round_idx"]
            self._interval_by_step.pop(rid, None)
            del messages[first_u : first_u + 2]
            self._data["videos"] = videos[n0:]
            self._data["video_metadata"] = video_metadata[n0:]
            videos = self._data["videos"]
            video_metadata = self._data["video_metadata"]
            self._turn_video_counts.pop(0)
            self._turn_meta.pop(0)

    def step_clip(
        self,
        frames: Sequence[Image.Image],
        t_lo: float,
        t_hi: float,
        round_idx: int,
        clip_fps: Optional[float] = None,
    ) -> Optional[str]:
        """将多帧聚合为一段短视频并推理一步；时间窗 [t_lo, t_hi] 写入 metadata。"""
        if not frames:
            raise ValueError("frames must be non-empty")
        video_tensor = self.engine.pil_frames_to_video_tensor(list(frames))
        meta = self.engine.build_clip_metadata(list(frames), t_lo, t_hi, clip_fps=clip_fps)
        self._append_turn_video(video_tensor, meta, round_idx, (float(t_lo), float(t_hi)))

        answer = self.engine.infer_video(
            self._data, max_tokens=self.max_tokens, temperature=self.temperature,
        )
        self._last_answer = answer
        return answer

    def reset(self) -> None:
        self._data = None
        self._last_answer = None
        self._interval_by_step = {}
        self._turn_video_counts = []
        self._turn_meta = []
        self._last_injected_question_sig = None


# ---------------------------------------------------------------------------
# 便捷函数: 跑完整条视频
# ---------------------------------------------------------------------------
def run_video(
    engine: Qwen3VLStreamEngine,
    video_path: str,
    question: str,
    target_fps: float = 1.0,
    system: str = SYSTEM,
    question_time: int = 0,
    max_num_frames: int = 120,
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
        max_num_frames=max_num_frames,
        global_question=global_question,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    outputs = {}
    try:
        for i in range(extractor.get_total_rounds()):
            t0=time.time()
            frame = extractor.get_frame_at_round(i)
            answer = session.step(frame, round_idx=i)
            outputs[f"Round {i}"] = answer
            if verbose:
                print(f"[{i}s-{i + 1}s] {answer}, step time: {time.time() - t0}")
    finally:
        extractor.close()

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MIN_PIXELS", "3136")
    os.environ.setdefault("MAX_PIXELS", "100352")

    MODEL_PATH = "/mnt/petrelfs/share_data/zengxiangyu/models/VideoChat3/VideoChat3_4B_train_stage4_online_v1/20260411115016/hf-1284"
    VIDEO_PATH = "/mnt/petrelfs/zhangzhiqiu/LLaMA-Factory/data/mllm_demo_data/1.mp4"
    QUESTION = "Describe the video in detail."

    engine = Qwen3VLStreamEngine(MODEL_PATH)

    t0 = time.time()
    outputs = run_video(
        engine,
        video_path=VIDEO_PATH,
        question=QUESTION,
        target_fps=1.0,
        max_num_frames=120,
        global_question=True,
    )
    print(f"Total time: {time.time() - t0:.2f}s")

    with open("./test_sample_video.jsonl", "a") as f:
        f.write(json.dumps({
            "video_path": VIDEO_PATH,
            "target_fps": 1.0,
            "output": outputs,
        }) + "\n")
