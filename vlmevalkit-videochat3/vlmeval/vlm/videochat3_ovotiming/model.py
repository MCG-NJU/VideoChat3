from __future__ import annotations

import json
import re
import time

import numpy as np
import torch
from PIL import Image

from ..videochat3.model import VideoChat3
from .utils import (
    _VideoChat3InferenceFastEngine,
    extract_ovo_online_meta,
    max_num_frames_from_ovo_meta,
    smart_video_resize,
    strip_file_url,
)


class VideoChat3OVOTiming(VideoChat3):
    """VideoChat3 adapter dedicated to the ovo_timing streaming benchmark."""

    def __init__(
        self,
        *args,
        encode_frame_size: tuple[int, int] = (224, 224),
        encode_frame_size_standby: tuple[int, int] = (224, 224),
        standby_high_res_frames: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encode_frame_size = encode_frame_size
        self.encode_frame_size_standby = encode_frame_size_standby
        self.max_pixels_online_v2 = int(np.prod(encode_frame_size))
        self.max_pixels_online_v2_standby = int(np.prod(encode_frame_size_standby))
        self.standby_high_res_frames = int(standby_high_res_frames)

    @torch.inference_mode()
    def generate_inner_transformers(self, message, dataset=None):
        """逐帧在线推理：复用 `inference_fast.StreamingSession`（每步全序列 `model.generate` + 滑动窗口）。

        与 `generate_inner_transformers_online` 使用相同的 `ovo_online_meta`；时间标签为浮点区间
        ``<curr>s-(curr+dt)s>``（三位小数），``dt=1/streaming_fps``（例如 fps=2 时为 0.5s），与 v1 一致。
        高 fps（如 2fps、4fps）时：每 ``streaming_fps`` 帧合并为**一条** user 消息（多段时间窗 + 多图）再
        ``infer`` 一次，中间时刻不再插入单独的 ``</Silence>`` 轮次。
        """
        from .inference_fast import StreamingSession, SYSTEM as _IF_STREAM_SYSTEM

        meta = extract_ovo_online_meta(message)
        if meta is None:
            raise ValueError('OVOBench online mode requires a message item with `ovo_online_meta`.')

        streaming_fps = float(meta.get('streaming_fps', 2.0))
        start_time = float(meta['start_time'])
        end_time = float(meta['end_time'])
        question = str(meta['question'])
        max_num_frames = max_num_frames_from_ovo_meta(meta)
        max_new_tok = int(meta.get('max_new_tokens', 128))
        question_time = int(meta.get('question_time', 0))
        global_question = bool(meta.get('global_question', True))
        temperature = float(meta.get('temperature', self.temperature))

        frame_paths = meta.get('frame_paths')
        stream_times = meta.get('stream_times')
        use_prebuilt = (
            isinstance(frame_paths, list)
            and isinstance(stream_times, list)
            and len(frame_paths) == len(stream_times)
            and len(frame_paths) > 0
        )

        video_path = strip_file_url(str(meta['video_path'])) if meta.get('video_path') else ''

        system_text = None
        for msg in message:
            if isinstance(msg, dict) and msg.get('role') == 'system' and msg.get('type') == 'text':
                system_text = str(msg.get('value', ''))
                break

        engine = _VideoChat3InferenceFastEngine(self)
        session = StreamingSession(
            engine=engine,
            system=system_text or _IF_STREAM_SYSTEM,
            question=question,
            question_time=question_time,
            max_num_frames=max_num_frames,
            global_question=global_question,
            max_tokens=max_new_tok,
            temperature=temperature,
        )

        cap = None
        original_fps = 30.0
        total_frames = 0

        if not use_prebuilt:
            import cv2

            if not video_path:
                raise ValueError('OVOBench online mode requires either `frame_paths`+`stream_times` or `video_path`.')
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f'Cannot open video: {video_path}')
            original_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        def read_frame_at(t_sec: float) -> Image.Image | None:
            import cv2

            assert cap is not None
            frame_idx = int(t_sec * original_fps)
            frame_idx = min(max(0, frame_idx), max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                return None
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        checked_detections: list[dict] = []
        response_events: list[dict] = []
        step_outputs: list[dict] = []

        step = 0
        dt = 1.0 / streaming_fps if streaming_fps > 0 else 0.5
        frames_per_infer = max(1, int(round(streaming_fps)))

        # </Standby> 后接下来 1s 用高分辨率：1fps→1 帧，4fps→4 帧
        # standby_high_res_frames = frames_per_infer
        standby_high_res_frames = self.standby_high_res_frames

        try:
            if use_prebuilt:
                step_iter = list(zip(stream_times, frame_paths))
            else:
                step_iter = []
                _t = float(start_time)
                while _t <= end_time + 1e-6:
                    step_iter.append((_t, None))
                    _t += dt

            n_stream = len(step_iter)
            infer_round = 0
            batch_buf: list[tuple] = []

            frame_size = self.encode_frame_size
            frame_size_after_standby = self.encode_frame_size_standby

            standby_high_res_remaining = 0

            for stream_idx, pair in enumerate(step_iter):
                if use_prebuilt:
                    curr_time, fp = float(pair[0]), pair[1]
                    frame = Image.open(strip_file_url(str(fp))).convert('RGB')
                else:
                    curr_time = float(pair[0])
                    frame = read_frame_at(curr_time)
                    if frame is None:
                        break

                is_after_standby = standby_high_res_remaining > 0
                if standby_high_res_remaining > 0:
                    standby_high_res_remaining -= 1

                w, h = frame.size
                # print(f"frame size before resize : {frame.size}, total pixels: {w * h}",flush=True)

                h_bar, w_bar = smart_video_resize(
                    num_frames=1,
                    height=h,
                    width=w,
                    frame_min_pixels=28*28,
                    frame_max_pixels=self.max_pixels_online_v2_standby if is_after_standby else self.max_pixels_online_v2,
                    force_resize=True
                )
                frame = frame.resize((w_bar, h_bar))


                # frame = self._maybe_resize_frame(frame, frame_size if not is_after_standby else frame_size_after_standby)
                # frame = self._maybe_resize_frame(frame, frame_size)

                print(f"frame size after resize : {frame.size}, total pixels: {w_bar * h_bar}",flush=True)

                t_lo, t_hi = float(curr_time), float(curr_time) + dt
                batch_buf.append((frame, (t_lo, t_hi)))

                flush = (len(batch_buf) >= frames_per_infer) or (stream_idx == n_stream - 1)
                if flush:
                    frames = [b[0] for b in batch_buf]
                    time_secs = [b[1] for b in batch_buf]
                    t_anchor = time_secs[-1][0]
                    t_win_end = time_secs[-1][1]

                    t0 = time.perf_counter()
                    answer = session.step_frames(frames, time_secs, infer_round)
                    elapsed = time.perf_counter() - t0

                    step_outputs.append({'curr_time': t_anchor, 'window_end': t_win_end, 'raw': answer, 'wall_time_sec': elapsed})

                    print(
                        f'curr_time {t_anchor}-{t_win_end} ({len(frames)} frames) answer {answer}, step time: {elapsed}',
                        flush=True,
                    )

                    if '</Response>' in answer:
                        body = answer.split('</Response>', 1)[-1].strip()
                        body = re.sub(r'^[^\w\u4e00-\u9fff]*', '', body)
                        if body:
                            checked_detections.append({'time': float(t_win_end), 'response': body})
                            response_events.append({'time': float(t_win_end), 'answer': body, 'raw': answer})
                    if '</Standby>' in answer:
                        standby_high_res_remaining = standby_high_res_frames

                    batch_buf.clear()
                    infer_round += 1

                step += 1
        finally:
            if cap is not None:
                cap.release()
            session.reset()

        det_times = [float(x['time']) for x in checked_detections]
        out_obj = {
            'sample_id': meta.get('sample_id'),
            'task': meta.get('task'),
            'gt_timestamp': meta.get('gt_timestamp'),
            'checked_detections': checked_detections,
            'detections': det_times,
            'response_events': response_events,
            'streaming_fps': streaming_fps,
            'prebuilt_frames': use_prebuilt,
            'n_stream_steps': step,
            'video_path': video_path if not use_prebuilt else '',
            'online_backend': 'inference_fast_streaming_session',
        }
        if self.verbose:
            out_obj['per_step'] = step_outputs
        return json.dumps(out_obj, ensure_ascii=False)


    def generate_inner_vllm(self, message, dataset=None):
        raise NotImplementedError("VideoChat3 does not support vLLM")

    def generate_inner(self, message, dataset=None):
        if extract_ovo_online_meta(message) is not None:
            return self.generate_inner_transformers(message, dataset=dataset)
        else:
            raise ValueError("VideoChat3OVOTiming online mode requires a message item with `ovo_online_meta`.")
