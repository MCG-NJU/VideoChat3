"""VideoChat3 proactive streaming demo.

Frame-by-frame inference at TARGET_FPS; after </Standby>, the next frame is
upscaled 2x in width and height.

Usage:
    # Edit MODEL_PATH / VIDEO_PATH / QUESTION below, then run:
    python demo_vc3_proactive.py

    # Or pass via CLI:
    python demo_vc3_proactive.py --model /path/to/model --video a.mp4 --question "What happened?"
"""
import argparse
import os
import sys
import math

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "16"

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from inference_fast_vc3 import (
    SYSTEM,
    StreamingSession,
    VideoChat3StreamEngine,
    VideoFrameExtractor,
)

# ---------- edit these constants to run directly ----------
MODEL_PATH = "MCG-NJU/VideoChat3-4B"
VIDEO_PATH = "example.mp4"
QUESTION = "Your question"
# ----------------------------------------------------------
TARGET_FPS = 2.0
MAX_PIXELS = 224*224          # pixel budget for normal frames
MAX_ROUNDS = 16
MAX_NEW_TOKENS = 128
# ----------------------------------------------------------

def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    force_resize: bool = False,
):
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


def _resize_frame(frame: Image.Image, max_pixels: int) -> Image.Image:
    w, h = frame.size
    h_bar, w_bar = smart_resize(
        h, w, min_pixels=28 * 28, max_pixels=max_pixels, force_resize=True
    )
    return frame.resize((w_bar, h_bar))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--question", default=QUESTION)
    args = parser.parse_args()

    standby_max_pixels = MAX_PIXELS * 4  # 2x width/height -> 4x pixels

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        min_pixels=28 * 28,
        max_pixels=standby_max_pixels,
    )

    class _Engine:
        def infer(self, messages, max_tokens=MAX_NEW_TOKENS, **_):
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                )
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            text = processor.batch_decode(
                trimmed, skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            return VideoChat3StreamEngine._strip_end_tokens(text)

    session = StreamingSession(
        _Engine(), system=SYSTEM, question=args.question,
        question_time=0, max_rounds=MAX_ROUNDS, max_tokens=MAX_NEW_TOKENS,
    )

    extractor = VideoFrameExtractor(args.video, target_fps=TARGET_FPS)
    actual_fps = extractor.actual_fps if extractor.actual_fps > 0 else TARGET_FPS
    frames_per_round = max(1, round(actual_fps))
    standby_remaining = 0

    try:
        total_frames = extractor.get_total_rounds()
        for frame_start in range(0, total_frames, frames_per_round):
            raw_frames = [
                extractor.get_frame_at_round(i)
                for i in range(
                    frame_start,
                    min(frame_start + frames_per_round, total_frames),
                )
            ]
            high_res = standby_remaining > 0
            if standby_remaining > 0:
                standby_remaining -= 1

            max_px = standby_max_pixels if high_res else MAX_PIXELS
            frames = [_resize_frame(frame, max_px) for frame in raw_frames]

            round_idx = frame_start // frames_per_round
            time_start = frame_start / actual_fps
            time_end = (frame_start + len(frames)) / actual_fps
            answer = session.step(
                frames,
                round_idx=round_idx,
                frame_max_pixels=max_px,
                time_start=time_start,
                time_end=time_end,
            )
            if "</Standby>" in answer:
                standby_remaining = 1

            tag = " [HIGH-RES]" if high_res else ""
            print(f"[{time_start:g}s-{time_end:g}s]{tag} {answer}",flush=True)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
