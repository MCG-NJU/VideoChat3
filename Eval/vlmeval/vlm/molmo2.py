import torch
import logging
from PIL import Image
import numpy as np
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

class Molmo2(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self, model_path='allenai/Molmo2-4B', **kwargs):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except Exception as e:
            logging.critical('Please install transformers before using Molmo2.')
            raise e

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        # Monkeypatch the video processor method directly to handle dictionary inputs
        orig_decode = self.processor.video_processor._decode_and_sample_videos
        def patched_decode(videos, video_metadata=None, **kwargs):
            # videos can be [[video_dict]], [video_dict], or video_dict
            def _extract_dict(v_input):
                if isinstance(v_input, dict):
                    v_meta = {
                        "total_num_frames": len(v_input["frames"]),
                        "frames_indices": list(range(len(v_input["frames"])))
                    }
                    if "sampled_fps" in v_input: v_meta["fps"] = v_input["sampled_fps"]
                    return v_input["frames"], v_meta
                elif isinstance(v_input, list):
                    new_v = []
                    new_m = []
                    for v in v_input:
                        f, m = _extract_dict(v)
                        new_v.append(f)
                        new_m.append(m)
                    return new_v, new_m
                else:
                    return v_input, None

            if isinstance(videos, (list, dict)):
                new_videos, new_metadata = _extract_dict(videos)
                # If the original was a list of dicts or list of list of dicts, we use new_metadata
                if video_metadata is None and new_metadata is not None:
                    video_metadata = new_metadata
                videos = new_videos

            return orig_decode(videos=videos, video_metadata=video_metadata, **kwargs)

        self.processor.video_processor._decode_and_sample_videos = patched_decode

        self.kwargs = kwargs
        self.model_name = model_path

    def generate_inner(self, message, dataset=None):

        has_video = any(msg['type'] == 'video' for msg in message)
        has_image = any(msg['type'] == 'image' for msg in message)

        if has_video and has_image:
            # Molmo2 does not support pixel_values and pixel_values_videos at the same time.
            # We append images as additional frames to the video.
            video_msg = next(msg for msg in message if msg['type'] == 'video')
            video_val = video_msg['value']
            
            frames = []
            if isinstance(video_val, list):
                for path in video_val:
                    if isinstance(path, str):
                        img = Image.open(path).convert("RGB")
                    else:
                        img = path.convert("RGB")
                    frames.append(np.array(img))
            elif isinstance(video_val, dict) and "frames" in video_val:
                frames = list(video_val["frames"])
            else:
                frames = list(video_val)
                
            for msg in message:
                if msg['type'] == 'image':
                    val = msg['value']
                    if isinstance(val, str):
                        img = Image.open(val).convert("RGB")
                    else:
                        img = val.convert("RGB")
                    frames.append(np.array(img))
                    
            frames = np.stack(frames)
            video_dict = {"frames": frames}
            _fps = video_msg.get('sample_fps')
            if _fps is not None and _fps > 0:
                video_dict["sampled_fps"] = _fps
                video_dict["timestamps"] = np.arange(len(frames)) / _fps
                
            new_message = []
            for msg in message:
                if msg['type'] == 'video':
                    new_message.append(dict(type='video', value=video_dict))
                elif msg['type'] == 'image':
                    continue
                else:
                    new_message.append(msg)
            message = new_message

        content = []
        for msg in message:
            if msg['type'] == 'text':
                content.append(dict(type="text", text=msg['value']))
            elif msg['type'] == 'image':
                val = msg['value']
                if isinstance(val, str):
                    image = Image.open(val)
                else:
                    image = val
                if image.mode != "RGB":
                    image = image.convert("RGB")
                content.append(dict(type="image", image=image))
            elif msg['type'] == 'video':
                video_val = msg['value']
                if isinstance(video_val, list):
                    # List of frame paths
                    frames = []
                    for path in video_val:
                        if isinstance(path, str):
                            img = Image.open(path).convert("RGB")
                        else:
                            img = path.convert("RGB")
                        frames.append(np.array(img))
                    frames = np.stack(frames)
                    
                    video_dict = {"frames": frames}
                    _fps = msg.get('sample_fps')
                    if _fps is not None and _fps > 0:
                        video_dict["sampled_fps"] = _fps
                        video_dict["timestamps"] = np.arange(len(frames)) / _fps
                    
                    content.append(dict(type="video", video=video_dict))
                else:
                    content.append(dict(type="video", video=video_val))

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        max_new_tokens = self.kwargs.get('max_new_tokens', 2048)
        
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        generated_text = self.processor.post_process_image_text_to_text(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return generated_text
