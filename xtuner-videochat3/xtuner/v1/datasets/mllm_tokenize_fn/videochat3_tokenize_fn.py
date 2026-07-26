# Copyright (c) OpenMMLab. All rights reserved.

import copy
import io
import os
import sys
import math
import torch
import numpy as np
from PIL import Image
from typing import Union
try:
    from petrel_client.client import Client
except ImportError:
    Client = None
from pydantic import ConfigDict
from collections.abc import Sequence
from transformers import AutoProcessor, PreTrainedTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from .video_utils import VideoChat3VideoMetadata, VIDEO_READER_MAP
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
from xtuner.v1.utils import get_logger

from ..data_item import CacheItem, VideoChat3DataItem
from ..utils import apply_exif_orientation
from .base_mllm_tokenize_fn import BaseMLLMTokenizeFnConfig, BaseMLLMTokenizeFunction, load_image, replace_image_token

logger = get_logger()

IMAGE_TOKEN_ALIAS = "XTUNER-ALIAS-ALIAS-XTUNER-2025"


def smart_video_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 1,
    factor: int = 28,
    frame_min_pixels: int = 16 * 28 * 28 * 4,
    frame_max_pixels: int = 1024 * 28 * 28 * 4,
    video_max_total_pixels: int = 5000 * 28 * 28 * 4,
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

    h_bar, w_bar = smart_resize(height, width, factor, frame_min_pixels, frame_max_pixels)
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > video_max_total_pixels:
        beta = math.sqrt((num_frames * height * width) / video_max_total_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)

    return h_bar, w_bar

def smart_get_video_thw(video_meta: VideoChat3VideoMetadata, video_processor):
    num_sampled_frames = video_processor.get_num_sampled_frames(video_meta, num_frames=video_processor.num_frames, fps=video_processor.fps)
    
    assert num_sampled_frames != 0, f"num_sampled_frames must be greater than 0, video_meta: {video_meta}"
    
    if video_meta.specified_wh is not None:
        resized_width, resized_height = video_meta.specified_wh
    else:
        resized_height, resized_width = smart_video_resize(
            num_frames=num_sampled_frames,
            height=video_meta.height,
            width=video_meta.width,
            temporal_factor=video_processor.temporal_patch_size,
            factor=video_processor.patch_size * video_processor.merge_size,
            frame_min_pixels=video_processor.size["shortest_edge"],
            frame_max_pixels=video_processor.size["longest_edge"],
            video_max_total_pixels=video_processor.video_max_total_pixels,
        )

    if video_meta.clip_start_time is not None and video_meta.clip_end_time is not None:
        video_meta.frames_indices = np.linspace(video_meta.clip_start_time * video_meta.fps, video_meta.clip_end_time * video_meta.fps, num_sampled_frames).round().astype(int)
    else:
        video_meta.frames_indices = np.linspace(0, video_meta.total_num_frames - 1, num_sampled_frames).round().astype(int)

    grid_t = num_sampled_frames
    grid_h, grid_w = resized_height // video_processor.patch_size, resized_width // video_processor.patch_size
    return [grid_t, grid_h, grid_w]

def smart_resize_long_side(w, h, max_long_side: int = 6500):
    """
    调整图像尺寸，保证长边不超过指定最大值，短边按比例缩放
    :param img: PIL Image 对象
    :param max_long_side: 长边最大限制（默认6500）
    :return: 调整后的 PIL Image 对象
    """
    # 计算当前长边长度
    current_max_side = max(w, h)
    
    # 若长边未超过限制，直接返回原图宽高
    if current_max_side <= max_long_side:
        return w, h
    
    # 计算缩放比例（保留浮点数精度）
    scale = max_long_side / current_max_side
    # 计算新的宽高（转整数，避免小数像素）
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    return new_w, new_h
    
def smart_get_image_thw(image_size, image_processor):
    orig_width, orig_height = image_size

    orig_width, orig_height = smart_resize_long_side(orig_width, orig_height, max_long_side=6500)  # NOTE: 暂时写死，为了RoPE2D不超长

    resized_height, resized_width = smart_resize(
        orig_height,
        orig_width,
        factor=image_processor.patch_size * image_processor.merge_size,
        min_pixels=image_processor.size["shortest_edge"],
        max_pixels=image_processor.size["longest_edge"],
    )
    grid_t = 1  # 单图
    grid_h, grid_w = resized_height // image_processor.patch_size, resized_width // image_processor.patch_size
    return [grid_t, grid_h, grid_w]

def resize_long_side(img: Image.Image, max_long_side: int = 6500) -> Image.Image:
    """
    调整图像尺寸，保证长边不超过指定最大值，短边按比例缩放
    :param img: PIL Image 对象
    :param max_long_side: 长边最大限制（默认6500）
    :return: 调整后的 PIL Image 对象
    """
    # 获取原始宽高 (width, height)
    w, h = img.size
    # 计算当前长边长度
    current_max_side = max(w, h)
    
    # 若长边未超过限制，直接返回原图
    if current_max_side <= max_long_side:
        return img
    
    # 计算缩放比例（保留浮点数精度）
    scale = max_long_side / current_max_side
    # 计算新的宽高（转整数，避免小数像素）
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    # 兼容不同PIL版本的插值算法
    try:
        # PIL 9.1.0+ 推荐使用 Resampling.LANCZOS（最高质量的下采样）
        resample_mode = Image.Resampling.LANCZOS
    except AttributeError:
        # 旧版PIL用 Image.ANTIALIAS（等价于LANCZOS）
        resample_mode = Image.ANTIALIAS
    
    # 执行缩放（resize返回新对象，不修改原图）
    resized_img = img.resize((new_w, new_h), resample=resample_mode)
    return resized_img




class VideoChat3TokenizeFunction(BaseMLLMTokenizeFunction):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor_path: str,
        anno_name: str,
        image_min_pixels: int | None = 0,  # Max image pixels (H*W) for image
        image_max_pixels: int | None = 100000 * 4 * 28 * 28,  # Min image pixels (H*W) for image
        frame_min_pixels: int | None = 0,  # Max image pixels (H*W) for frame
        frame_max_pixels: int | None = 100000 * 4 * 28 * 28,  # Min image pixels (H*W) for frame
        video_max_total_pixels: int = 1000000000 * 4 * 28 * 28,  # Max pixels within a video
        video_min_frames: int = 4,  # Min frames per video
        video_max_frames: int = 2048,  # Max frames per video
        fixed_num_sampled_frames: int | None = None,
        video_sample_fps: Union[int, float] = 2,  # Sample fps for video
        video_read_type: str | None = None,
        system_message: str | None = None,
        add_vision_id: bool = False,
        max_length: int | None = None,
        tokenizer_hash: str | None = None,
        hash: str | None = None,
    ):
        self.ceph_client = Client(conf_path='~/petreloss.conf')       
        self.media_processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        self.image_processor = self.media_processor.image_processor
        self.video_processor = self.media_processor.video_processor

        if image_min_pixels is not None:
            self.image_processor.min_pixels = image_min_pixels
        if image_max_pixels is not None:
            self.image_processor.max_pixels = image_max_pixels
        self.image_processor.size["shortest_edge"] = self.image_processor.min_pixels
        self.image_processor.size["longest_edge"] = self.image_processor.max_pixels

        self.video_max_total_pixels = video_max_total_pixels
        self.video_processor.video_max_total_pixels = video_max_total_pixels
        self.video_processor.size["shortest_edge"] = frame_min_pixels
        self.video_processor.size["longest_edge"] = frame_max_pixels
        self.video_processor.min_frames = video_min_frames
        self.video_processor.max_frames = video_max_frames
        self.video_processor.num_frames = fixed_num_sampled_frames
        self.video_processor.fps = video_sample_fps
        self.video_read_type = video_read_type

        self.add_vision_id = add_vision_id
    
        self.spatial_merge_length = self.image_processor.merge_size**2
        self.temporal_merge_length = self.video_processor.temporal_merge_size

        self.data_name = os.path.basename(anno_name)
        logger.info(
            f"[{self.data_name}] image_min_pixels: {self.image_processor.min_pixels}, image_max_pixels: {self.image_processor.max_pixels},"
            f"video_max_total_pixels: {self.video_max_total_pixels}, frame_min_pixels: {frame_min_pixels}, frame_max_pixels: {frame_max_pixels},"
            f"video_min_frames: {video_min_frames}, video_max_frames: {video_max_frames}, fixed_num_sampled_frames: {fixed_num_sampled_frames}, video_sample_fps: {video_sample_fps},"
            f"video_read_type: {self.video_read_type}, add_vision_id: {self.add_vision_id}, "
            f"spatial_merge_length: {self.spatial_merge_length}, temporal_merge_length: {self.temporal_merge_length}"
        )

        self.chat_template = copy.deepcopy(CHAT_TEMPLATE_MAP["videochat3"])
        if system_message is not None:
            self.chat_template.default_system = system_message

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_context_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.video_context_token)

        # Note: 比较重要，防止改了参数但是没有重新 cache
        self._hash_str = (
            f"{self.image_processor.min_pixels}_"
            f"{self.image_processor.max_pixels}_"
            f"{self.video_max_total_pixels}_"
            f"{frame_min_pixels}_"
            f"{frame_max_pixels}_"
            f"{video_min_frames}_"
            f"{video_max_frames}_"
            f"{fixed_num_sampled_frames}_"
            f"{video_sample_fps}_"
            f"{self.video_read_type}_"
            f"{self.add_vision_id}_"
            f"{self.spatial_merge_length}_"
            f"{self.temporal_merge_length}_"
            f"{system_message}_{max_length}"
        )

        # 必须要最后调用
        super().__init__(tokenizer, self.chat_template, max_length, tokenizer_hash, hash)

    def _truncated_data_item(
        self, input_ids: list[int], labels: list[int] | None = None):
        if self.max_length is not None and len(input_ids) > self.max_length:
            logger.warning(
                f"WARNING: input_ids length {len(input_ids)} exceeds model_max_length {self.max_length}. truncated!"
            )
            input_ids = input_ids[: self.max_length]
            if labels is not None:
                labels = labels[: self.max_length]

        return input_ids, labels

    def _my_load_image(self, image_path: str):
        try:
            if "s3://" in image_path:
                image_path = image_path.replace("https://", "https:/") # make ceph happy!
                if self.ceph_client is None:
                    raise RuntimeError("Ceph client is not available. Cannot load image from s3:// path.")
                img_bytes = self.ceph_client.get(image_path)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                # Local file path
                return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            raise e
            return None


    def _process_image(self, image_file: str, media_root: str = ""):
        processor = copy.deepcopy(self.image_processor)
        if "s3://" in media_root:  # for ceph
            if media_root[-1] != '/':
                media_root += '/'
            image_file = media_root + image_file
        elif media_root != '':  # for local image
            image_file = os.path.join(media_root, image_file)

        image = self._my_load_image(image_file)
        image = apply_exif_orientation(image)
        image = resize_long_side(image, max_long_side=6500)  # NOTE: 暂时写死，为了RoPE2D不超长
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            assert len(image_tensor) == 1, f"image_tensor should have only one element, but got {len(image_tensor)}"
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw


    def _my_load_video(self, video_path: str, video_meta: VideoChat3VideoMetadata, frame_sample_indices: list):
        video_backend = self.video_read_type or (video_meta.video_backend if video_meta and video_meta.video_backend else "decord")
        if "s3://" in video_path and self.ceph_client is None:
            raise RuntimeError("Ceph client is not available. Cannot load video from s3:// path.")
        if video_backend not in VIDEO_READER_MAP:
            raise ValueError(f"Unsupported video_backend: {video_backend}")
        
        frames = VIDEO_READER_MAP[video_backend](
            video_path,
            frame_sample_indices,
            self.ceph_client,
            video_meta=video_meta,
        )
        assert len(frames) == len(frame_sample_indices), f"len(frames) {len(frames)} != len(frame_sample_indices) {len(frame_sample_indices)}"
        real_w, real_h = frames[0].size
        assert video_meta.width == real_w, f"video_meta.width {video_meta.width} != real_w {real_w}"
        assert video_meta.height == real_h, f"video_meta.height {video_meta.height} != real_h {real_h}"
        return frames


    def _process_video(self, video_file: str, media_root: str = "", video_meta: VideoChat3VideoMetadata = None): # TODO
        processor = copy.deepcopy(self.video_processor)
        if "s3://" in media_root:  # for ceph
            if media_root[-1] != '/':
                media_root += '/'
            video_path = media_root + video_file
        elif media_root != '':  # for local image
            video_path = os.path.join(media_root, video_file)
        else:
            video_path = video_file
        # if "s3://" in video_path: NOTE: 无论在不在ceph都走这个
        # 让ceph读取操作对hf processor不可见，首先我们需要明白video processor的主要处理逻辑
        """
        def preprocess(self, videos: VideoInput, **kwargs: Unpack[VideosKwargs]) -> BatchFeature:
            ...
            input_data_format = kwargs.pop("input_data_format")
            do_sample_frames = kwargs.pop("do_sample_frames")
            device = kwargs.pop("device")
            video_metadata = kwargs.pop("video_metadata")
            # 1. 进行decode和采帧，_decode_and_sample_videos由video_processing_videochat3.py中重写过
            sample_indices_fn = partial(self.sample_frames, **kwargs) if do_sample_frames else None
            videos, video_metadata = self._decode_and_sample_videos(
                videos,
                video_metadata=video_metadata,
                do_sample_frames=do_sample_frames,
                sample_indices_fn=sample_indices_fn,
            )
            # 2. 转成pytorch tensor并reshape
            videos = self._prepare_input_videos(videos=videos, input_data_format=input_data_format, device=device)
            ...
            # 3. 对读出来的video pixel tensor进行数据预处理
            preprocessed_videos = self._preprocess(videos=videos, **kwargs)
            ...
            return preprocessed_videos
        """
        # 因此我们只需要考虑读取ceph的操作怎么和video_processing_videochat3._decode_and_sample_videos配合,
        # 最方便的办法就是我们在外面decode加采帧，直接给一个PIL.Image list进去然后do_sample_frames设为False即可
        
        frame_sample_indices = processor.sample_frames(metadata=video_meta, num_frames=processor.num_frames, fps=processor.fps)
        video_meta.frames_indices = frame_sample_indices
        frames = self._my_load_video(video_path, video_meta, frame_sample_indices)
        
        do_resize = True
        if video_meta.specified_wh is not None:
            s_w, s_h = video_meta.specified_wh
            frames = [frame.resize((s_w, s_h), resample=Image.BICUBIC) for frame in frames]
            do_resize = False
            
        visual_processed = processor.preprocess(frames, do_sample_frames=False, do_resize=do_resize, return_tensors="pt", video_metadata=video_meta, return_metadata=True)
        # else:
        # assert os.path.exists(video_path), f"video_path {video_path} does not exist!"
        # if os.path.isdir(video_path):
        #     assert video_meta is not None, "video_meta is required for video in directory"
        #     video_path = sorted(os.listdir(video_path))
        # visual_processed = processor.preprocess(video_path, return_tensors="pt", video_metadata=video_meta, return_metadata=True)

        
        video_tensor = visual_processed["pixel_values_videos"]
        assert len(visual_processed["video_grid_thw"]) == 1, f"video_grid_thw should have only one element, but got {len(visual_processed['video_grid_thw'])}"
        grid_thw = visual_processed["video_grid_thw"][0]

        if isinstance(video_tensor, list):
            assert len(video_tensor) == 1, f"video_tensor should have only one element, but got {len(video_tensor)}"
            video_tensor = video_tensor[0]
        
        video_meta = visual_processed["video_metadata"]
        # 如果video_meta是tuple，取第一个元素
        if isinstance(video_meta, tuple):
            video_meta = video_meta[0]
        # 如果video_meta是列表，取第一个元素
        if isinstance(video_meta, list):
            assert len(video_meta) == 1, f"video_meta should have only one element, but got {len(video_meta)}"
            video_meta = video_meta[0]
        return video_tensor, grid_thw, video_meta

    def _replace_image_token(self, messages: ChatMessages, num_image_tokens_list: list[int], add_vision_id: bool = False):
        chat_template = self.chat_template
        current_image_idx = 0
        for msg in messages.messages:
            if msg.role == "user":
                content = msg.content
                if isinstance(content, list):
                    for c in content:
                        if c.type == "text":
                            text = c.text
                            # assert "<IMG_CONTEXT>" in text
                            text = text.replace("<IMG_CONTEXT>", IMAGE_TOKEN_ALIAS)
                            image_cnt = text.count(IMAGE_TOKEN_ALIAS)
                            for _ in range(image_cnt):
                                # 使用完整的vision token格式：<|vision_start|><|image_pad|><|vision_end|>
                                image_tokens = f"{chat_template.image_start_token}{chat_template.image_context_token * num_image_tokens_list[current_image_idx]}{chat_template.image_end_token}"  # type: ignore
                                if add_vision_id and image_cnt > 1:
                                    # add vision id for each image when there are multiple images
                                    image_tokens = f"Picture {current_image_idx + 1}: " + image_tokens
                                text = text.replace(IMAGE_TOKEN_ALIAS, image_tokens, 1)
                                current_image_idx += 1
                            c.text = text
        # if current_image_idx < num_image, it means <image> placeholder is less than num_image
        assert current_image_idx == len(num_image_tokens_list), (
            f"ERROR: current_image_idx: {current_image_idx} != num_image: {len(num_image_tokens_list)}"
        )

    def _replace_video_token(self, messages: ChatMessages, video_grid_thw: list[int], add_vision_id: bool = False):
        chat_template = self.chat_template
        merge_length = self.video_processor.merge_size ** 2
        current_video_idx = 0
        for msg in messages.messages:
            if msg.role == "user":
                content = msg.content
                if isinstance(content, list):
                    for c in content:
                        if c.type == "text":
                            text = c.text
                            # assert "<VIDEO_CONTEXT>" in text
                            text = text.replace("<VIDEO_CONTEXT>", IMAGE_TOKEN_ALIAS)
                            video_cnt = text.count(IMAGE_TOKEN_ALIAS)

                            for _ in range(video_cnt):
                                
                                curr_timestamp = self.media_processor._calculate_timestamps(self._video_meta_list[current_video_idx], self.video_processor.temporal_merge_size)
                                video_tokens = ""
                                frame_seqlen = video_grid_thw[current_video_idx][1:].prod() // merge_length
                                for curr_time in curr_timestamp:
                                    video_tokens += f"<{curr_time:.1f} seconds>"
                                    video_tokens += (
                                        chat_template.image_start_token + chat_template.video_context_token * frame_seqlen + chat_template.image_end_token
                                    )

                                if add_vision_id and video_cnt > 1:
                                    # add vision id for each video when there are multiple videos
                                    video_tokens = f"Video {current_video_idx + 1}: " + video_tokens
                                text = text.replace(IMAGE_TOKEN_ALIAS, video_tokens, 1)
                                current_video_idx += 1
                            c.text = text
        # if current_video_idx < num_video, it means <video> placeholder is less than num_video
        assert current_video_idx == len(self._video_meta_list), (
            f"ERROR: current_video_idx: {current_video_idx} != num_video: {len(self._video_meta_list)}"
        )

    def pure_text_get_item(self, data_item: dict) -> VideoChat3DataItem:
        messages = ChatMessages(messages=data_item["messages"])
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels: list[int] = tokenized["labels"]

        input_ids, labels = self._truncated_data_item(input_ids, labels)

        ret = VideoChat3DataItem(
            input_ids=input_ids,
            labels=labels,
            num_tokens=len(input_ids),
            num_img_tokens=[0],
            num_imgs=[0],
        )
        return ret

    def calc_num_tokens_multi_modal_get_item(self, data_item: dict) -> CacheItem:
        if len(self._video_path) > 0:
            assert len(self._image_path) == 0, "image and video cannot be mixed"
            return self.calc_num_tokens_video_get_item(data_item)
        else:
            assert len(self._video_path) == 0, "image and video cannot be mixed"
            return self.calc_num_tokens_image_get_item(data_item)

    def multi_modal_get_item(self, data_item: dict, media_root: str = "") -> VideoChat3DataItem:
        if len(self._video_path) > 0:
            assert len(self._image_path) == 0, "image and video cannot be mixed"
            return self.video_get_item(data_item, media_root)
        else:
            assert len(self._video_path) == 0, "image and video cannot be mixed"
            return self.image_get_item(data_item, media_root)

    def calc_num_tokens_image_get_item(self, data_item: dict) -> CacheItem:
        try:
            assert len(self._image_wh_list) >= 1, "image must have `hw` attribute when packing data"
            for size in self._image_wh_list:
                if size[0] == 0 or size[1] == 0:
                    # Image is corrupted, flag=0, and this data will be removed later
                    return {"num_tokens": 0}  # type: ignore
        except Exception as e:
            print(f"ERROR of image_wh: {e}, data_name: {self.data_name}")
            return {"num_tokens": 0}  # type: ignore

        media_grid_thw = []
        for size in self._image_wh_list:
            media_grid_thw.append(smart_get_image_thw(size, self.image_processor))
        media_grid_thw = torch.tensor(media_grid_thw, dtype=torch.int).reshape(-1, 3)  # type: ignore
        sum_media_grid_thw = media_grid_thw.prod(dim=1) // self.spatial_merge_length  # type: ignore

        messages = ChatMessages(messages=data_item["messages"])
        self._replace_image_token(messages, sum_media_grid_thw, add_vision_id=self.add_vision_id)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]

        input_ids, _ = self._truncated_data_item(input_ids)

        # 如果图片被截断，则该数据丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.image_token_id).sum()
        num_image_tokens_2 = sum_media_grid_thw.sum()
        if num_image_tokens_1 != num_image_tokens_2:
            logger.warning(
                f"num_image_tokens_1.shape {num_image_tokens_1} != num_image_tokens_2.shape {num_image_tokens_2}, "
                f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
            )
            return {"num_tokens": 0}

        return {"num_tokens": len(input_ids)}

    def image_get_item(self, data_item: dict, media_root: str = "") -> VideoChat3DataItem:
        results = [self._process_image(file, media_root) for file in self._image_path]
        image, grid_thw = zip(*results)

        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [merged_thw.prod() // self.spatial_merge_length for merged_thw in grid_thw_merged]  # type: ignore
        messages = ChatMessages(messages=data_item["messages"])
        self._replace_image_token(messages, grid_thw_merged, add_vision_id=self.add_vision_id)  # type: ignore
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]

        input_ids, labels = self._truncated_data_item(input_ids, labels)

        # 如果图片被截断，则该数据要丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.image_token_id).sum()
        num_image_tokens_2 = torch.stack(grid_thw_merged, dim=0).sum()
        # assert 会被捕获，该数据会丢弃
        assert num_image_tokens_1 == num_image_tokens_2, (
            f"num_image_tokens_1数量 {num_image_tokens_1} != num_image_tokens_2数量 {num_image_tokens_2}, "
            f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. 丢弃该数据。"
        )

        num_img_tokens = sum(grid_thw_merged[i].item() + 2 for i in range(len(grid_thw_merged)))

        ret = VideoChat3DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=torch.cat(image, dim=0),
            image_grid_thw=torch.cat([_thw.unsqueeze(0) for _thw in grid_thw], dim=0),  # b,3
            num_tokens=len(input_ids),
            num_img_tokens=[num_img_tokens],
            num_imgs=[len(self._image_path)],
        )
        return ret

    def calc_num_tokens_video_get_item(self, data_item: dict) -> CacheItem:
        try:
            assert len(self._video_meta_list) >= 1, "video must have `video_meta` attribute when packing data"
            for video_meta in self._video_meta_list:
                if video_meta.height == 0 or video_meta.width == 0:
                    # Video is corrupted, flag=0, and this data will be removed later
                    return {"num_tokens": 0}  # type: ignore
        except Exception as e:
            print(f"ERROR of video_meta: {e}, data_name: {self.data_name}")
            return {"num_tokens": 0}  # type: ignore

        media_grid_thw = []
        num_video_tokens_list = []
        for video_meta in self._video_meta_list:
            grid_t, grid_h, grid_w = smart_get_video_thw(video_meta, self.video_processor)
            media_grid_thw.append([grid_t, grid_h, grid_w])
            num_video_tokens_list.append(self.video_processor.get_number_of_video_tokens(grid_t, grid_h, grid_w))
        media_grid_thw = torch.tensor(media_grid_thw, dtype=torch.int).reshape(-1, 3)  # type: ignore
        try:
            messages = ChatMessages(messages=data_item["messages"])
        except Exception as e:
            raise ValueError(f"{data_item}: {e}")
        self._replace_video_token(messages, media_grid_thw, add_vision_id=self.add_vision_id)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]


        input_ids, _ = self._truncated_data_item(input_ids)

        # 如果视频被截断，则该数据丢弃
        num_video_tokens = sum(num_video_tokens_list)
        num_video_tokens_real = (torch.tensor(input_ids) == self.video_token_id).sum()
        if num_video_tokens != num_video_tokens_real:
            logger.warning(
                f"num_video_tokens: {num_video_tokens} != num_video_tokens_real: {num_video_tokens_real}, "
                f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
            )
            return {"num_tokens": 0}

        return {"num_tokens": len(input_ids)}

    def video_get_item(self, data_item: dict, media_root: str = "") -> VideoChat3DataItem:
        results = []
        for i in range(len(self._video_path)):
            video_tensor, grid_thw, self._video_meta_list[i] = self._process_video(self._video_path[i], media_root, self._video_meta_list[i]) 
            results.append((video_tensor, grid_thw))

        video, grid_thw = zip(*results)

        grid_thw_copy = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_copy = [grid_thw_copy]
            grid_thw = [grid_thw]

        num_video_tokens_list = [self.video_processor.get_number_of_video_tokens(grid_t, grid_h, grid_w) for grid_t, grid_h, grid_w in grid_thw_copy]  # type: ignore
        messages = ChatMessages(messages=data_item["messages"])
        self._replace_video_token(messages, grid_thw_copy, add_vision_id=self.add_vision_id)  # type: ignore
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]

        input_ids, labels = self._truncated_data_item(input_ids, labels)

        # 如果图片被截断，则该数据要丢弃
        num_video_tokens = sum(num_video_tokens_list)
        num_video_tokens_real = (torch.tensor(input_ids) == self.video_token_id).sum()
        # assert 会被捕获，该数据会丢弃
        assert num_video_tokens == num_video_tokens_real, (
            f"num_video_tokens: {num_video_tokens} != num_video_tokens_real: {num_video_tokens_real}, "
            f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
        )

        num_img_tokens = 0
        for num_video_token, _thw in zip(num_video_tokens_list, grid_thw_copy):
            if _thw[0].item() % self.video_processor.temporal_merge_size == 0:
                num_clips = _thw[0].item() // self.video_processor.temporal_merge_size
            else:
                num_clips = _thw[0].item() // self.video_processor.temporal_merge_size + 1
            num_img_tokens += num_video_token +  num_clips * 2
        

        ret = VideoChat3DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=torch.cat(video, dim=0),
            image_grid_thw=torch.cat([_thw.unsqueeze(0) for _thw in grid_thw], dim=0),  # b,3
            num_tokens=len(input_ids),
            num_img_tokens=[num_img_tokens],
            num_imgs=[len(self._video_path)],
        )
        return ret


class VideoChat3TokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    processor_path: str
    image_min_pixels: int | None = 0
    image_max_pixels: int | None = 512 * 512 * 28 * 28
    frame_min_pixels: int | None = 0
    frame_max_pixels: int | None = 512 * 512 * 28 * 28
    video_max_total_pixels: int = 1000000000 * 4 * 28 * 28
    video_min_frames: int = 4
    video_max_frames: int = 1024 
    fixed_num_sampled_frames: int | None = None
    video_sample_fps: Union[int, float] = 2 
    video_read_type: str | None = None
    # When handling multiple images, it's helpful to add labels to the images and videos for better reference.
    add_vision_id: bool = False

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> VideoChat3TokenizeFunction:
        return VideoChat3TokenizeFunction(
            tokenizer,
            self.processor_path,
            anno_name,
            image_min_pixels=self.image_min_pixels,
            image_max_pixels=self.image_max_pixels,
            video_min_frames=self.video_min_frames,
            video_max_frames=self.video_max_frames,
            frame_min_pixels=self.frame_min_pixels,
            frame_max_pixels=self.frame_max_pixels,
            fixed_num_sampled_frames=self.fixed_num_sampled_frames,
            video_sample_fps=self.video_sample_fps,
            video_read_type=self.video_read_type,
            video_max_total_pixels=self.video_max_total_pixels,
            add_vision_id=self.add_vision_id,
            max_length=self.max_length,
            system_message=self.system_message,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )