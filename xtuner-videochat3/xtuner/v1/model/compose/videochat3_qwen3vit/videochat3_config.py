from pathlib import Path
from typing import Literal, Optional, Any

from mmengine import is_installed
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from xtuner.v1.float8 import Float8Config
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig, Qwen3Dense4BConfig, Qwen3Dense1_7BConfig
from xtuner.v1.utils import get_logger


logger = get_logger()


class VideoChat3VisionConfig(BaseModel):
    model_config = ConfigDict(
        title="VideoChat3 vision config for xtuner",
        extra="forbid",
    )
    # 基于VideoChat3-debug/config.json的vision_config
    model_type: str = "moonvit"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    hidden_act: str = "gelu"
    patch_size: int = 14  # 从16改为14
    merge_kernel_size: list[int] = [2, 2]  # 新增
    temporal_patch_size: int = 1  # 从2改为1
    temporal_merge_size: int = 4  # 新增
    init_pos_emb_height: int = 64  # 新增
    init_pos_emb_width: int = 64  # 新增
    in_channels: int = 3
    initializer_range: float = 0.02
    torch_dtype: str = "bfloat16"  # 新增
    float8_cfg: Optional["Float8Config"] = None
    attn_impl: Literal["flash_attention_2", "eager_attention"] = "eager_attention"

    def model_post_init(self, __context: Any) -> None: 
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention_2":
            logger.warning("flash-attn-2 is not installed, using `eager_attention` instead.")
            self.attn_impl = "eager_attention"

    def build(self):
        from .modeling_vision import VideoChat3VisionModel

        return VideoChat3VisionModel(self)


class VideoChat3ProjectorConfig(BaseModel):
    # 基于KimiVLMultiModalProjector的配置
    vision_hidden_size: int = 1152
    text_hidden_size: int = 2048
    merge_kernel_size: list[int] = [2, 2]  # 与vision_config保持一致
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import VideoChat3MultiModalProjector

        return VideoChat3MultiModalProjector(self)


class VideoChat3BaseConfig(BaseModel):
    model_config = ConfigDict(
        title="Base VideoChat3 model config for xtuner",
        extra="forbid",
    )
    vision_config: VideoChat3VisionConfig
    projector_config: VideoChat3ProjectorConfig
    text_config: TransformerConfig

    # 基于VideoChat3-debug/config.json
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    
    # 基于video_preprocessor_config.json和config.json
    patch_size: int = 14
    temporal_patch_size: int = 1
    merge_size: int = 2
    temporal_merge_size: int = 4
    merge_kernel_size: list[int] = [2, 2]  # 与vision_config保持一致
    
    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False
    dcp_ignore_frozen_params: bool = True  # align with VisionComposeConfigProtocol

    def build(self) -> "VideoChat3ForConditionalGeneration":
        from .modeling_videochat3 import VideoChat3ForConditionalGeneration

        return VideoChat3ForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError

class VideoChat3Dense8BConfig(VideoChat3BaseConfig):
    # 简化版本，使用dense模型而不是MoE
    vision_config: VideoChat3VisionConfig = VideoChat3VisionConfig(attn_impl="flash_attention_2")
    projector_config: VideoChat3ProjectorConfig = VideoChat3ProjectorConfig(text_hidden_size=4096)
    text_config: Qwen3Dense8BConfig = Qwen3Dense8BConfig(vocab_size=151936)

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None

class VideoChat3Dense4BConfig(VideoChat3BaseConfig):
    vision_config: VideoChat3VisionConfig = VideoChat3VisionConfig(attn_impl="flash_attention_2")
    projector_config: VideoChat3ProjectorConfig = VideoChat3ProjectorConfig(text_hidden_size=2560)
    text_config: Qwen3Dense4BConfig = Qwen3Dense4BConfig(vocab_size=151936)

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None

class VideoChat3Dense4BT1Config(VideoChat3BaseConfig):
    vision_config: VideoChat3VisionConfig = VideoChat3VisionConfig(attn_impl="flash_attention_2", temporal_merge_size=1)
    projector_config: VideoChat3ProjectorConfig = VideoChat3ProjectorConfig(text_hidden_size=2560)
    text_config: Qwen3Dense4BConfig = Qwen3Dense4BConfig(vocab_size=151936)

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None

class VideoChat3Dense2BConfig(VideoChat3BaseConfig):
    vision_config: VideoChat3VisionConfig = VideoChat3VisionConfig(attn_impl="flash_attention_2")
    projector_config: VideoChat3ProjectorConfig = VideoChat3ProjectorConfig(text_hidden_size=2048)
    text_config: Qwen3Dense1_7BConfig = Qwen3Dense1_7BConfig(vocab_size=151936)

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None