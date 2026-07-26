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


class VideoChat3OryxVisionConfig(BaseModel):
    model_config = ConfigDict(
        title="VideoChat3-Oryx vision config for xtuner (SigLIP ViT, no RoPE)",
        extra="forbid",
    )
    model_type: str = "oryx_siglip"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    hidden_act: str = "gelu"
    patch_size: int = 16
    mlp_ratio: float = 3.7362
    merge_kernel_size: list[int] = [2, 2]
    temporal_patch_size: int = 1
    temporal_merge_size: int = 4
    init_pos_emb_height: int = 128
    init_pos_emb_width: int = 128
    in_channels: int = 3
    initializer_range: float = 0.02
    torch_dtype: str = "bfloat16"
    float8_cfg: Optional["Float8Config"] = None
    attn_impl: Literal["flash_attention_2", "eager_attention"] = "eager_attention"

    def model_post_init(self, __context: Any) -> None:
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention_2":
            logger.warning("flash-attn-2 is not installed, using `eager_attention` instead.")
            self.attn_impl = "eager_attention"

    def build(self):
        from .modeling_vision import VideoChat3OryxVisionModel
        return VideoChat3OryxVisionModel(self)


class VideoChat3OryxProjectorConfig(BaseModel):
    vision_hidden_size: int = 1152
    text_hidden_size: int = 2048
    merge_kernel_size: list[int] = [2, 2]
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import VideoChat3OryxMultiModalProjector
        return VideoChat3OryxMultiModalProjector(self)


class VideoChat3OryxBaseConfig(BaseModel):
    model_config = ConfigDict(
        title="Base VideoChat3-Oryx model config for xtuner",
        extra="forbid",
    )
    vision_config: VideoChat3OryxVisionConfig
    projector_config: VideoChat3OryxProjectorConfig
    text_config: TransformerConfig

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

    patch_size: int = 16
    temporal_patch_size: int = 1
    merge_size: int = 2
    temporal_merge_size: int = 4
    merge_kernel_size: list[int] = [2, 2]

    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False
    dcp_ignore_frozen_params: bool = True

    def build(self) -> "VideoChat3OryxForConditionalGeneration":
        from .modeling_videochat3_oryx import VideoChat3OryxForConditionalGeneration
        return VideoChat3OryxForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError


class VideoChat3OryxDense4BConfig(VideoChat3OryxBaseConfig):
    vision_config: VideoChat3OryxVisionConfig = VideoChat3OryxVisionConfig(attn_impl="flash_attention_2")
    projector_config: VideoChat3OryxProjectorConfig = VideoChat3OryxProjectorConfig(text_hidden_size=2560)
    text_config: Qwen3Dense4BConfig = Qwen3Dense4BConfig(vocab_size=151936)

    @property
    def hf_config(self):
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None
