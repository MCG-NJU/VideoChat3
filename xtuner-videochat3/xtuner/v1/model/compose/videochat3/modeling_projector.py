from typing_extensions import override
from torch import nn
import torch

from functools import partial

from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.utils import get_device, get_torch_device_module, init_params
from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from .videochat3_config import VideoChat3ProjectorConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.utils.compile import maybe_compile
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from .modeling_vision import init_world_mesh

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class VideoChat3MultiModalProjector(BaseModel):
    config: VideoChat3ProjectorConfig

    def __init__(self, config: VideoChat3ProjectorConfig):
        super().__init__()
        # 基于KimiVLMultiModalProjector的实现
        self.hidden_size = (
            config.vision_hidden_size
            * config.merge_kernel_size[0]
            * config.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(config.vision_hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = get_act_fn("gelu")  # 使用gelu激活函数
        self.linear_2 = nn.Linear(
            self.hidden_size, config.text_hidden_size, bias=True
        )

        self._hf_prefix = "model.multi_modal_projector."
        self._init_load_spec()

    @maybe_compile(fullgraph=True)
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # 适配KimiVLMultiModalProjector的forward方法
        # 如果输入是list，则concat；如果是tensor，则直接使用
        if isinstance(image_features, list):
            image_features = torch.cat(image_features, dim=0)
        
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    # Note: 因为 model 本身就是 self，暂时无法实现在 fully_shard 时候进行 checkpoint
    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):  
        self.fsdp_config = fsdp_config
        assert float8_handler is None
        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None

        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self

    @torch.no_grad()
    def init_weights(self):
        init_params(self.pre_norm.weight, nn.init.ones_)
        init_params(self.pre_norm.bias, nn.init.zeros_)
        init_params(self.linear_1.bias, nn.init.zeros_)
        init_params(self.linear_1.weight, partial(nn.init.normal_, mean=0.0, std=0.02))
        init_params(self.linear_2.bias, nn.init.zeros_)
        init_params(self.linear_2.weight, partial(nn.init.normal_, mean=0.0, std=0.02))
