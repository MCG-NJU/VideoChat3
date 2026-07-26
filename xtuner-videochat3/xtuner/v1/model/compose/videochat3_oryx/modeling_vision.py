from functools import partial
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union, Optional
from typing_extensions import override
import numpy as np
import math
from collections.abc import Sequence

from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer

try:
    from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_oryx
    has_flash_attn = True
except Exception:
    has_flash_attn = False
    flash_attn_varlen_func_oryx = None

from tqdm import tqdm
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_device, get_torch_device_module, init_params
from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from .videochat3_oryx_config import VideoChat3OryxVisionConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from xtuner.v1.utils.compile import maybe_compile
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from xtuner.v1.model.utils.checkpointing import checkpoint_wrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from xtuner.v1.utils import get_logger

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


def init_world_mesh():
    device = DEVICE
    world_size = dist.get_world_size()
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


# ---------------------------------------------------------------------------
# Positional embedding (reused from VideoChat3: spatial bicubic + temporal sincos)
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class VideoChat3InterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, max_clip_length: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.max_clip_length = max_clip_length
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        if max_clip_length > 1:
            self.time_weight = nn.Parameter(torch.empty(max_clip_length, 1, dim))
        else:
            self.time_weight = None

        self.dim = dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.time_weight is not None:
            initial_time_weight = (
                torch.from_numpy(get_1d_sincos_pos_embed_from_grid(self.dim, np.arange(self.max_clip_length, dtype=np.float32)))
                .float()
                .unsqueeze(1)
            )
            with torch.no_grad():
                self.time_weight.copy_(initial_time_weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        real_num_tokens = x.shape[0]

        num_tokens = 0
        for t, h, w in grid_thws.tolist():
            num_tokens += t * h * w
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = (
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=(h, w),
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )

            if self.time_weight is None:
                pos_emb_3d = pos_emb_2d
            else:
                if t == 1:
                    pos_emb_3d = pos_emb_2d + self.time_weight.sum() * 0.0
                else:
                    pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        if real_num_tokens != num_tokens:
            raise ValueError(f"x.shape:{x.shape}, grid_thws:{grid_thws}, real_num_tokens={real_num_tokens}, num_tokens={num_tokens}")
        out = x + torch.cat(pos_embs)
        return out


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, tuple[int, int]] = (16, 16),
        pos_emb_height: int = 128,
        pos_emb_width: int = 128,
        max_clip_length: int = 4,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_emb = VideoChat3InterpPosEmb(
            height=pos_emb_height, width=pos_emb_width, max_clip_length=max_clip_length, dim=out_dim
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_dim, self.patch_size[0], self.patch_size[1])
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


# ---------------------------------------------------------------------------
# Attention implementations (no RoPE)
# ---------------------------------------------------------------------------

def flash_attention_2(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    assert q.dim() == k.dim() == v.dim() == 3
    assert q_cu_seqlens[-1] == q.shape[0]
    assert k_cu_seqlens[-1] == k.shape[0] == v.shape[0]
    assert q.dtype in [torch.bfloat16, torch.float16], f"unsupported dtype {q.dtype}"

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func_oryx(
        q, k, v,
        q_cu_seqlens, k_cu_seqlens,
        max_seqlen_q, max_seqlen_k,
        causal=False,
    )
    attn_out = attn_out.flatten(start_dim=-2)
    return attn_out


def eager_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": flash_attention_2,
    "eager_attention": eager_attention,
}


# ---------------------------------------------------------------------------
# Vision MLP (Oryx/timm naming: fc1, fc2)
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionMLP(nn.Module):
    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc1 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc2 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc1, self.fc2]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Vision transformer layer (no RoPE — key difference from VideoChat3)
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_impl: str = "eager_attention",
        activation=F.gelu,
        attn_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = hidden_dim // num_heads
        self.attn_impl = attn_impl

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = VideoChat3OryxVisionMLP([hidden_dim, mlp_dim, hidden_dim], activation)
        self.attn = nn.ModuleDict({
            "qkv": nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias),
            "proj": nn.Linear(hidden_dim, hidden_dim, bias=attn_bias),
        })

    def attention_forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor):
        xqkv = self.attn["qkv"](x)
        qkv_shape = xqkv.size()[:-1] + (3, self.num_heads, self.hidden_size_per_attention_head)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_impl]
        attn_out = attn_func(xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens)

        attn_out = self.attn["proj"](attn_out)
        return attn_out

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_out = self.attention_forward(hidden_states, cu_seqlens)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Vision encoder (no RoPE, no final layernorm)
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([VideoChat3OryxVisionLayer(**block_cfg) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_thws.dtype),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens)

        return hidden_states


# ---------------------------------------------------------------------------
# Patch merger (identical to VideoChat3)
# ---------------------------------------------------------------------------

def patch_merger(x, grid_thws, merge_kernel_size=(2, 2)):
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w
    return outputs


# ---------------------------------------------------------------------------
# Vision model (xtuner BaseModel)
# ---------------------------------------------------------------------------

class VideoChat3OryxVisionModel(BaseModel):
    config: VideoChat3OryxVisionConfig

    def __init__(self, config: VideoChat3OryxVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = VideoChat3OryxVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            max_clip_length=config.temporal_merge_size,
        )
        self.encoder = VideoChat3OryxVisionEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": ACT2FN["gelu"],
                "attn_bias": True,
                "attn_impl": config.attn_impl,
            },
        )

        self._hf_prefix = "model.vision_tower."
        self._init_load_spec()

    def get_input_embeddings(self):
        return self.patch_embed.pos_emb

    def split_grid_thws_clip_by_clip(self, grid_thws: torch.Tensor) -> torch.Tensor:
        tmp_thw_list = []
        for t, h, w in grid_thws.tolist():
            if t > self.config.temporal_merge_size:
                _t = t
                for _ in range(self.config.temporal_merge_size, t, self.config.temporal_merge_size):
                    tmp_thw_list.append([self.config.temporal_merge_size, h, w])
                    _t -= self.config.temporal_merge_size
                if _t != 0:
                    tmp_thw_list.append([_t, h, w])
            else:
                assert t != 0, grid_thws
                tmp_thw_list.append([t, h, w])
        return torch.tensor(tmp_thw_list, device=grid_thws.device, dtype=grid_thws.dtype)

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        import copy
        num_tokens = 0
        old_grid_thws = copy.deepcopy(grid_thws)
        for t, h, w in grid_thws.tolist():
            num_tokens += t * h * w
        grid_thws = self.split_grid_thws_clip_by_clip(grid_thws)
        num_tokens2 = 0
        for t, h, w in grid_thws.tolist():
            num_tokens2 += t * h * w
        assert num_tokens == num_tokens2, f"{num_tokens} != {num_tokens2}, {old_grid_thws} / {grid_thws}"
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        hidden_states = patch_merger(hidden_states, grid_thws, merge_kernel_size=self.config.merge_kernel_size)
        return hidden_states

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        assert float8_handler is None

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        device = "cpu" if fsdp_config.cpu_offload else str(DEVICE)

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

        recompute_ratio = fsdp_config.vision_recompute_ratio
        num_recompute_layers = int(len(self.encoder.blocks) * recompute_ratio)
        generator = torch.Generator()
        generator.manual_seed(dist.get_rank())
        shuffled_layers_idxs = torch.randperm(len(self.encoder.blocks), generator=generator)

        for layer_idx in tqdm(shuffled_layers_idxs, desc="[Vision Fully Shard]"):
            layer = self.encoder.blocks[layer_idx]

            if layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(
                    layer,
                    preserve_rng_state=checkpoint_preserve_rng_state,
                    checkpoint_impl=CheckpointImpl.REENTRANT,
                )

            self.encoder.blocks[layer_idx] = layer

            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
            )

        for layer_cur, layer_next in zip(self.encoder.blocks[:-1], self.encoder.blocks[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self
