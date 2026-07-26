from functools import partial
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union, Optional
from typing_extensions import override
import numpy as np
import math
from collections.abc import Sequence

from transformers.modeling_outputs import BaseModelOutput
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer

try:
    from timm.layers import DropPath
    has_timm = True
except:
    has_timm = False

try:
    from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_videochat
    has_flash_attn = True
except:
    has_flash_attn = False
    flash_attn_varlen_func_videochat = None

from tqdm import tqdm
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_device, get_torch_device_module, init_params
from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from .videochat3_config import VideoChat3VisionConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from xtuner.v1.utils.compile import maybe_compile
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from xtuner.v1.ops.attn_imp import attn_impl_mapping
from xtuner.v1.model.utils.checkpointing import checkpoint_wrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from xtuner.v1.module import RMSNorm
from xtuner.v1.ops.others import Dropout
from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.utils import get_logger

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


def init_world_mesh():
    device = DEVICE
    world_size = dist.get_world_size()

    # TODO: Support hsdp_sharding_size
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


NORM2FN = {"layer_norm": nn.LayerNorm, "rms_norm": RMSNorm}


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From: https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
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

        self.dim = dim  # Store dim for reset_parameters

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


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self.freqs_cis = None

    def extra_repr(self):
        return (
            f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"
        )

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(grid_thws.device)

        shapes = grid_thws.tolist()
        assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1) for t, h, w in shapes],
            dim=0,
        )
        return freqs_cis


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


class VideoChat3VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, tuple[int, int]] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        max_clip_length: int = 4,
    ):
        super().__init__()
        assert isinstance(patch_size, (int, Sequence)), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_emb = VideoChat3InterpPosEmb(
            height=pos_emb_height, width=pos_emb_width, max_clip_length=max_clip_length, dim=out_dim
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_thws (N, 2): grid height and width

        Returns:
            (L, Cout) tensor
        """
        x = x.view(-1, self.in_dim, self.patch_size[0], self.patch_size[1])
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


def flash_attention_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
):
    """Multi-head attention using flash attention 2.

    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.
            The first element should be 0 and the last element should be q.shape[0].
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.
            The first element should be 0 and the last element should be k.shape[0].

    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """
    # Unified format legal check
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"
    assert k_cu_seqlens[-1] == k.shape[0] == v.shape[0], "k_cu_seqlens must sum to k.shape[0]"
    assert q.dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"unsupported dtype {q.dtype} for multihead attn"

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func_videochat(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
    )
    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SDPA attention.

    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
    """
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
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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


VL_VISION_ATTENTION_FUNCTIONS = { # TODO 写死防止没用flash attention，后面也许要改写法使其支持更高版本的flash attn
    "flash_attention_2": flash_attention_2,
    # "sdpa": sdpa_attention,
    "eager_attention": eager_attention,
}


class VideoChat3VisionLayer(GradientCheckpointingLayer):
    """VideoChat3 vision transformer layer."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_impl: str = "eager_attention",
        activation=F.gelu,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_impl = attn_impl

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = VideoChat3VisionMLP([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_impl]
        attn_out = attn_func(xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens)

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
            self,
            hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set

        Returns:
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attn_out = self.attention_qkvpacked(hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm1(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class VideoChat3VisionMLP(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class VideoChat3VisionEncoder(nn.Module):
    """VideoChat3 vision encoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
    ) -> None:
        super().__init__()

        self.rope_2d = Rope2DPosEmb(block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.ModuleList([VideoChat3VisionLayer(**block_cfg) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(hidden_dim)


    def forward(self, hidden_states: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws)

        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_thws.dtype),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for _, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: list[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        # Get the current sequence
        seq = x[pre_sum : pre_sum + t * h * w]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)  # NOTE: temporal pooling
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs




class VideoChat3VisionModel(BaseModel):
    config: VideoChat3VisionConfig

    def __init__(self, config: VideoChat3VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = VideoChat3VisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            max_clip_length=config.temporal_merge_size,
        )
        self.encoder = VideoChat3VisionEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": ACT2FN["gelu_pytorch_tanh"],
                "attn_bias": True,
                "attn_impl": config.attn_impl,
            },
        )

        self._hf_prefix = "model.vision_tower."
        self._init_load_spec()

    def get_input_embeddings(self):
        return self.patch_embed.pos_emb

    def split_grid_thws_clip_by_clip(self, grid_thws: torch.Tensor) -> torch.Tensor:
        # 将grid_t分割成多段，每段的长度为temporal_merge_size
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
        
        
    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): The grid height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        num_tokens = 0
        import copy
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

        # NOTE: 在 cpu_offload 模式下，mesh 应该是 cuda 的，在 meta fully_shard 后在调用 .to_empty(device=cpu)
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
                layer = checkpoint_wrapper(layer, 
                                        preserve_rng_state=checkpoint_preserve_rng_state,
                                        checkpoint_impl=CheckpointImpl.REENTRANT)

            self.encoder.blocks[layer_idx] = layer

            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy()
                if fsdp_config.cpu_offload
                else None,
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
