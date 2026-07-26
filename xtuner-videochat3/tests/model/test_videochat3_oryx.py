"""
Test for VideoChat3-Oryx model (SigLIP ViT replacing MoonVit).

The checkpoint is assembled on-the-fly from:
  - Oryx ViT weights:  ORYX_VIT_PTH  (siglip2_so400m_oryx.pth)
  - VideoChat3-4B HF:  VIDEOCHAT3_ORIG_PATH  (language_model + projector + lm_head)

Weight mapping (Oryx pth → VideoChat3-Oryx HF):
  After stripping prefix "base_model.model.model.vision_tower.vision_tower.":
    patch_embed.proj.*        → model.vision_tower.patch_embed.proj.*
    pos_embed (1,16384,1152)  → model.vision_tower.patch_embed.pos_emb.weight (128,128,1152)
    (new)                     → model.vision_tower.patch_embed.pos_emb.time_weight (4,1,1152)
    blocks.{i}.*              → model.vision_tower.encoder.blocks.{i}.*
"""

import os
import json
from pathlib import Path

import numpy as np
import parametrize
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import save_file, load_file

from xtuner._testing import patch_hf_rms_norm, DeterministicDDPTestCase
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from xtuner.v1.model import VideoChat3OryxDense4BConfig
from xtuner.v1.model.compose.videochat3_oryx.modeling_vision import init_world_mesh
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.utils.test_utils import init_data_mesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)

ORYX_VIT_PTH = os.environ.get(
    "ORYX_VIT_PTH",
    "/mnt/castle/dyh/videochat3/siglip2_so400m_oryx.pth",
)
VIDEOCHAT3_ORIG_PATH = os.environ.get(
    "VIDEOCHAT3_ORIG_PATH",
    "/mnt/castle/dyh/videochat3/videosignal",
)
VIDEOCHAT3_ORYX_DENSE_PATH = "/home/models/VideoChat3-Oryx-4B"
VIDEO_ROOT = "tests/resource"

ORYX_PTH_PREFIX = "base_model.model.model.vision_tower.vision_tower."
HF_VISION_PREFIX = "model.vision_tower."
INIT_POS_EMB_HEIGHT = 128
INIT_POS_EMB_WIDTH = 128
TEMPORAL_MERGE_SIZE = 4
HIDDEN_DIM = 1152


# ---------------------------------------------------------------------------
# Checkpoint assembly helpers
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _build_oryx_vision_state_dict():
    """Load Oryx ViT pth and convert keys to VideoChat3-Oryx HF naming."""
    raw_sd = torch.load(ORYX_VIT_PTH, map_location="cpu")

    stripped = {}
    for k, v in raw_sd.items():
        if k.startswith(ORYX_PTH_PREFIX):
            stripped[k[len(ORYX_PTH_PREFIX):]] = v
    if not stripped:
        stripped = raw_sd

    vision_sd = {}
    for k, v in stripped.items():
        if k == "pos_embed":
            v = v.squeeze(0).reshape(INIT_POS_EMB_HEIGHT, INIT_POS_EMB_WIDTH, HIDDEN_DIM)
            vision_sd[HF_VISION_PREFIX + "patch_embed.pos_emb.weight"] = v
        elif k.startswith("patch_embed."):
            vision_sd[HF_VISION_PREFIX + k] = v
        elif k.startswith("blocks."):
            vision_sd[HF_VISION_PREFIX + "encoder." + k] = v
        else:
            print(f"  [_build_oryx_vision_sd] skipping key: {k}")

    time_weight = torch.from_numpy(
        _get_1d_sincos_pos_embed(HIDDEN_DIM, np.arange(TEMPORAL_MERGE_SIZE, dtype=np.float32))
    ).float().unsqueeze(1)
    vision_sd[HF_VISION_PREFIX + "patch_embed.pos_emb.time_weight"] = time_weight

    return vision_sd


def _load_non_vision_state_dict():
    """Load language_model + projector + lm_head from the original VideoChat3-4B."""
    index_path = Path(VIDEOCHAT3_ORIG_PATH) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        needed_files = set()
        for key, fname in weight_map.items():
            if not key.startswith("model.vision_tower."):
                needed_files.add(fname)
        non_vision_sd = {}
        for fname in needed_files:
            shard = load_file(str(Path(VIDEOCHAT3_ORIG_PATH) / fname))
            for key, val in shard.items():
                if not key.startswith("model.vision_tower."):
                    non_vision_sd[key] = val
    else:
        shard = load_file(str(Path(VIDEOCHAT3_ORIG_PATH) / "model.safetensors"))
        non_vision_sd = {k: v for k, v in shard.items() if not k.startswith("model.vision_tower.")}
    return non_vision_sd


def prepare_checkpoint():
    """Assemble VideoChat3-Oryx HF checkpoint if model.safetensors does not exist."""
    output_path = "/home/models/VideoChat3-Oryx-4B"
    safetensors_path = f"{output_path}/model.safetensors"

    if os.path.exists(safetensors_path):
        print(f"Checkpoint already exists at {safetensors_path}, skipping rebuild.")
        return

    print(f"Building VideoChat3-Oryx checkpoint ...")
    print(f"  Oryx ViT:   {ORYX_VIT_PTH}")
    print(f"  VideoChat3: {VIDEOCHAT3_ORIG_PATH}")

    vision_sd = _build_oryx_vision_state_dict()
    print(f"  Vision keys: {len(vision_sd)}")

    non_vision_sd = _load_non_vision_state_dict()
    print(f"  Non-vision keys: {len(non_vision_sd)}")

    combined = {**non_vision_sd, **vision_sd}
    print(f"  Total keys: {len(combined)}")

    save_file(combined, str(safetensors_path))
    print(f"  Saved to {safetensors_path}")


# ---------------------------------------------------------------------------
# Test class — mirrors TestVideoChat3 from test_videochat3.py
# ---------------------------------------------------------------------------

class TestVideoChat3Oryx(DeterministicDDPTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        rank = int(os.environ.get("RANK", "0"))
        print(f"[STEP] setUpClass: rank={rank}, preparing checkpoint ...")
        if rank == 0:
            prepare_checkpoint()
        if dist.is_initialized():
            dist.barrier()
        print(f"[STEP] setUpClass: checkpoint ready.")

    def _test_all(self, hf_model, videochat3_oryx_model, type, device, sp_size, tol):
        rank = dist.get_rank()
        print(f"[STEP] _test_all: type={type}, rank={rank}, device={device}, sp_size={sp_size}")
        print(f"[STEP]   Preparing input data ...")
        if type == 'image':
            processor = AutoProcessor.from_pretrained(VIDEOCHAT3_ORYX_DENSE_PATH, trust_remote_code=True)
            images = [
                Image.open("tests/resource/mscoco_twocat_000000039769.jpg"),
                Image.open("tests/resource/mscoco_dog_000000319154.jpg"),
            ]
            text = (
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                "请描述下第二幅图片中的狗是什么颜色？<|im_end|>\n"
                "<|im_start|>assistant\n"
                "图片中的狗是棕色的。<|im_end|>\n"
            )
            inputs = processor(images=images, text=text, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            labels = input_ids.clone()
            pixel_values = inputs["pixel_values"].cuda().to(torch.bfloat16)
            image_grid_thw = inputs["image_grid_thw"].cuda()
        elif type == 'video':
            processor = AutoProcessor.from_pretrained(VIDEOCHAT3_ORYX_DENSE_PATH, trust_remote_code=True)
            import decord
            video_path = os.path.join(VIDEO_ROOT, "tennis.mp4")
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            num_sample = min(14, total_frames)
            indices = np.linspace(0, total_frames - 1, num_sample, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
            duration = total_frames / fps

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "videochat3_utils",
                os.path.join(VIDEOCHAT3_ORYX_DENSE_PATH, "videochat3_utils.py"),
            )
            vc3_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vc3_utils)
            VideoMetadata = vc3_utils.VideoChat3VideoMetadata

            meta = VideoMetadata(
                total_num_frames=num_sample,
                fps=fps,
                duration=num_sample / fps,
            )
            meta.frames_indices = indices.tolist()

            video_inputs_1 = processor.video_processor(
                videos=[video_tensor], return_tensors="pt",
                video_metadata=[meta],
                num_frames=num_sample,
                do_sample_frames=False,
            )
            meta2 = VideoMetadata(
                total_num_frames=num_sample,
                fps=fps,
                duration=num_sample / fps,
            )
            meta2.frames_indices = indices.tolist()
            video_inputs_2 = processor.video_processor(
                videos=[video_tensor], return_tensors="pt",
                video_metadata=[meta2],
                num_frames=num_sample,
                do_sample_frames=False,
            )
            pixel_values = torch.cat([
                video_inputs_1["pixel_values_videos"],
                video_inputs_2["pixel_values_videos"],
            ], dim=0).cuda().to(torch.bfloat16)
            image_grid_thw = torch.cat([
                video_inputs_1["video_grid_thw"],
                video_inputs_2["video_grid_thw"],
            ], dim=0).cuda()

            timestamps_1 = [idx / fps for idx in indices]
            timestamps_2 = timestamps_1.copy()

            temporal_merge_size = processor.video_processor.temporal_merge_size
            merge_length = processor.video_processor.merge_size ** 2
            def _build_video_placeholder(grid_thw, timestamps, merge_length, temporal_merge_size):
                frame_seqlen = grid_thw[1:].prod() // merge_length
                new_ts = []
                for i in range(0, len(timestamps), temporal_merge_size):
                    end = min(i + temporal_merge_size - 1, len(timestamps) - 1)
                    new_ts.append((timestamps[i] + timestamps[end]) / 2)
                parts = []
                for t in new_ts:
                    parts.append(f"<{t:.1f} seconds>")
                    parts.append("<|vision_start|>" + "<|video_pad|>" * frame_seqlen + "<|vision_end|>")
                return "".join(parts)

            vid_ph_1 = _build_video_placeholder(image_grid_thw[0], timestamps_1, merge_length, temporal_merge_size)
            vid_ph_2 = _build_video_placeholder(image_grid_thw[1], timestamps_2, merge_length, temporal_merge_size)

            text = (
                f"<|im_start|>user\n"
                f"{vid_ph_1}{vid_ph_2}"
                f"两个视频中都在做什么？<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"打网球<|im_end|>\n"
            )
            tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_ORYX_DENSE_PATH, trust_remote_code=True)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
            labels = input_ids.clone()
        else:
            tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_ORYX_DENSE_PATH, trust_remote_code=True)
            text = f"今天天气不错，是学习的好日子。请听题： 1+{rank} 等于多少？"
            input_ids = tokenizer(f"今天天气不错，是学习的好日子。请听题： 1+{rank} 等于多少？",
                                  return_tensors="pt").input_ids.to(device)
            labels = input_ids.clone()
            pixel_values = None
            image_grid_thw = None

        print(f"[STEP]   Input data ready. input_ids shape={input_ids.shape}, "
              f"pixel_values={'None' if pixel_values is None else pixel_values.shape}, "
              f"image_grid_thw={'None' if image_grid_thw is None else image_grid_thw.shape}, "
              f"text={text}, "
              f"image_grid_thw={image_grid_thw if image_grid_thw is not None else None}, "
              f"type={type}")

        print(f"[STEP]   Running HF model forward ...")
        hf_model.to(device)
        with torch.no_grad():
            if type == 'video':
                output = hf_model(
                    input_ids=input_ids,
                    labels=labels,
                    pixel_values_videos=pixel_values,
                    video_grid_thw=image_grid_thw,
                )
            else:
                output = hf_model(
                    input_ids=input_ids,
                    labels=labels,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
        expected_loss = output.loss
        dist.all_reduce(expected_loss.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
        print(f"[STEP]   HF model loss = {expected_loss.item():.6f}")

        hf_model.to('cpu')
        torch.cuda.empty_cache()

        loss_cfg = CELossConfig()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = labels[:, 1:]

        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        seq_ctx.image_grid_thw = image_grid_thw
        seq_ctx.pixel_values = pixel_values
        seq_ctx.to('cuda')
        loss_ctx_input = CELossContextInputItem(shifted_labels=shifted_labels)
        loss_ctx_input = loss_ctx_input.to('cuda')

        if sp_size > 1:
            seq_ctx = seq_ctx.split(sp_mesh)
            loss_ctx_input = loss_ctx_input.sp_split(sp_mesh)

        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [loss_ctx_input]

        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list,
            loss_cfg,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)
        seq_ctx = seq_ctx_list[0]

        print(f"[STEP]   Running XTuner model forward ...")
        videochat3_oryx_model.to(device)
        with torch.no_grad():
            output = videochat3_oryx_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        videochat3_oryx_model.to('cpu')
        torch.cuda.empty_cache()
        loss = output["loss"]
        print(f"[STEP]   XTuner model loss = {loss.item():.6f}")
        match = torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol)
        print(f"[STEP]   type={type} PASSED (diff={abs(loss.item() - expected_loss.item()):.2e})" if match
              else f"[STEP]   type={type} FAILED (hf={expected_loss.item():.6f}, xtuner={loss.item():.6f})")
        self.assertTrue(match)

    @parametrize.parametrize(
        "device,sp_size,tol",
        [
            ("cuda", 1, 1e-2)
        ],
    )
    def test_videochat3_oryx_run(self, device, sp_size, tol):
        print(f"\n{'='*60}")
        print(f"[TEST] test_videochat3_oryx_run  device={device} sp_size={sp_size} tol={tol}")
        print(f"{'='*60}")
        self.create_pg(device)
        maybe_compile.clear_compile_targets()

        print(f"[STEP] Loading HF model from {VIDEOCHAT3_ORYX_DENSE_PATH} ...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_ORYX_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
            trust_remote_code=True
        ).eval()
        patch_hf_rms_norm(hf_model)
        print(f"[STEP] HF model loaded.")

        print(f"[STEP] Building XTuner model on meta device ...")
        with torch.device("meta"):
            model_cfg = VideoChat3OryxDense4BConfig()
            videochat3_oryx_model = model_cfg.build().to(torch.bfloat16)
        print(f"[STEP] XTuner model built.")

        print(f"[STEP] Loading XTuner model weights from HF checkpoint ...")
        videochat3_oryx_model.from_hf(VIDEOCHAT3_ORYX_DENSE_PATH)
        videochat3_oryx_model.eval()
        videochat3_oryx_model.to('cpu')
        print(f"[STEP] XTuner model weights loaded.")

        self._test_all(hf_model, videochat3_oryx_model, 'text', device, sp_size, tol)
        self._test_all(hf_model, videochat3_oryx_model, 'image', device, sp_size, tol)
        self._test_all(hf_model, videochat3_oryx_model, 'video', device, sp_size, tol)
        print(f"[TEST] test_videochat3_oryx_run DONE")

    @parametrize.parametrize(
        "device,sp_size,compile, tol",
        [
            ("cuda", 1, False, 1e-2)
        ],
    )
    def test_fsdp_videochat3_oryx_run(self, device, sp_size, compile, tol):
        print(f"\n{'='*60}")
        print(f"[TEST] test_fsdp_videochat3_oryx_run  device={device} sp_size={sp_size} compile={compile} tol={tol}")
        print(f"{'='*60}")
        self.create_pg(device)
        if compile is False:
            maybe_compile.clear_compile_targets()

        print(f"[STEP] Loading HF model from {VIDEOCHAT3_ORYX_DENSE_PATH} ...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_ORYX_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
            trust_remote_code=True
        ).eval()
        patch_hf_rms_norm(hf_model)
        print(f"[STEP] HF model loaded.")

        print(f"[STEP] Building XTuner model on meta device ...")
        with torch.device("meta"):
            model_cfg = VideoChat3OryxDense4BConfig()
            videochat3_oryx_model = model_cfg.build().to(torch.bfloat16)
        print(f"[STEP] XTuner model built.")

        print(f"[STEP] Applying FSDP sharding ...")
        fsdp_config = FSDPConfig(
            cpu_offload=False,
            torch_compile=compile
        )

        videochat3_oryx_model.language_model.fully_shard(fsdp_config=fsdp_config)
        videochat3_oryx_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        videochat3_oryx_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        videochat3_oryx_model.fully_shard(fsdp_config=fsdp_config)
        print(f"[STEP] FSDP sharding applied.")

        print(f"[STEP] Loading XTuner model weights from HF checkpoint ...")
        videochat3_oryx_model.from_hf(VIDEOCHAT3_ORYX_DENSE_PATH)
        videochat3_oryx_model.eval()
        videochat3_oryx_model.to('cpu')
        print(f"[STEP] XTuner model weights loaded.")

        self._test_all(hf_model, videochat3_oryx_model, 'text', device, sp_size, tol)
        self._test_all(hf_model, videochat3_oryx_model, 'image', device, sp_size, tol)
        self._test_all(hf_model, videochat3_oryx_model, 'video', device, sp_size, tol)
        print(f"[TEST] test_fsdp_videochat3_oryx_run DONE")

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))


if __name__ == "__main__":
    import unittest
    unittest.main()
