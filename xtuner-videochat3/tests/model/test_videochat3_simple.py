import os
from packaging import version
import parametrize
import torch
from xtuner._testing import patch_hf_rms_norm, DeterministicDDPTestCase
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
import tempfile
from pathlib import Path
import json
from safetensors import safe_open
from unittest import skipIf
import transformers
from xtuner.v1.model.compose.videochat3.videochat3_config import VideoChat3Dense4BConfig
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.utils.test_utils import init_data_mesh

# 设置环境变量路径 - 需要根据实际情况调整
VIDEOCHAT3_DENSE_PATH = os.environ.get("VIDEOCHAT3_DENSE_PATH", "./VideoChat3-4B")



class TestVideoChat3(DeterministicDDPTestCase):
    @parametrize.parametrize(
        "device,tol",
        [
            ("cuda", 1e-2),
        ],
    )
    def test_videochat3_text_run(self, device, tol):
        """测试 VideoChat3 模型的纯文本推理"""
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        
        # 加载 HuggingFace 模型作为参考
        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

        rank = dist.get_rank()
        tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH, trust_remote_code=True)
        input_ids = tokenizer(f"今天天气不错，是学习的好日子。请听题： 1+{rank} 等于多少？",
                              return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss
        dist.all_reduce(expected_loss.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)

        del hf_model
        torch.cuda.empty_cache()

        # 构建 XTuner VideoChat3 模型
        with torch.device("meta"):
            model_cfg = VideoChat3Dense4BConfig()
            videochat3_model = model_cfg.build().to(torch.bfloat16)

        videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
        videochat3_model.eval()

        loss_cfg = CELossConfig()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to(device),))

        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=shifted_labels)]
        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list,
            loss_cfg,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)
        seq_ctx = seq_ctx_list[0]

        with torch.no_grad():
            output = videochat3_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,sp_size,tol",
        [
            ("cuda", 1, 1e-2)
        ],
    )
    def test_videochat3_image_run(self, device, sp_size, tol):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        # patch_hf_rms_norm(hf_model)

        rank = dist.get_rank()
        tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH, trust_remote_code=True)
        image_str = '<|vision_start|><|image_pad|><|vision_end|>'
        input_ids = tokenizer(image_str + "吃葡萄不吐葡萄皮" * 20, return_tensors="pt").input_ids.to("cuda")
        pixel_values = torch.randn(4, 588, device='cuda', dtype=torch.bfloat16) # (h*w, 14 * 14 * 3 * 1)
        # TODO: 不合理，为啥一定要每个 rank 数据完全一样才能通过 CI ?
        dist.broadcast(pixel_values, src=0)

        image_grid_thw = torch.tensor([[1, 2, 2]], device='cuda')

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            model_cfg = VideoChat3Dense4BConfig()
            videochat3_model = model_cfg.build().to(torch.bfloat16)

        videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
        videochat3_model.eval()

        loss_cfg = CELossConfig()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]

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

        with torch.no_grad():
            output = videochat3_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,sp_size,tol",
        [
            ("cuda", 1, 1e-2)
        ],
    )
    def test_videochat3_multi_image_run(self, device, sp_size, tol):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_DENSE_PATH,
            dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        # patch_hf_rms_norm(hf_model)

        rank = dist.get_rank()
        tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH, trust_remote_code=True)
        image_str = '看这张图 <|vision_start|><|image_pad|><|vision_end|> 和 这张图<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>'
        input_ids = tokenizer(image_str + "它们有什么区别" * 20, return_tensors="pt").input_ids.to("cuda")
        pixel_values = torch.randn(4 + 12, 588, device='cuda', dtype=torch.float16) # (h*w, 14 * 14 * 3 * 1)
        # TODO: 不合理，为啥一定要每个 rank 数据完全一样才能通过 CI ?
        dist.broadcast(pixel_values, src=0)

        image_grid_thw = torch.tensor([[1, 2, 2], [1, 6, 2]], device='cuda')

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            model_cfg = VideoChat3Dense4BConfig()
            videochat3_model = model_cfg.build().to(torch.float16)

        videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
        videochat3_model.eval()

        loss_cfg = CELossConfig()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]

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

        with torch.no_grad():
            output = videochat3_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))


    @parametrize.parametrize(
        "device,sp_size,tol",
        [
            ("cuda", 1, 1e-2)
        ],
    )
    def test_videochat3_video_run(self, device, sp_size, tol):
        """测试 VideoChat3 模型的视频推理"""
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        
        # 加载 HuggingFace 模型作为参考
        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

        rank = dist.get_rank()
        tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH, trust_remote_code=True)
        # 使用视频相关的特殊 token
        # video_str = '<0 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|> <1 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>'
        video_str = '<0 seconds><|vision_start|><|video_pad|><|vision_end|> <1 seconds><|vision_start|><|video_pad|><|vision_end|>'
        input_ids = tokenizer(video_str + "这个视频中发生了什么？" * 10, return_tensors="pt").input_ids.to("cuda")
        
        # 模拟视频数据：batch_size=1, channels=3, temporal=4, height=28, width=28
        pixel_values_videos = torch.randn(7*2*2, 3 * 14 * 14, device='cuda', dtype=torch.bfloat16)
        # 确保所有 rank 使用相同的数据
        dist.broadcast(pixel_values_videos, src=0)
        # 视频网格信息：temporal=7, height=28, width=28
        video_grid_thw = torch.tensor([[7, 2, 2]], device='cuda')

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        # 构建 XTuner VideoChat3 模型
        with torch.device("meta"):
            model_cfg = VideoChat3Dense4BConfig()
            videochat3_model = model_cfg.build().to(torch.bfloat16)

        videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
        videochat3_model.eval()

        loss_cfg = CELossConfig()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]

        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        seq_ctx.image_grid_thw = video_grid_thw
        seq_ctx.pixel_values = pixel_values_videos
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

        with torch.no_grad():
            output = videochat3_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))



    @parametrize.parametrize(
        "device,sp_size,tol",
        [
            ("cuda", 1, 1e-2)
        ],
    )
    def test_videochat3_video_vit_run(self, device, sp_size, tol):
        """测试 VideoChat3 模型的视频推理"""
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        
        # 加载 HuggingFace 模型作为参考
        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

        rank = dist.get_rank()
        tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH, trust_remote_code=True)
        # 使用视频相关的特殊 token
        # video_str = '<0 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|> <1 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>'
        video_str = '<0 seconds><|vision_start|><|video_pad|><|vision_end|> <1 seconds><|vision_start|><|video_pad|><|vision_end|>'
        input_ids = tokenizer(video_str + "这个视频中发生了什么？" * 10, return_tensors="pt").input_ids.to("cuda")
        
        # 模拟视频数据：batch_size=1, channels=3, temporal=4, height=28, width=28
        pixel_values_videos = torch.randn(7*2*2, 3 * 14 * 14, device='cuda', dtype=torch.bfloat16)
        # 确保所有 rank 使用相同的数据
        dist.broadcast(pixel_values_videos, src=0)
        # 视频网格信息：temporal=7, height=28, width=28
        video_grid_thw = torch.tensor([[7, 2, 2]], device='cuda')

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        # 加载 HuggingFace 模型作为参考
        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH, trust_remote_code=True)
        # 使用视频相关的特殊 token
        # video_str = '<0 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|> <1 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>'
        video_str = '<0 seconds><|vision_start|><|video_pad|><|vision_end|> <1 seconds><|vision_start|><|video_pad|><|vision_end|>'
        input_ids = tokenizer(video_str + "这个视频中发生了什么？" * 10, return_tensors="pt").input_ids.to("cuda")
        
        # 模拟视频数据：batch_size=1, channels=3, temporal=4, height=28, width=28
        pixel_values_videos = torch.randn(7*2*2, 3 * 14 * 14, device='cuda', dtype=torch.bfloat16)
        # 确保所有 rank 使用相同的数据
        dist.broadcast(pixel_values_videos, src=0)
        # 视频网格信息：temporal=7, height=28, width=28
        video_grid_thw = torch.tensor([[4, 2, 2], [3, 2, 2]], device='cuda')

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        loss = output.loss

        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))



    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
