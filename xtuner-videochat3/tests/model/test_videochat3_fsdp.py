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
from xtuner.v1.model.compose.videochat3.videochat3_config import VideoChat3Dense2BConfig
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.utils.test_utils import init_data_mesh
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig

# 设置环境变量路径 - 需要根据实际情况调整
VIDEOCHAT3_DENSE_PATH = os.environ.get("VIDEOCHAT3_DENSE_PATH", "./VideoChat3-2B")



class TestVideoChat3(DeterministicDDPTestCase):
    
    # @parametrize.parametrize(
    #     "device,tol",
    #     [
    #         ("cuda", 1e-2),
    #     ],
    # )
    # def test_fsdp_text_accuracy(self, device, tol):
    #     """测试 FSDP 下的文本推理精度"""
    #     self.create_pg(device)
    #     maybe_compile.clear_compile_targets()
        
    #     # 加载 HuggingFace 模型作为参考
    #     hf_model = AutoModelForCausalLM.from_pretrained(
    #         VIDEOCHAT3_DENSE_PATH,
    #         dtype=torch.bfloat16,
    #         attn_implementation="flash_attention_2",
    #         device_map="cuda",
    #         trust_remote_code=True,
    #     ).eval()
    #     patch_hf_rms_norm(hf_model)

    #     rank = dist.get_rank()
    #     tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH, trust_remote_code=True)
    #     input_ids = tokenizer(f"今天天气不错，是学习的好日子。请听题： 1+{rank} 等于多少？",
    #                           return_tensors="pt").input_ids.to(device)

    #     with torch.no_grad():
    #         output = hf_model(
    #             input_ids=input_ids,
    #             labels=input_ids.clone(),
    #         )
    #     expected_loss = output.loss
    #     dist.all_reduce(expected_loss.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)

    #     del hf_model
    #     torch.cuda.empty_cache()

    #     # 构建 XTuner VideoChat3 模型
    #     with torch.device("meta"):
    #         model_cfg = VideoChat3Dense2BConfig()
    #         videochat3_model = model_cfg.build().to(torch.bfloat16)

    #     fsdp_config = FSDPConfig(
    #         cpu_offload=False,
    #     )

    #     # 对各个组件进行 FSDP 切分
    #     videochat3_model.language_model.fully_shard(fsdp_config=fsdp_config)
    #     videochat3_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
    #     videochat3_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
    #     videochat3_model.fully_shard(fsdp_config=fsdp_config)

    #     videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
    #     videochat3_model.eval()

    #     shift_input_ids = input_ids[:, :-1]
    #     shifted_labels = input_ids[:, 1:]
    #     seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
    #     loss_ctx_input = CELossContextInputItem(shifted_labels=shifted_labels)
    #     loss_ctx_input = loss_ctx_input.to('cuda')

    #     seq_ctx_list = [seq_ctx]
    #     loss_ctx_input_list: list[CELossContextInputItem] = [loss_ctx_input]

    #     loss_cfg = CELossConfig()
    #     LossContext = loss_cfg.loss_ctx_cls
    #     batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
    #         loss_ctx_input_list,
    #         loss_cfg,
    #     )
    #     loss_kwargs = batches_loss_kwargs[0]
    #     loss_ctx = LossContext(loss_cfg, loss_kwargs)
    #     seq_ctx = seq_ctx_list[0]

    #     with torch.no_grad():
    #         output = videochat3_model(
    #             seq_ctx=seq_ctx,
    #             loss_ctx=loss_ctx,
    #         )
    #     loss = output["loss"]
    #     self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device, sp_size, compile, tol",
        [
            ("cuda", 1, False, 1e-2),
            # ("cuda", 2, False, 1e-2),
            ("cuda", 1, True, 1e-2),
            # ("cuda", 2, True, 1e-2),
        ],
    )
    def test_fsdp_image_accuracy(self, device, sp_size, compile, tol):
        self.create_pg(device)
        if not compile:
            maybe_compile.clear_compile_targets()

        hf_model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        patch_hf_rms_norm(hf_model)

        tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH, trust_remote_code=True)
        tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_DENSE_PATH, add_vision_id=True).build(tokenizer)

        raw_data = {"id": 3, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {
            "url": "tests/resource/mscoco_twocat_000000039769.jpg", "image_wh": [640, 480]}}, {"type": "image_url",
                                                                                               "image_url": {
                                                                                                   "url": "tests/resource/mscoco_dog_000000319154.jpg",
                                                                                                   "image_wh": [375,
                                                                                                                500]}},
                                                                       {"type": "text",
                                                                        "text": "<IMG_CONTEXT>\n<IMG_CONTEXT>\n请描述下第二幅图片中的狗是什么颜色？"}]},
                                          {"role": "assistant", "content": "图片中的狗是棕色的。"}]}

        tokenized_data = tokenize_fn(raw_data)
        input_ids = torch.tensor(tokenized_data['input_ids'])[None].cuda()
        labels = torch.tensor(tokenized_data['labels'])[None].cuda()
        pixel_values = tokenized_data['pixel_values'].cuda()
        image_grid_thw = tokenized_data['image_grid_thw'].cuda()

        # rank = dist.get_rank()
        # image_str = '<|vision_start|><|image_pad|><|vision_end|>'
        # input_ids = tokenizer(image_str + "吃葡萄不吐葡萄皮" * 20, return_tensors="pt").input_ids.to("cuda")
        # pixel_values = torch.randn(4, 588, device='cuda', dtype=torch.bfloat16) # (h*w, 14 * 14 * 3 * 1)
        # # TODO: 不合理，为啥一定要每个 rank 数据完全一样才能通过 CI ?
        # dist.broadcast(pixel_values, src=0)

        # image_grid_thw = torch.tensor([[1, 2, 2]], device='cuda')

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            model_cfg = VideoChat3Dense2BConfig()
            videochat3_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            cpu_offload=False,
        )
        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]

        videochat3_model.language_model.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.fully_shard(fsdp_config=fsdp_config)

        videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
        videochat3_model.eval()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = labels[:, 1:]
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

        loss_cfg = CELossConfig()
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


    # @parametrize.parametrize(
    #     "device, sp_size, compile, tol",
    #     [
    #         ("cuda", 1, False, 1e-2),
    #         # ("cuda", 2, False, 1e-2),
    #         ("cuda", 1, True, 1e-2),
    #         # ("cuda", 2, True, 1e-2),
    #     ],
    # )
    # def test_fsdp_multi_image_accuracy(self, device, sp_size, compile, tol):
    #     self.create_pg(device)
    #     if not compile:
    #         maybe_compile.clear_compile_targets()

    #     hf_model = AutoModelForCausalLM.from_pretrained(
    #         VIDEOCHAT3_DENSE_PATH,
    #         dtype=torch.bfloat16,
    #         attn_implementation="flash_attention_2",
    #         device_map="cuda",
    #         trust_remote_code=True,
    #     ).eval()
    #     # patch_hf_rms_norm(hf_model)

    #     rank = dist.get_rank()
    #     tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_DENSE_PATH)
    #     image_str = '看这张图 <|vision_start|><|image_pad|><|vision_end|> 和 这张图<|vision_start|><|image_pad|><|vision_end|>'
    #     input_ids = tokenizer(image_str + "它们有什么区别" * 20, return_tensors="pt").input_ids.to("cuda")
    #     pixel_values = torch.randn(4 + 12, 588, device='cuda', dtype=torch.bfloat16)
    #     # TODO: 不合理，为啥一定要每个 rank 数据完全一样才能通过 CI ?
    #     dist.broadcast(pixel_values, src=0)
    #     image_grid_thw = torch.tensor([[1, 2, 2], [1, 6, 2]], device='cuda')

    #     with torch.no_grad():
    #         output = hf_model(
    #             input_ids=input_ids,
    #             labels=input_ids.clone(),
    #             pixel_values=pixel_values,
    #             image_grid_thw=image_grid_thw,
    #         )
    #     expected_loss = output.loss

    #     del hf_model
    #     torch.cuda.empty_cache()

    #     with torch.device("meta"):
    #         model_cfg = VideoChat3Dense2BConfig()
    #         videochat3_model = model_cfg.build().to(torch.bfloat16)

    #     fsdp_config = FSDPConfig(
    #         cpu_offload=False,
    #     )
    #     sp_mesh = None
    #     if sp_size > 1:
    #         data_mesh = init_data_mesh(device, sp_size=sp_size)
    #         sp_mesh = data_mesh["sp"]

    #     videochat3_model.language_model.fully_shard(fsdp_config=fsdp_config)
    #     videochat3_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
    #     videochat3_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
    #     videochat3_model.fully_shard(fsdp_config=fsdp_config)

    #     videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
    #     videochat3_model.eval()

    #     shift_input_ids = input_ids[:, :-1]
    #     shifted_labels = input_ids[:, 1:]
    #     seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
    #     seq_ctx.image_grid_thw = image_grid_thw
    #     seq_ctx.pixel_values = pixel_values
    #     seq_ctx.to('cuda')
    #     loss_ctx_input = CELossContextInputItem(shifted_labels=shifted_labels)
    #     loss_ctx_input = loss_ctx_input.to('cuda')

    #     if sp_size > 1:
    #         seq_ctx = seq_ctx.split(sp_mesh)
    #         loss_ctx_input = loss_ctx_input.sp_split(sp_mesh)

    #     seq_ctx_list = [seq_ctx]
    #     loss_ctx_input_list: list[CELossContextInputItem] = [loss_ctx_input]

    #     loss_cfg = CELossConfig()
    #     LossContext = loss_cfg.loss_ctx_cls
    #     batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
    #         loss_ctx_input_list,
    #         loss_cfg,
    #     )
    #     loss_kwargs = batches_loss_kwargs[0]
    #     loss_ctx = LossContext(loss_cfg, loss_kwargs)
    #     seq_ctx = seq_ctx_list[0]

    #     with torch.no_grad():
    #         output = videochat3_model(
    #             seq_ctx=seq_ctx,
    #             loss_ctx=loss_ctx,
    #         )
    #     loss = output["loss"]
    #     self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device, sp_size, compile, tol",
        [
            ("cuda", 1, False, 1e-2),
            ("cuda", 1, True, 1e-2),
        ],
    )
    def test_fsdp_video_accuracy(self, device, sp_size, compile, tol):
        """测试 FSDP 下的视频推理精度"""
        self.create_pg(device)
        if not compile:
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
        video_str = '<0 seconds><|vision_start|><|video_pad|><|vision_end|> <1 seconds><|vision_start|><|video_pad|><|vision_end|>'
        input_ids = tokenizer(video_str + "这个视频中发生了什么？" * 10, return_tensors="pt").input_ids.to("cuda")
        
        # 模拟视频数据
        pixel_values_videos = torch.randn(7*2*2, 3 * 14 * 14, device='cuda', dtype=torch.bfloat16)
        dist.broadcast(pixel_values_videos, src=0)
        video_grid_thw = torch.tensor([[4, 2, 2], [3, 2, 2]], device='cuda')

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
            model_cfg = VideoChat3Dense2BConfig()
            videochat3_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            cpu_offload=False,
        )
        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]

        # 对各个组件进行 FSDP 切分
        videochat3_model.language_model.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.fully_shard(fsdp_config=fsdp_config)

        videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
        videochat3_model.eval()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
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

        loss_cfg = CELossConfig()
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
        "device,tp_size",
        [
            ("cuda", 1),
        ],
    )
    def test_save_hf(self, device, tp_size):
        """测试保存为 HuggingFace 格式"""
        self.create_pg(device)
        with torch.device("meta"):
            model_cfg = VideoChat3Dense2BConfig()
            videochat3_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            tp_size=tp_size,
            cpu_offload=False,
        )

        cache_save_fh = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
        tmpdir = Path(syncdir[0])
        
        # 对各个组件进行 FSDP 切分
        videochat3_model.language_model.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        videochat3_model.fully_shard(fsdp_config=fsdp_config)
        
        videochat3_model.from_hf(VIDEOCHAT3_DENSE_PATH)
        videochat3_model.save_hf(tmpdir)

        origin_hf_path = Path(VIDEOCHAT3_DENSE_PATH)
        origin_index_path = origin_hf_path / "model.safetensors.index.json"
        saved_index_path = tmpdir / "model.safetensors.index.json"

        # 测试保存的 HuggingFace 张量值是否与原始值匹配
        if dist.get_rank() == 0:
            with open(origin_index_path, "r") as f:
                origin_index = json.load(f)
            with open(saved_index_path, "r") as f:
                saved_index = json.load(f)

            for key in origin_index["weight_map"].keys():
                origin_safetensor_name = origin_index["weight_map"][key]
                saved_safetensor_name = saved_index["weight_map"][key]

                origin_sf_fh_name = str(origin_hf_path / origin_safetensor_name)
                expected_sf_fh_name = str(tmpdir / saved_safetensor_name)

                if origin_safetensor_name not in cache_save_fh:
                    cache_save_fh[origin_safetensor_name] = safe_open(origin_sf_fh_name, framework="pt")
                if saved_safetensor_name not in cache_save_fh:
                    cache_save_fh[saved_safetensor_name] = safe_open(expected_sf_fh_name, framework="pt")

                origin_fh = cache_save_fh[origin_safetensor_name]
                saved_fh = cache_save_fh[saved_safetensor_name]

                origin_tensor = origin_fh.get_tensor(key)
                saved_tensor = saved_fh.get_tensor(key)
                self.assertTrue(torch.equal(origin_tensor.to(saved_tensor.dtype), saved_tensor))

            # 测试 safetensors 中的张量数量是否与模型索引中的张量数量匹配
            safetensor_keys = []
            for safetensor_path in tmpdir.glob("*.safetensors"):
                fh = cache_save_fh[safetensor_path.name]
                safetensor_keys.extend(fh.keys())
                safetensor_keys.sort()
            model_index_keys = list(saved_index["weight_map"].keys())
            model_index_keys.sort()

            self.assertListEqual(safetensor_keys, model_index_keys)
        dist.barrier()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
