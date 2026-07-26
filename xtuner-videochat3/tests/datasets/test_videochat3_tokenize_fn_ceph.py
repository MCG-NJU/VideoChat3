import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from transformers import AutoTokenizer, AutoProcessor
import json
import parametrize

LOCAL_MEDIA_ROOT = "tests/resource"
CEPH_ROOT = "pnorm2:s3://videochat3/xtuner_example_data"
VIDEOCHAT3_PATH = os.environ.get("VIDEOCHAT3_PATH", "VideoChat3-2B")


class TestVideoChat3TokenizeFn(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)
        self.tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH).build(self.tokenizer)
        self.processor = AutoProcessor.from_pretrained(VIDEOCHAT3_PATH, trust_remote_code=True)

    def test_videochat3_single_image(self):
        data_path = 'tests/resource/mllm_sft_single_image_example_data.jsonl'
        total_step = 5
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)
                
                ret = self.tokenize_fn(raw_data, media_root=LOCAL_MEDIA_ROOT)
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                ret_ceph = self.tokenize_fn(raw_data, media_root=CEPH_ROOT)
                input_ids_xtuner_ceph = ret_ceph['input_ids']
                pixel_values_xtuner_ceph: torch.Tensor = ret_ceph['pixel_values']
                image_grid_thw_xtuner_ceph: torch.Tensor = ret_ceph['image_grid_thw']
                
                self.assertIsInstance(input_ids_xtuner, list)
                self.assertIsInstance(input_ids_xtuner_ceph, list)
                self.assertGreater(len(input_ids_xtuner), 0)
                self.assertGreater(len(input_ids_xtuner_ceph), 0)
                # 检查pixel_values和image_grid_thw的形状
                self.assertEqual(pixel_values_xtuner.shape, pixel_values_xtuner_ceph.shape)
                self.assertEqual(image_grid_thw_xtuner.shape, image_grid_thw_xtuner_ceph.shape)
                self.assertEqual(input_ids_xtuner, input_ids_xtuner_ceph)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_xtuner_ceph))
                self.assertTrue(torch.equal(image_grid_thw_xtuner, image_grid_thw_xtuner_ceph))

                # 检查calc_num_tokens_get_item和get_item出来的token数是否一致
                # 需要加载原始数据备份，因为raw_data被修改了
                raw_data_copy = json.loads(line)
                self.tokenize_fn.state = "cache"
                cache_result = self.tokenize_fn(raw_data_copy, media_root=LOCAL_MEDIA_ROOT)
                self.tokenize_fn.state = "get_item"
                self.assertEqual(len(input_ids_xtuner), cache_result['num_tokens'], 
                                "calc_num_tokens_get_item和get_item出来的token数不一致")
                self.assertEqual(len(input_ids_xtuner_ceph), cache_result['num_tokens'], 
                                "calc_num_tokens_get_item和get_item出来的token数不一致")

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_videochat3_video(self, add_vision_id):
        tokenize_fn = VideoChat3TokenizeFnConfig(processor_path=VIDEOCHAT3_PATH,
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_sft_video_example_data.jsonl'
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1:
                    break
                raw_data = json.loads(line)

                # 测试collect_image_video_paths_and_extra函数
                from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import collect_image_video_paths_and_extra
                image_paths, video_paths, extra_info = collect_image_video_paths_and_extra(raw_data['messages'])
                
                # 验证路径和元数据提取正确
                self.assertEqual(len(video_paths), 1)
                self.assertEqual(video_paths[0], 'tennis.mp4')
                self.assertEqual(len(extra_info['video_meta_list']), 1)
                self.assertIsNotNone(extra_info['video_meta_list'][0])
                
                # 验证消息结构
                self.assertIn('messages', raw_data)
                self.assertIsInstance(raw_data['messages'], list)
                self.assertGreater(len(raw_data['messages']), 0)
                # 对xtuner的结果
                ret_xtuner = tokenize_fn(raw_data, media_root=LOCAL_MEDIA_ROOT)
                input_ids_xtuner = ret_xtuner['input_ids']
                pixel_values_xtuner: torch.Tensor = ret_xtuner['pixel_values']
                video_grid_thw_xtuner: torch.Tensor = ret_xtuner['image_grid_thw']

                ret_xtuner_ceph = tokenize_fn(raw_data, media_root=CEPH_ROOT)
                input_ids_xtuner_ceph = ret_xtuner_ceph['input_ids']
                pixel_values_xtuner_ceph: torch.Tensor = ret_xtuner_ceph['pixel_values']
                video_grid_thw_xtuner_ceph: torch.Tensor = ret_xtuner_ceph['image_grid_thw']
                

                self.assertEqual(input_ids_xtuner, input_ids_xtuner_ceph)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_xtuner_ceph))
                self.assertTrue(torch.equal(video_grid_thw_xtuner, video_grid_thw_xtuner_ceph))
                
                # 检查calc_num_tokens_get_item和get_item出来的token数是否一致
                # 需要重新加载原始数据，因为raw_data被修改了
                # cache状态需要保持<VIDEO_CONTEXT>占位符，因为calc_num_tokens_video_get_item需要它们
                raw_data_copy = json.loads(line)
                tokenize_fn.state = "cache"
                cache_result = tokenize_fn(raw_data_copy, media_root=LOCAL_MEDIA_ROOT)
                tokenize_fn.state = "get_item"
                self.assertEqual(len(input_ids_xtuner), cache_result['num_tokens'], 
                                "calc_num_tokens_get_item和get_item出来的token数不一致")
                self.assertEqual(len(input_ids_xtuner_ceph), cache_result['num_tokens'], 
                                "calc_num_tokens_get_item和get_item出来的token数不一致")

    