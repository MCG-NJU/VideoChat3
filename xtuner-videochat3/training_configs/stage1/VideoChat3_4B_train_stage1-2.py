from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.model import VideoChat3Dense4BConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.config import FSDPConfig
import json

# export XTUNER_TOKENIZE_WORKERS=16
# export XTUNER_USE_FA3=1

# model config
model_cfg = VideoChat3Dense4BConfig(freeze_vision=False, freeze_language=False)

model_path = "work_dir/VideoChat3_4B_train_stage1-1/xxxx/hf-latest"
meta_data_path = 'training_data_annotations/data_stage1-2.json'
work_dir = "work_dir/VideoChat3_4B_train_stage1-2"
cache_dir = "dataset_cache/cache_videochat3_4B_stage1-2"


sample_max_length = 16384
pack_max_length = 16384
global_batch_size = 256
total_epoch = 1
# total_num_tokens = 105332882
# total_step = int(total_num_tokens  / pack_max_length / global_batch_size)
hf_interval = 500
hf_max_keep = 1
checkpoint_interval = 200
checkpoint_maxkeep = 2

lr = 4e-5
weight_decay = 0.0
warmup_ratio = 0.03
lr_min = 1e-6
recompute_ratio = 1.0
loss_reduction = "square"

# oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": "/mnt/shared-storage-user/huanghaian/petreloss.conf"})
oss_loader_cfg=None
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['anno_path'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name='VLMJsonlDataset',
                                          cache_dir=cache_dir),
                 "tokenize_fn": VideoChat3TokenizeFnConfig(
                    model_cfg=model_cfg,
                    max_length=sample_max_length,
                    image_min_pixels=_data.get('image_min_pixels', 28*28),
                    image_max_pixels=_data.get('image_max_pixels', int(sample_max_length * 0.8 * 28 * 28)),
                    frame_min_pixels=_data.get('frame_min_pixels', 28*28),
                    frame_max_pixels=_data.get('frame_max_pixels', int(sample_max_length * 0.8 * 28 * 28)),
                    video_max_total_pixels=_data.get('video_max_total_pixels', int(sample_max_length * 0.8 * 4 * 28 * 28)),
                    video_min_frames=_data.get('video_min_frames', 1),
                    video_max_frames=_data.get('video_max_frames', 2048), 
                    fixed_num_sampled_frames=_data.get('fixed_num_sampled_frames', None),
                    video_sample_fps=_data.get('video_sample_fps', 4), 
                    processor_path=model_path,
                    data_augment=_data.get('data_augment', False),
                    system_message=_data.get('system_message', None),
                    hash=_data.get('hash', None),
                    oss_loader_cfg=oss_loader_cfg
                    )
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    collator="videochat3_sft_collator",
    num_workers=4,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(sp_size=1, recompute_ratio=recompute_ratio, torch_compile=False)

resume_cfg = ResumeConfig(auto_resume=True)

# trainer config
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker='tensorboard',
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
)