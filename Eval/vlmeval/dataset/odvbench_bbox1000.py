# Environment variables used by this dataset:
# - ODVBENCH_ROOT: ODVBench video root directory inherited from ODVBench.
# - ODVBENCH_BBOX1000_ANNO_JSON: path to the BBox1000 annotation JSON file.

# Download the JSON file from https://huggingface.co/datasets/ZZQ987/VideoChat3_Eval_Online_Json
# Place it under VLMEvalkit/json_data/odvbench
# Please ensure the following directory structure:
# 1
# VLMEvalKit/json_data/odvbench/
# ├── ODVbench_bbox1000.json

# 2
# export ODVBENCH_BBOX1000_ANNO_JSON="VLMEvalkit/json_data/odvbench/ODVbench_bbox1000.json"

import os
import os.path as osp
from pathlib import Path

from .odvbench import ODVBench, _default_odvbench_root


def _default_odvbench_bbox1000_anno_json():
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / 'json_data/odvbench/ODVbench_bbox1000.json'
    return os.environ.get('ODVBENCH_BBOX1000_ANNO_JSON', str(default_path)).strip()


class ODVBenchBBox1000(ODVBench):
    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'
    BASE_DATASET_ALIASES = {
        'ODVBench_BBox1000_2fps': 'ODVBench_2fps',
        'ODVBench_BBox1000_4fps': 'ODVBench_4fps',
    }

    @classmethod
    def supported_datasets(cls):
        return ['ODVBench_BBox1000_2fps', 'ODVBench_BBox1000_4fps']

    def __init__(self, dataset='ODVBench_BBox1000_2fps', *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        base_dataset_name = self.BASE_DATASET_ALIASES.get(dataset)
        if base_dataset_name is not None:
            self.frame_root = osp.join(osp.dirname(self.frame_root), base_dataset_name)

    def prepare_dataset(self, dataset='ODVBench_BBox1000_2fps', root_dir=None, ann_json=None):
        root_dir = root_dir if root_dir is not None else _default_odvbench_root()
        ann_json = ann_json if ann_json is not None else _default_odvbench_bbox1000_anno_json()
        tsv_path = osp.join(root_dir, f'{dataset}.tsv')

        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f'ODVBench root directory not found: {root_dir}. '
                'Please set environment variable ODVBENCH_ROOT.'
            )
        if not osp.isfile(ann_json):
            raise FileNotFoundError(
                f'ODVBench BBox1000 annotation json not found: {ann_json}. '
                'Run process_odvbench_bbox1000.py or set ODVBENCH_BBOX1000_ANNO_JSON.'
            )

        if not self._sample_tsv_valid(tsv_path, root_dir):
            self._build_tsv(root_dir, ann_json, tsv_path)

        return dict(root=root_dir, data_file=tsv_path)
