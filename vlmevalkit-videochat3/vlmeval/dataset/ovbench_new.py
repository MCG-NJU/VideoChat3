# Environment variables used by this dataset:
# - OVBENCH_ROOT and OVBENCH_ANNO_JSON: base OVBench media and split annotation.
# - OVBENCH_NEW_AVA_ROOT: directory containing ovbench_ava_raw.json and AVA media.
# - OVBENCH_NEW_BASE_ANNO_JSON (optional) / OVBENCH_NEW_BBOX1000_BASE_ANNO_JSON (optional): New split overrides.
# - OVBENCH_NEW_AVA_ANNO_JSON (optional) / OVBENCH_NEW_AVA_BBOX1000_ANNO_JSON (optional): AVA annotation overrides.

# Download the JSON file from https://huggingface.co/datasets/ZZQ987/VideoChat3_Eval_Online_Json
# Place it under VLMEvalkit/json_data/ovbench

# Please ensure the following directory structure:
# 1
# VLMEvalKit/json_data/ovbench/
# ├── ovbench_split_with_fps_ava_raw_frame_processed_128.json
# └── ovbench_split_with_fps_ava_raw_frame_processed_128_bbox1000.json
# 2
# VLMEvalkit/json_data/ovbench/ava/
# ├── ovbench_ava_raw.json
# └── ovbench_ava_raw_bbox1000.json
# 3
# export OVBENCH_ANNO_JSON="VLMEvalkit/json_data/ovbench/ovbench_split_with_fps_ava_raw_frame_processed_128.json"
# export OVBENCH_NEW_AVA_ROOT="VLMEvalkit/json_data/ovbench/ava"

import json
import math
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

from ..smp import *
from .ovbench import (
    OVBench,
    _cache_safe_token,
    _default_ovbench_root,
    _has_value,
    _infer_subset,
    _load_json_records,
    _parse_question_idx_from_sample_id,
    _resolve_split_annotation_json,
    _strip_option_prefix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_ANNO = PROJECT_ROOT / 'json_data/ovbench/ovbench_split_with_fps_ava_raw_frame_processed_128.json'
DEFAULT_BASE_BBOX1000_ANNO = (
    PROJECT_ROOT / 'json_data/ovbench/ovbench_split_with_fps_ava_raw_frame_processed_128_bbox1000.json'
)
DEFAULT_AVA_ROOT = os.environ.get('OVBENCH_NEW_AVA_ROOT', '/mnt/petrelfs/zengxiangyu/Research_yyd/A_Datasets/online_eval/OVBench')
DEFAULT_AVA_ANNO = osp.join(DEFAULT_AVA_ROOT, 'ovbench_ava_raw.json')
DEFAULT_AVA_BBOX1000_ANNO = osp.join(DEFAULT_AVA_ROOT, 'ovbench_ava_raw_bbox1000.json')

SYSTEM_PROMPT = (
    'Carefully watch the sampled video frames and pay attention to the cause and '
    'sequence of events, the detail and movement of objects, and the action and '
    'pose of persons. Based on your observations, select the best option that '
    'accurately addresses the question. Answer with only the option letter.'
)


def _env_float(name, default):
    value = os.environ.get(name, '').strip()
    return float(value) if value else float(default)


def _env_int(name, default):
    value = os.environ.get(name, '').strip()
    return int(value) if value else int(default)


def _is_flat_ava_record(record):
    video_id = str(record.get('video_id', '')).replace('\\', '/').strip()
    return video_id.startswith('AVA')


def _resolve_existing_json(candidates):
    for candidate in candidates:
        if candidate and osp.isfile(str(candidate)):
            return str(candidate)
    return str(candidates[0])


def _uniform_sample(values, max_items):
    if max_items <= 0 or len(values) <= max_items:
        return values
    if max_items == 1:
        return [values[-1]]
    positions = [int(round(i * (len(values) - 1) / (max_items - 1))) for i in range(max_items)]
    return [values[pos] for pos in positions]


def _sample_times(end_time, window_seconds, fps, max_frames):
    start_time = max(0.0, float(end_time) - float(window_seconds))
    step = 1.0 / max(float(fps), 1e-6)
    count = max(1, int(math.floor((float(end_time) - start_time) / step)) + 1)
    times = [start_time + i * step for i in range(count)]
    if not times or times[-1] < float(end_time):
        times.append(float(end_time))
    times = [min(float(end_time), t) for t in times]
    return _uniform_sample(times, max_frames)


def _ava_window_start(timestamp, window_seconds, min_start):
    timestamp = float(timestamp)
    min_start = float(min_start)
    if timestamp >= min_start:
        return max(timestamp - float(window_seconds), min_start)
    return max(timestamp - float(window_seconds), 0.0)


def _sample_times_from_window(start_time, end_time, fps, max_frames):
    start_time = max(0.0, float(start_time))
    end_time = max(start_time, float(end_time))
    step = 1.0 / max(float(fps), 1e-6)
    count = max(1, int(math.floor((end_time - start_time) / step)) + 1)
    times = [start_time + i * step for i in range(count)]
    if not times or times[-1] < end_time:
        times.append(end_time)
    times = [min(end_time, t) for t in times]
    return _uniform_sample(times, max_frames)


class OVBenchNew(OVBench):
    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    DATASET_ALIASES = {
        'OVBench_New': 'OVBench',
        'OVBench_New_1fps': 'OVBench_1fps',
        'OVBench_New_2fps': 'OVBench_2fps',
        'OVBench_New_4fps': 'OVBench_4fps',
    }

    def __init__(
        self,
        dataset='OVBench_New',
        nframe=0,
        fps=1.0,
        frames_limit=4096,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=65536 * 4 * 16 * 16,
        ava_raw_fps=None,
        ava_raw_max_frames=None,
        ava_raw_window_seconds=None,
        ava_root=None,
        ava_anno_json=None,
        check_extracted_frames=True,
    ):
        self.ava_raw_fps = float(ava_raw_fps) if ava_raw_fps is not None else _env_float('OVBENCH_NEW_AVA_FPS', 1.0)
        self.ava_raw_max_frames = (
            int(ava_raw_max_frames)
            if ava_raw_max_frames is not None
            else _env_int('OVBENCH_NEW_AVA_MAX_FRAMES', 16)
        )
        self.ava_raw_window_seconds = (
            float(ava_raw_window_seconds)
            if ava_raw_window_seconds is not None
            else _env_float('OVBENCH_NEW_AVA_WINDOW_SECONDS', 120.0)
        )
        self.ava_raw_min_start = _env_float('OVBENCH_NEW_AVA_MIN_START', 900.0)
        self.ava_root = ava_root or os.environ.get('OVBENCH_NEW_AVA_ROOT', DEFAULT_AVA_ROOT).strip()
        self.ava_anno_json = ava_anno_json or os.environ.get('OVBENCH_NEW_AVA_ANNO_JSON', DEFAULT_AVA_ANNO).strip()

        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        
        super().__init__(
            dataset=dataset,
            nframe=nframe,
            fps=fps,
            check_extracted_frames=check_extracted_frames,
            # frames_limit=frames_limit,
            # min_pixels=min_pixels,
            # max_pixels=max_pixels,
            # total_pixels=total_pixels,
        )

    @classmethod
    def supported_datasets(cls):
        return ['OVBench_New', 'OVBench_New_1fps', 'OVBench_New_2fps', 'OVBench_New_4fps']

    @classmethod
    def _default_base_ann_json(cls):
        env_path = os.environ.get('OVBENCH_NEW_BASE_ANNO_JSON', '').strip()
        if env_path:
            return env_path
        if osp.isfile(str(DEFAULT_BASE_ANNO)):
            return str(DEFAULT_BASE_ANNO)
        return _resolve_split_annotation_json(dataset='OVBench_2fps')

    @classmethod
    def _build_rows_from_records(cls, records):
        rows = []
        per_video_question_counter = defaultdict(int)
        for record in records:
            video_rel_path = str(record.get('video_id', '')).strip()
            if not video_rel_path:
                raise ValueError('OVBench_New split record is missing `video_id`')
            subset = _infer_subset(video_rel_path)
            if not subset:
                raise ValueError(f'OVBench_New split record has invalid `video_id`: {video_rel_path!r}')

            sample_id = str(record.get('id', '')).strip()
            question_idx = _parse_question_idx_from_sample_id(sample_id)
            if question_idx is None:
                question_idx = per_video_question_counter[video_rel_path]
            per_video_question_counter[video_rel_path] = max(
                per_video_question_counter[video_rel_path],
                question_idx + 1,
            )

            options = record.get('options', [])
            if not isinstance(options, list):
                raise ValueError(f'OVBench_New record {sample_id or video_rel_path} has invalid options')
            if len(options) < 2 or len(options) > 5:
                raise ValueError(f'OVBench_New record {sample_id or video_rel_path} has {len(options)} options')

            option_map = {ch: '' for ch in 'ABCDE'}
            stripped_options = [_strip_option_prefix(x) for x in options]
            for ch, text in zip('ABCDE', stripped_options):
                option_map[ch] = text

            answer_letter = str(record.get('answer', '')).strip().upper()
            if answer_letter not in option_map or not option_map[answer_letter]:
                raise ValueError(
                    f'OVBench_New record {sample_id or video_rel_path} has invalid answer letter: {answer_letter!r}'
                )

            question_start = float(record.get('start'))
            question_end = float(record.get('end'))
            video_time = float(record.get('video_time'))
            if not (question_start <= question_end <= video_time):
                raise ValueError(
                    f'OVBench_New record {sample_id or video_rel_path} has invalid time range: '
                    f'start={question_start}, end={question_end}, video_time={video_time}'
                )

            row = dict(
                index=len(rows),
                sample_id=sample_id or f'{video_rel_path}__q{question_idx}',
                subset=subset,
                question_idx=question_idx,
                question=str(record.get('question', '')),
                video=video_rel_path,
                video_id=video_rel_path,
                answer=answer_letter,
                answer_text=option_map[answer_letter],
                answer_type=str(record.get('answer_type', '')),
                sub_answer_type=str(record.get('sub_answer_type', '')),
                A=option_map['A'],
                B=option_map['B'],
                C=option_map['C'],
                D=option_map['D'],
                E=option_map['E'],
                start=question_start,
                end=question_end,
                video_time=video_time,
                fps=record.get('fps'),
                middle_frame_timestamp=record.get('middle_frame_timestamp'),
            )
            rows.append(row)
        return rows

    @classmethod
    def _flatten_ava_records(cls, ava_json, window_seconds):
        with open(ava_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = []
        for video_item in data:
            video_id = str(video_item.get('video_id', '')).strip()
            if not video_id:
                raise ValueError(f'OVBench_New AVA item missing video_id in {ava_json}')
            for question_idx, question in enumerate(video_item.get('questions', [])):
                timestamp = float(question['middle_frame_timestamp'])
                records.append(
                    dict(
                        id=f'{video_id}__q{question_idx}',
                        video_id=video_id,
                        question=question.get('question', ''),
                        answer=question.get('answer', ''),
                        answer_type=question.get('answer_type', ''),
                        sub_answer_type=question.get('sub_answer_type', ''),
                        options=question.get('options', []),
                        start=_ava_window_start(
                            timestamp,
                            window_seconds,
                            _env_float('OVBENCH_NEW_AVA_MIN_START', 900.0),
                        ),
                        end=timestamp,
                        video_time=max(timestamp, 0.0),
                        middle_frame_timestamp=timestamp,
                    )
                )
        return records

    @classmethod
    def _build_mixed_tsv(cls, base_ann_json, ava_ann_json, tsv_path, window_seconds):
        base_records = _load_json_records(base_ann_json)
        non_ava_records = [record for record in base_records if not _is_flat_ava_record(record)]
        ava_records = cls._flatten_ava_records(ava_ann_json, window_seconds=window_seconds)
        rows = cls._build_rows_from_records(non_ava_records + ava_records)
        dump(pd.DataFrame(rows), tsv_path)

    def _new_tsv_valid(self, tsv_path):
        if not osp.exists(tsv_path):
            return False
        try:
            data = load(tsv_path)
        except Exception:
            return False
        if 'video' not in data or 'start' not in data or 'end' not in data:
            return False
        ava = data[data['video'].astype(str).str.startswith('AVA_RAW/')]
        if len(ava) == 0:
            return False
        checked = ava[ava['end'].astype(float) >= float(self.ava_raw_min_start)]
        if len(checked) == 0:
            return True
        return bool((checked['start'].astype(float) >= float(self.ava_raw_min_start) - 1e-6).all())

    def prepare_dataset(self, dataset='OVBench_New', root_dir=None, ann_json=None, ann_dir=None):
        root_dir = root_dir if root_dir is not None else _default_ovbench_root()
        tsv_path = osp.join(root_dir, f'{dataset}.tsv')
        base_ann_json = ann_json if ann_json is not None else self._default_base_ann_json()

        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f'OVBench_New root directory not found: {root_dir}. Please set environment variable OVBENCH_ROOT.'
            )
        if not osp.isfile(base_ann_json):
            raise FileNotFoundError(f'OVBench_New base annotation json not found: {base_ann_json}')
        if not osp.isfile(self.ava_anno_json):
            raise FileNotFoundError(f'OVBench_New AVA_RAW annotation json not found: {self.ava_anno_json}')

        force_rebuild = os.environ.get('OVBENCH_NEW_FORCE_REBUILD', '').strip() == '1'
        if (
            force_rebuild
            or not self._sample_tsv_valid(tsv_path, require_directory_fps=False)
            or not self._new_tsv_valid(tsv_path)
        ):
            self._build_mixed_tsv(
                base_ann_json=base_ann_json,
                ava_ann_json=self.ava_anno_json,
                tsv_path=tsv_path,
                window_seconds=self.ava_raw_window_seconds,
            )

        return dict(root=root_dir, data_file=tsv_path)

    def _base_dataset_name(self):
        return self.DATASET_ALIASES.get(self.dataset_name, self.dataset_name)

    def _is_ava_raw_line(self, line):
        video_id = str(line.get('video_id', line.get('video', ''))).replace('\\', '/').strip()
        return video_id.startswith('AVA_RAW/')

    def _resolve_ava_raw_path(self, line):
        video_rel_path = str(line.get('video', line.get('video_id', ''))).replace('\\', '/').strip().lstrip('/')
        path = osp.join(self.ava_root, video_rel_path)
        if osp.exists(path):
            return path
        stem, ext = osp.splitext(path)
        if ext:
            return path
        for suffix in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
            candidate = stem + suffix
            if osp.exists(candidate):
                return candidate
        return path

    @staticmethod
    def _read_frame(cap, frame_index):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(0, min(int(frame_index), max(total_frames - 1, 0)))
        for offset in (0, -1, 1, -2, 2, -5):
            candidate = max(0, min(frame_index + offset, max(total_frames - 1, 0)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
            ok, frame = cap.read()
            if ok and frame is not None:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None

    def _sample_ava_raw_frames(self, video_path, start_time, timestamp):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open OVBench_New AVA_RAW video: {video_path}')
        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if source_fps <= 0:
                raise RuntimeError(f'Cannot read FPS from OVBench_New AVA_RAW video: {video_path}')
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / source_fps if total_frames > 0 else float(timestamp)
            safe_timestamp = min(max(float(timestamp), 0.0), max(duration, 0.0))
            safe_start = min(max(float(start_time), 0.0), safe_timestamp)
            times = _sample_times_from_window(
                start_time=safe_start,
                end_time=safe_timestamp,
                fps=self.ava_raw_fps,
                max_frames=self.ava_raw_max_frames,
            )
            frames, kept_times, indices = [], [], []
            for t in times:
                frame_index = int(round(t * source_fps))
                frame = self._read_frame(cap, frame_index)
                if frame is None:
                    continue
                frames.append(frame)
                kept_times.append(float(t))
                indices.append(frame_index)
            if not frames:
                raise RuntimeError(f'No frames decoded from OVBench_New AVA_RAW video: {video_path}')
            return frames, kept_times, indices, source_fps, duration
        finally:
            cap.release()

    def _save_ava_raw_frames(self, line):
        video_path = self._resolve_ava_raw_path(line)
        timestamp = float(line.get('middle_frame_timestamp')) if _has_value(line.get('middle_frame_timestamp')) else float(line['end'])
        mode = f'avaRaw{self.ava_raw_fps:g}fpsMax{self.ava_raw_max_frames}'
        cache_key = self._cache_key(
            line.get('subset', 'AVA_RAW'),
            line['video'],
            line.get('question_idx'),
            line.get('sample_id'),
            line.get('start'),
            line.get('end'),
            mode,
            'file',
            extra_token=f'win{self.ava_raw_window_seconds:g}-min{self.ava_raw_min_start:g}',
        )
        metadata_path = self._frame_metadata_path(cache_key)
        cached_info = self._load_cached_video_info(metadata_path)
        if cached_info is not None:
            frame_paths = self._cached_frame_paths_from_info(metadata_path, cached_info)
            if frame_paths is not None:
                return frame_paths, cached_info

        frames, sample_times_sec, frame_indices, source_fps, duration = self._sample_ava_raw_frames(
            video_path,
            start_time=float(line['start']),
            timestamp=timestamp,
        )
        frame_root = osp.dirname(metadata_path)
        frame_paths = [
            osp.join(frame_root, f'frame-{idx:04d}-of-{len(frames):04d}-ava-raw.jpg')
            for idx in range(1, len(frames) + 1)
        ]
        for frame, frame_path in zip(frames, frame_paths):
            if not osp.exists(frame_path):
                frame.save(frame_path)

        output_frames = []
        for position, (frame_path, frame_idx, time_sec) in enumerate(zip(frame_paths, frame_indices, sample_times_sec), start=1):
            output_frames.append(
                dict(
                    position=int(position),
                    total_frames=int(len(frame_paths)),
                    time_sec=float(time_sec),
                    orig_index=int(frame_idx),
                    output_name=osp.basename(frame_path),
                )
            )
        span = max(0.0, float(sample_times_sec[-1]) - float(sample_times_sec[0])) if len(sample_times_sec) > 1 else 0.0
        sample_fps = ((len(sample_times_sec) - 1) / span) if span > 0 and len(sample_times_sec) > 1 else self.ava_raw_fps
        video_info = {
            'metadata_version': 'ovbench_new_ava_raw_v1',
            'dataset_name': str(self.dataset_name),
            'sample_id': str(line.get('sample_id', '')).strip(),
            'video_id': str(line.get('video_id', line.get('video', ''))).strip(),
            'media_path': str(video_path),
            'source_kind': 'file',
            'video_read_type': 'opencv',
            'fps': float(source_fps),
            'source_fps': float(source_fps),
            'n_frames': None,
            'duration': float(duration),
            'clip_start': int(min(frame_indices)),
            'clip_end': int(max(frame_indices) + 1),
            'effective_start_sec': float(line['start']),
            'effective_end_sec': float(line['end']),
            'clip_duration': float(float(line['end']) - float(line['start'])),
            'clip_frames': int(max(frame_indices) - min(frame_indices) + 1),
            'sample_fps': float(sample_fps),
            'sample_n_frame': len(frame_paths),
            'sample_times_sec': sample_times_sec,
            'frame_indices': frame_indices,
            'max_frames_num': int(self.ava_raw_max_frames),
            'dynamic_fps': float(self.ava_raw_fps),
            'output_frames': output_frames,
        }
        self._dump_video_info(metadata_path, video_info)
        return frame_paths, video_info

    def save_video_frames(self, line, video_llm=False, verbose=False):
        if isinstance(line, int):
            line = self.data.iloc[line]
        if self._is_ava_raw_line(line):
            return self._save_ava_raw_frames(line)
        original_dataset_name = self.dataset_name
        self.dataset_name = self._base_dataset_name()
        try:
            return super().save_video_frames(line, video_llm=video_llm, verbose=verbose)
        finally:
            self.dataset_name = original_dataset_name

    def _build_ava_raw_question_text(self, line):
        timestamp = float(line.get('middle_frame_timestamp')) if _has_value(line.get('middle_frame_timestamp')) else float(line['end'])
        segments = [f"Question at {timestamp:.1f}s: {line['question']}", 'Options:']
        for ch in 'ABCDE':
            value = line.get(ch, '')
            if _has_value(value) and str(value).strip():
                segments.append(f'{ch}. {value}')
        segments.append('Only give the best option.')
        return '\n'.join(segments)

    def _build_ava_raw_prompt(self, line):
        frame_paths, video_info = self.save_video_frames(line)
        sample_times_sec = video_info.get('sample_times_sec') or []
        message = [dict(type='text', value=SYSTEM_PROMPT, role='system')]
        for idx, frame_path in enumerate(frame_paths):
            message.append(dict(type='text', value=f'Frame{idx + 1}:'))
            message.append(dict(type='image', value=frame_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels))
        sec_text = ', '.join(f'{float(sec):.1f}' for sec in sample_times_sec)
        subtitle = f'The video contains {len(frame_paths)} frames sampled at {sec_text} seconds.\n'
        message.append(dict(type='text', value=subtitle + self._build_ava_raw_question_text(line)))
        return message

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]
        if self._is_ava_raw_line(line):
            return self._build_ava_raw_prompt(line)
        original_dataset_name = self.dataset_name
        self.dataset_name = self._base_dataset_name()
        try:
            return super().build_prompt(line, video_llm=video_llm)
        finally:
            self.dataset_name = original_dataset_name


class OVBenchBBox1000New(OVBenchNew):
    DATASET_ALIASES = {
        'OVBench_BBox1000_New': 'OVBench',
        'OVBench_BBox1000_New_1fps': 'OVBench_1fps',
        'OVBench_BBox1000_New_2fps': 'OVBench_2fps',
        'OVBench_BBox1000_New_4fps': 'OVBench_4fps',
    }

    def __init__(self, dataset='OVBench_BBox1000_New', *args, ava_anno_json=None, **kwargs):
        ava_anno_json = (
            ava_anno_json
            or os.environ.get('OVBENCH_NEW_AVA_BBOX1000_ANNO_JSON', DEFAULT_AVA_BBOX1000_ANNO).strip()
        )
        super().__init__(dataset=dataset, *args, ava_anno_json=ava_anno_json, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [
            'OVBench_BBox1000_New',
            'OVBench_BBox1000_New_1fps',
            'OVBench_BBox1000_New_2fps',
            'OVBench_BBox1000_New_4fps',
        ]

    @classmethod
    def _default_base_ann_json(cls):
        env_path = os.environ.get('OVBENCH_NEW_BBOX1000_BASE_ANNO_JSON', '').strip()
        if env_path:
            return env_path
        return str(DEFAULT_BASE_BBOX1000_ANNO)
