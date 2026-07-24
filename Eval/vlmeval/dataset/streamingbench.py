# Environment variables used by this dataset:
# - STREAMINGBENCH_ROOT: StreamingBench video root directory.
# - STREAMINGBENCH_ANNO_JSON: path to the StreamingBench annotation JSON file.

# Download the JSON file from https://huggingface.co/datasets/ZZQ987/VideoChat3_Eval_Online_Json
# Place it under VLMEvalkit/json_data/streamingbench
# Please ensure the following directory structure:
# 1
# VLMEvalKit/json_data/streamingbench/
# ├── real_time_visual_understanding.json
# 2
# export STREAMINGBENCH_ANNO_JSON="VLMEvalkit/json_data/streamingbench/real_time_visual_understanding.json"

import json
import os
import re
import string
import warnings
from collections import defaultdict

import pandas as pd
from PIL import Image

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset


def _default_streamingbench_root():
    return os.environ.get(
        'STREAMINGBENCH_ROOT',
        '/mnt/petrelfs/zengxiangyu/Research_Zhang/s3/videogpu2/online-eval/StreamingBench/',
    ).strip()


def _default_streamingbench_anno_json():
    return os.environ.get(
        'STREAMINGBENCH_ANNO_JSON',
        '/mnt/petrelfs/zengxiangyu/Research_Zhang/eval_way/VLMEvalKit/json_data/streamingbench/real_time_visual_understanding.json',
    ).strip()


def _has_value(value):
    return value is not None and not (isinstance(value, float) and pd.isna(value))


def _float_token(value):
    if not _has_value(value):
        return 'na'
    return str(int(round(float(value) * 1000)))


_PERIOD_STRIP = re.compile(r'(?!<=\d)(\.)(?!\d)')
_COMMA_STRIP = re.compile(r'(\d)(\,)(\d)')
_OPTION_REGEX = re.compile(r'^([A-E])\.\s*(.+)$', re.IGNORECASE)
_LETTER_REGEX = re.compile(r'\b([A-E])\b', re.IGNORECASE)
_PUNCT = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']


def _process_mcq_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''

    text = str(text).strip()
    if not text:
        return ''

    match = _OPTION_REGEX.match(text)
    if match:
        return match.group(1).upper()

    out_text = text.replace('\n', ' ').replace('\t', ' ').strip()
    for punct in _PUNCT:
        if (punct + ' ' in out_text or ' ' + punct in out_text) or re.search(_COMMA_STRIP, out_text) is not None:
            out_text = out_text.replace(punct, '')
        else:
            out_text = out_text.replace(punct, ' ')
    out_text = _PERIOD_STRIP.sub('', out_text, re.UNICODE)
    out_text = out_text.strip("'").strip('"').strip(')').strip('(').strip().lower()

    letter_match = _LETTER_REGEX.search(out_text)
    if letter_match:
        return letter_match.group(1).upper()

    return out_text


class StreamingBench(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    SYS = (
        'You are a real-time streaming video analysis assistant. Please observe the live video stream and answer the question strictly based on the visual information.\n'
        'After watching the streaming video, you will be given a multiple-choice question. Select the single best option based on the visual evidence.\n'
    )
    POST_PROMPT = "Answer with the option's letter from the given choices directly."

    def __init__(
        self,
        dataset='StreamingBench',
        nframe=0,
        fps=-1,
        frames_limit=4096,
        min_pixels=28*28,
        max_pixels=448*448,
        total_pixels=65536*4*16*16,
        online_mode=False,
        max_nframe=32,
        check_extracted_frames=True,
    ):
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.online_mode = bool(online_mode)
        self.max_nframe = int(max_nframe)
        if self.online_mode and self.max_nframe <= 0:
            raise ValueError('online_mode requires max_nframe > 0')
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames)

    @classmethod
    def supported_datasets(cls):
        return ['StreamingBench', 'StreamingBench_2fps', 'StreamingBench_4fps']

    @staticmethod
    def _cache_key(video_rel_path, start, end, mode):
        stem = str(video_rel_path).replace('\\', '/')
        stem = stem.rsplit('.', 1)[0]
        stem = stem.replace('/', '__')
        return f'{stem}__s{_float_token(start)}__e{_float_token(end)}__{mode}'

    @staticmethod
    def _sample_tsv_valid(tsv_path, root_dir):
        if not osp.exists(tsv_path):
            return False
        try:
            data = load(tsv_path)
        except Exception:
            return False

        required = {'index', 'task', 'subtask', 'question', 'video', 'answer', 'answer_text', 'A', 'B', 'C', 'D', 'start', 'end'}
        if not required.issubset(set(data.columns)):
            return False

        sample = data.head(min(16, len(data)))
        for _, row in sample.iterrows():
            video_path = osp.join(root_dir, str(row['video']))
            if not osp.exists(video_path):
                return False
            answer = str(row['answer']).strip().upper()
            if answer not in string.ascii_uppercase[:4]:
                return False
        return True

    @staticmethod
    def _build_tsv(root_dir, ann_json, tsv_path):
        if not osp.exists(ann_json):
            raise FileNotFoundError(f'StreamingBench annotation json not found: {ann_json}')

        with open(ann_json, 'r', encoding='utf-8') as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f'StreamingBench annotation json should be a list, got {type(records)}')

        rows = []
        for idx, record in enumerate(records):
            video_rel_path = str(record.get('video', '')).strip()
            if not video_rel_path:
                raise ValueError(f'StreamingBench sample {idx} is missing `video`')

            video_path = osp.join(root_dir, video_rel_path)
            if not osp.exists(video_path):
                raise FileNotFoundError(f'StreamingBench video not found: {video_path}')

            candidates = record.get('candidates', [])
            if not isinstance(candidates, list):
                raise ValueError(f'StreamingBench sample {idx} has invalid `candidates`: {type(candidates)}')
            if len(candidates) < 2 or len(candidates) > 4:
                raise ValueError(
                    f'StreamingBench sample {idx} has {len(candidates)} candidates, expected between 2 and 4'
                )

            option_map = {ch: '' for ch in 'ABCD'}
            candidate_texts = [str(x) for x in candidates]
            for ch, text in zip('ABCD', candidate_texts):
                option_map[ch] = text

            answer_text = str(record.get('answer', ''))
            if answer_text not in candidate_texts:
                raise ValueError(
                    f'StreamingBench sample {idx} answer does not match any candidate: {answer_text!r}'
                )
            answer_idx = candidate_texts.index(answer_text)
            answer_letter = string.ascii_uppercase[answer_idx]

            rows.append(
                dict(
                    index=idx,
                    task=str(record.get('task', '')),
                    subtask=str(record.get('subtask', '')),
                    question=str(record.get('question', '')),
                    video=video_rel_path,
                    answer=answer_letter,
                    answer_text=answer_text,
                    A=option_map['A'],
                    B=option_map['B'],
                    C=option_map['C'],
                    D=option_map['D'],
                    start=record.get('start'),
                    end=record.get('end'),
                )
            )

        dump(pd.DataFrame(rows), tsv_path)

    def prepare_dataset(self, dataset='StreamingBench', root_dir=None, ann_json=None):
        root_dir = root_dir if root_dir is not None else _default_streamingbench_root()
        ann_json = ann_json if ann_json is not None else _default_streamingbench_anno_json()
        tsv_path = osp.join(root_dir, f'{dataset}.tsv')

        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f'StreamingBench root directory not found: {root_dir}. '
                'Please set environment variable STREAMINGBENCH_ROOT.'
            )

        if not self._sample_tsv_valid(tsv_path, root_dir):
            self._build_tsv(root_dir, ann_json, tsv_path)

        return dict(root=root_dir, data_file=tsv_path)

    def _resolve_clip(self, line, video_fps, n_frames):
        clip_start = 0
        clip_end = n_frames

        if _has_value(line.get('start')):
            clip_start = max(0, int(np.floor(float(line['start']) * video_fps)))
        if _has_value(line.get('end')):
            clip_end = min(n_frames, int(np.ceil(float(line['end']) * video_fps)))

        clip_end = max(clip_start + 1, clip_end)
        clip_end = min(n_frames, clip_end)
        return clip_start, clip_end

    def _frame_name_suffix(self):
        if self.online_mode:
            return f'-online-max{self.max_nframe}'
        return ''

    def _frame_paths_nframe(self, cache_key, num_frames):
        frame_root = osp.join(self.frame_root, cache_key)
        os.makedirs(frame_root, exist_ok=True)
        suffix = self._frame_name_suffix()
        tmpl = f'frame-{{}}-of-{{}}{suffix}.jpg'
        return [osp.join(frame_root, tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def _frame_paths_fps_mode(self, cache_key, num_frames, fps_value):
        frame_root = osp.join(self.frame_root, cache_key)
        os.makedirs(frame_root, exist_ok=True)
        fps_token = str(fps_value).replace('.', 'p')
        suffix = self._frame_name_suffix()
        tmpl = f'frame-{{}}-of-{{}}-{fps_token}fps{suffix}.jpg'
        return [osp.join(frame_root, tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def _sample_indices_online(self, clip_start, clip_end, source_fps):
        clip_frames = max(1, clip_end - clip_start)
        source_fps = float(source_fps) if source_fps and source_fps > 0 else float(self.fps if self.fps > 0 else 1.0)
        clip_duration = clip_frames / source_fps if source_fps > 0 else float(clip_frames)

        if self.fps <= 0:
            raise ValueError('online_mode requires fps > 0 as sample fps')

        sample_fps = float(self.fps)
        step_size = source_fps / sample_fps
        requested_frames = max(1, int(clip_duration * sample_fps))
        num_frames = min(self.max_nframe, requested_frames)

        if requested_frames <= self.max_nframe:
            indices = [clip_start + int(i * step_size) for i in range(num_frames)]
        else:
            start_i = requested_frames - num_frames
            indices = [clip_start + int(i * step_size) for i in range(start_i, requested_frames)]

        indices = [min(max(clip_start, idx), clip_end - 1) for idx in indices]
        return indices, clip_duration, sample_fps, clip_frames

    def _sample_indices(self, clip_start, clip_end, source_fps):
        if self.online_mode:
            return self._sample_indices_online(clip_start, clip_end, source_fps)

        clip_frames = max(1, clip_end - clip_start)
        source_fps = float(source_fps) if source_fps and source_fps > 0 else float(self.fps if self.fps > 0 else 1.0)
        clip_duration = clip_frames / source_fps if source_fps > 0 else float(clip_frames)

        if self.nframe > 0 and self.fps <= 0:
            step_size = clip_frames / (self.nframe + 1)
            indices = [clip_start + int(i * step_size) for i in range(1, self.nframe + 1)]
            sample_fps = (len(indices) / clip_duration) if clip_duration > 0 else source_fps
        elif self.fps > 0:
            requested_frames = max(1, int(clip_duration * self.fps))
            truncated = requested_frames > self.frames_limit
            if truncated:
                requested_frames = self.frames_limit

            if truncated or source_fps <= 0:
                step_size = clip_frames / (requested_frames + 1)
                indices = [clip_start + int(i * step_size) for i in range(1, requested_frames + 1)]
                sample_fps = (requested_frames / clip_duration) if clip_duration > 0 else self.fps
            else:
                step_size = source_fps / self.fps
                indices = [clip_start + int(i * step_size) for i in range(requested_frames)]
                sample_fps = self.fps
        else:
            raise ValueError('nframe and fps should be set with at least one valid value')

        indices = [min(max(clip_start, idx), clip_end - 1) for idx in indices]
        return indices, clip_duration, sample_fps, clip_frames

    def _save_frames_with_decord(self, video_path, indices, frame_paths):
        import decord
        from decord import cpu

        vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=1)
        images = [Image.fromarray(vr[i].asnumpy()) for i in indices]
        for image, path in zip(images, frame_paths):
            if not osp.exists(path):
                image.save(path)

    def _save_frames_with_av(self, video_path, indices, frame_paths):
        import av

        target_positions = defaultdict(list)
        for pos, idx in enumerate(indices):
            target_positions[idx].append(pos)

        seen = set()
        with av.open(video_path) as container:
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx not in target_positions or frame_idx in seen:
                    continue
                image = Image.fromarray(frame.to_ndarray(format='rgb24'))
                for pos in target_positions[frame_idx]:
                    path = frame_paths[pos]
                    if not osp.exists(path):
                        image.save(path)
                seen.add(frame_idx)

        missing = [path for path in frame_paths if not osp.exists(path)]
        if missing:
            raise RuntimeError(f'Failed to decode all requested StreamingBench frames from {video_path}')

    def save_video_frames(self, line, video_llm=False, verbose=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_rel_path = str(line['video'])
        video_path = osp.join(self.data_root, video_rel_path)
        if not osp.exists(video_path):
            raise FileNotFoundError(f'StreamingBench video not found: {video_path}')

        use_av = False
        try:
            import decord
            from decord import cpu

            vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=1)
            video_fps = float(vr.get_avg_fps())
            n_frames = len(vr)
            if video_fps <= 0:
                raise ValueError('invalid video fps')
        except Exception as err:
            warnings.warn(f'decord failed for {video_path}: {err}. Falling back to av.')
            use_av = True
            import av

            with av.open(video_path) as container:
                stream = container.streams.video[0]
                video_fps = float(stream.average_rate) if stream.average_rate else 30.0
                n_frames = stream.frames
                if not n_frames:
                    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
                    n_frames = max(1, int(duration * video_fps))

        clip_start, clip_end = self._resolve_clip(line, video_fps, n_frames)

        mode = f'{self.nframe}frame' if self.nframe > 0 else f'{self.fps}fps'
        if self.online_mode:
            mode = f'{mode}_online_max{self.max_nframe}'
        cache_key = self._cache_key(video_rel_path, line.get('start'), line.get('end'), mode)

        indices, clip_duration, sample_fps, clip_frames = self._sample_indices(
            clip_start, clip_end, video_fps
        )
        if self.nframe > 0 and self.fps <= 0:
            frame_paths = self._frame_paths_nframe(cache_key, len(indices))
        else:
            frame_paths = self._frame_paths_fps_mode(cache_key, len(indices), sample_fps)

        if not np.all([osp.exists(path) for path in frame_paths]):

            print("extract frames from video: ", video_path)
            if use_av:
                self._save_frames_with_av(video_path, indices, frame_paths)
            else:
                self._save_frames_with_decord(video_path, indices, frame_paths)

        video_info = {
            'fps': video_fps,
            'n_frames': n_frames,
            'duration': n_frames / video_fps,
            'clip_start': clip_start,
            'clip_end': clip_end,
            'clip_duration': clip_duration,
            'sample_fps': sample_fps,
            'sample_n_frame': len(indices),
        }
        return frame_paths, video_info

    def _build_question_block(self, line):
        lines = [str(line['question']).strip()]
        for ch in 'ABCD':
            value = line.get(ch, '')
            if _has_value(value) and str(value).strip():
                lines.append(f'{ch}. {value}')
        lines.append(self.POST_PROMPT)
        return '\n'.join(lines)

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        message = [dict(type='text', value=self.SYS, role='system')]
        frame_paths, video_info = self.save_video_frames(line, video_llm=video_llm)

        if video_llm:
            message.append(
                dict(
                    type='video',
                    value=frame_paths,
                    sample_fps=video_info['sample_fps'],
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    total_pixels=self.total_pixels,
                )
            )
        else:
            message.extend(dict(type='image', value=path) for path in frame_paths)

        message.append(dict(type='text', value=self._build_question_block(line)))
        message.append(dict(type='text', value='Best option:(', role='assistant'))
        return message

    @staticmethod
    def _group_acc(data, column):
        if column not in data.columns:
            return {}
        groups = {}
        valid_values = sorted({str(x) for x in data[column] if not pd.isna(x) and str(x).strip()})
        for value in valid_values:
            subset = data[data[column].astype(str) == value]
            groups[value] = float(np.mean(subset['hit'])) if len(subset) else 0.0
        return groups

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'Evaluation file should be in xlsx/json/tsv format'
        )

        judge_name = judge_kwargs.setdefault('model', 'exact_matching')
        if judge_name not in [None, 'exact_matching']:
            warnings.warn(
                'StreamingBench uses its built-in exact-matching evaluation to stay aligned with lmms_eval.'
            )

        score_file = get_intermediate_file_path(eval_file, f'_{judge_name}_score')
        acc_file = get_intermediate_file_path(eval_file, f'_{judge_name}_acc', 'json')

        if not osp.exists(score_file):
            data = load(eval_file)
            if 'prediction' not in data.columns:
                raise ValueError(f'Prediction column not found in evaluation file: {eval_file}')
            scored = data.copy()
            scored['pred_processed'] = [_process_mcq_text(pred) for pred in scored['prediction']]
            scored['gt_processed'] = [_process_mcq_text(ans) for ans in scored['answer']]
            scored['hit'] = [int(pred == gt) for pred, gt in zip(scored['pred_processed'], scored['gt_processed'])]
            dump(scored, score_file)

        scored = load(score_file)
        result = {
            'overall_accuracy': float(np.mean(scored['hit'])) if len(scored) else 0.0,
            'task_accuracy': self._group_acc(scored, 'task'),
            'subtask_accuracy': self._group_acc(scored, 'subtask'),
        }
        dump(result, acc_file)
        return result
