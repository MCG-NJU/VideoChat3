# Environment variables used by this dataset:
# - ODVBENCH_ROOT: ODVBench video root directory.
# - ODVBENCH_ANNO_JSON: path to the ODVBench annotation JSON file.

# Download the JSON file from https://huggingface.co/datasets/ZZQ987/VideoChat3_Eval_Online_Json
# Place it under VLMEvalkit/json_data/odvbench

# Please ensure the following directory structure:
# 1
# VLMEvalKit/json_data/odvbench/
# ├── ODVbench.json
# 2
# export ODVBENCH_ANNO_JSON="VLMEvalkit/json_data/odvbench/ODVbench.json"

import json
import os
import string
import warnings
from collections import defaultdict

import pandas as pd
from PIL import Image

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.multiple_choice import mcq_vanilla_eval


FAIL_MSG = 'Failed to obtain answer via API.'


def _default_odvbench_root():
    return os.environ.get(
        'ODVBENCH_ROOT',
        '/mnt/petrelfs/zengxiangyu/Research_Zhang/s3/videogpu2/online-eval/OMQA-DS-ALL',
    ).strip()


def _default_odvbench_anno_json():
    return os.environ.get(
        'ODVBENCH_ANNO_JSON',
        '/mnt/petrelfs/zengxiangyu/Research_Zhang/eval_way/VLMEvalKit/json_data/odvbench/ODVbench.json',
    ).strip()


def _has_value(value):
    return value is not None and not (isinstance(value, float) and pd.isna(value))


def _float_token(value):
    if not _has_value(value):
        return 'na'
    return str(int(round(float(value) * 1000)))


def _format_seconds(value):
    if not _has_value(value):
        return None
    return f'{float(value):.2f}s'


class ODVBench(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    SYS = (
        "You are an expert driving analyst. Observe the streaming video of the driving scenario and answer the question relying strictly on the visual information.\n"
        'After watching the streaming video, you will be given a multiple-choice question. Select the single best option based on the visual evidence.\n'
    )
    POST_PROMPT = 'Answer with ONLY the single uppercase letter of the correct option.'

    def __init__(self, dataset='ODVBench', nframe=0, fps=-1, frames_limit=4096, min_pixels=28*28, max_pixels=448*448, total_pixels=65536*4*16*16, check_extracted_frames=True):
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames)

    @classmethod
    def supported_datasets(cls):
        return ['ODVBench', 'ODVBench_2fps', 'ODVBench_4fps']

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

        required = {'index', 'task', 'subtask', 'question', 'video', 'answer', 'A', 'B', 'C', 'D'}
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
            raise FileNotFoundError(f'ODVBench annotation json not found: {ann_json}')

        with open(ann_json, 'r', encoding='utf-8') as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f'ODVBench annotation json should be a list, got {type(records)}')

        rows = []
        for idx, record in enumerate(records):
            video_rel_path = str(record.get('video', '')).strip()
            if not video_rel_path:
                raise ValueError(f'ODVBench sample {idx} is missing `video`')

            video_path = osp.join(root_dir, video_rel_path)
            if not osp.exists(video_path):
                raise FileNotFoundError(f'ODVBench video not found: {video_path}')

            candidates = record.get('candidates', [])
            if not isinstance(candidates, list):
                raise ValueError(f'ODVBench sample {idx} has invalid `candidates`: {type(candidates)}')
            if len(candidates) < 2 or len(candidates) > 4:
                raise ValueError(
                    f'ODVBench sample {idx} has {len(candidates)} candidates, expected between 2 and 4'
                )

            option_map = {ch: '' for ch in 'ABCD'}
            candidate_texts = [str(x) for x in candidates]
            for ch, text in zip('ABCD', candidate_texts):
                option_map[ch] = text

            answer_text = str(record.get('answer', ''))
            if answer_text not in candidate_texts:
                raise ValueError(
                    f'ODVBench sample {idx} answer does not match any candidate: {answer_text!r}'
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
                    fps=record.get('fps'),
                )
            )

        data = pd.DataFrame(rows)
        dump(data, tsv_path)

    def prepare_dataset(self, dataset='ODVBench', root_dir=None, ann_json=None):
        root_dir = root_dir if root_dir is not None else _default_odvbench_root()
        ann_json = ann_json if ann_json is not None else _default_odvbench_anno_json()
        tsv_path = osp.join(root_dir, f'{dataset}.tsv')

        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f'ODVBench root directory not found: {root_dir}. '
                'Please set environment variable ODVBENCH_ROOT.'
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

    def _frame_paths_nframe(self, cache_key, num_frames):
        frame_root = osp.join(self.frame_root, cache_key)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def _frame_paths_fps_mode(self, cache_key, num_frames, fps_value):
        frame_root = osp.join(self.frame_root, cache_key)
        os.makedirs(frame_root, exist_ok=True)
        fps_token = str(fps_value).replace('.', 'p')
        tmpl = f'frame-{{}}-of-{{}}-{fps_token}fps.jpg'
        return [osp.join(frame_root, tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def _save_frames_with_decord(self, video_path, indices, frame_paths):
        import decord
        from decord import cpu

        vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=1)
        images = [Image.fromarray(vr[i].asnumpy()) for i in indices]
        for im, pth in zip(images, frame_paths):
            if not osp.exists(pth):
                im.save(pth)

    def _save_frames_with_av(self, video_path, indices, frame_paths):
        import av

        target_positions = defaultdict(list)
        for pos, idx in enumerate(indices):
            target_positions[idx].append(pos)

        seen = set()
        with av.open(video_path) as container:
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx not in target_positions:
                    continue
                if frame_idx in seen:
                    continue
                image = Image.fromarray(frame.to_ndarray(format='rgb24'))
                for pos in target_positions[frame_idx]:
                    pth = frame_paths[pos]
                    if not osp.exists(pth):
                        image.save(pth)
                seen.add(frame_idx)

        missing = [pth for pth in frame_paths if not osp.exists(pth)]
        if missing:
            raise RuntimeError(f'Failed to decode all requested ODVBench frames from {video_path}')

    def save_video_frames(self, line, video_llm=False, verbose=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_rel_path = str(line['video'])
        video_path = osp.join(self.data_root, video_rel_path)
        if not osp.exists(video_path):
            raise FileNotFoundError(f'ODVBench video not found: {video_path}')

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
        clip_frames = max(1, clip_end - clip_start)
        clip_duration = clip_frames / video_fps

        mode = f'{self.nframe}frame' if self.nframe > 0 else f'{self.fps}fps'
        cache_key = self._cache_key(video_rel_path, line.get('start'), line.get('end'), mode)

        if self.nframe > 0 and self.fps <= 0:
            step_size = clip_frames / (self.nframe + 1)
            indices = [clip_start + int(i * step_size) for i in range(1, self.nframe + 1)]
            indices = [min(max(clip_start, idx), clip_end - 1) for idx in indices]
            frame_paths = self._frame_paths_nframe(cache_key, len(indices))
            sample_fps = (len(indices) / clip_duration) if clip_duration > 0 else video_fps
        elif self.fps > 0:
            requested_frames = max(1, int(clip_duration * self.fps))
            truncated = False
            if requested_frames > self.frames_limit:
                requested_frames = self.frames_limit
                truncated = True

            if truncated:
                step_size = clip_frames / (requested_frames + 1)
                indices = [clip_start + int(i * step_size) for i in range(1, requested_frames + 1)]
                sample_fps = (requested_frames / clip_duration) if clip_duration > 0 else self.fps
            else:
                step_size = video_fps / self.fps
                indices = [clip_start + int(i * step_size) for i in range(requested_frames)]
                sample_fps = self.fps

            indices = [min(max(clip_start, idx), clip_end - 1) for idx in indices]
            frame_paths = self._frame_paths_fps_mode(cache_key, len(indices), sample_fps)
        else:
            raise ValueError('nframe and fps should be set with at least one valid value')

        if not np.all([osp.exists(p) for p in frame_paths]):
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
        segments = []
        if str(line.get('task', '')).strip():
            segments.append(f"Task: {line['task']}")
        if str(line.get('subtask', '')).strip():
            segments.append(f"Subtask: {line['subtask']}")

        start_text = _format_seconds(line.get('start'))
        end_text = _format_seconds(line.get('end'))
        if start_text is not None and end_text is not None:
            segments.append(f'Video segment: {start_text} to {end_text}')

        segments.append(f"Question:\n{line['question']}")
        segments.append('Options:')
        for ch in 'ABCD':
            value = line.get(ch, '')
            if _has_value(value) and str(value).strip():
                segments.append(f'{ch}. {value}')
        return '\n'.join(segments)

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
            message.extend(dict(type='image', value=pth) for pth in frame_paths)

        message.append(dict(type='text', value=self._build_question_block(line)))
        message.append(dict(type='text', value=self.POST_PROMPT))
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
        nproc = judge_kwargs.pop('nproc', 1)

        tmp_file = get_intermediate_file_path(eval_file, f'_{judge_name}_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{judge_name}_score')
        acc_file = get_intermediate_file_path(eval_file, f'_{judge_name}_acc', 'json')

        if not osp.exists(score_file):
            data = load(eval_file)
            meta = data[['index', 'answer']].copy()
            meta['index'] = meta['index'].astype(int)

            judge_model = None
            if judge_name not in [None, 'exact_matching']:
                try:
                    judge_model = build_judge(**judge_kwargs)
                    if hasattr(judge_model, 'working') and not judge_model.working():
                        warnings.warn('Judge model is not working properly, will use exact matching instead.')
                        warnings.warn(DEBUG_MESSAGE)
                        judge_model = None
                except Exception as err:
                    warnings.warn(f'Failed to build judge model {judge_name}: {err}. Using exact matching.')
                    judge_model = None

            scored = mcq_vanilla_eval(
                model=judge_model,
                data=data,
                meta=meta,
                nproc=nproc,
                result_file=tmp_file,
                dataset_name='ODVBench',
            )
            dump(scored, score_file)

        scored = load(score_file)
        result = {
            'overall_accuracy': float(np.mean(scored['hit'])) if len(scored) else 0.0,
            'task_accuracy': self._group_acc(scored, 'task'),
            'subtask_accuracy': self._group_acc(scored, 'subtask'),
        }
        dump(result, acc_file)
        return result
