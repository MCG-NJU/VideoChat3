# Environment variables used by this dataset:
# - RIVER_ROOT: RIVER_Bench media root directory.
# - RIVER_ANNO_DIR: directory containing RIVER_Bench annotation JSON files.

# Download the JSON file from https://huggingface.co/datasets/ZZQ987/VideoChat3_Eval_Online_Json
# Place it under VLMEvalkit/json_data/river
# Please ensure the following directory structure:
# 1
# VLMEvalKit/json_data/river/json
# 2
# export RIVER_ANNO_DIR="VLMEvalkit/json_data/river/json"

import json
import os
import re
import string
import warnings
from collections import defaultdict
from typing import Any

import pandas as pd
from PIL import Image

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from ..utils import track_progress_rich
from .utils.judge_util import DEBUG_MESSAGE, build_judge
from .utils.omnimmi import normalize_yesno, parse_python_dict_text
from .video_base import VideoBaseDataset


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _default_river_bench_root():
    return os.environ.get(
        'RIVER_ROOT',
        '/mnt/petrelfs/share_data/shiyansong/datasets/',
    ).strip()


def _default_river_bench_anno_dir():
    return os.environ.get(
        'RIVER_ANNO_DIR',
        '/mnt/petrelfs/zhangzhiqiu/Online/VLMEvalkit/json_data/river'
    ).strip()


def _has_value(value):
    return value is not None and not (isinstance(value, float) and pd.isna(value))


def _float_token(value):
    if not _has_value(value):
        return 'na'
    return str(int(round(float(value) * 1000)))


def _normalize_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    text = str(text).replace('\n', ' ').replace('\t', ' ')
    text = text.strip().strip('"').strip("'").strip('(').strip(')')
    text = re.sub(r'[^0-9a-zA-Z]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def _extract_option_letter(prediction, valid_letters='ABCDE'):
    if prediction is None or (isinstance(prediction, float) and pd.isna(prediction)):
        return ''
    text = str(prediction).strip()
    if not text:
        return ''
    letters = ''.join(valid_letters)

    patterns = [
        rf'^\s*\(?\s*([{letters}])\s*\)?\s*$',
        rf'^\s*\(?\s*([{letters}])\s*\)?[\.\:\-\)]',
        rf'(?:answer|option|best option|correct option)\s*(?:is|:)?\s*\(?\s*([{letters}])\s*\)?',
        rf'\b([{letters}])\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ''


def _match_option_text(prediction, option_map):
    pred_norm = _normalize_text(prediction)
    if not pred_norm:
        return ''
    matches = []
    for letter, option_text in option_map.items():
        opt_norm = _normalize_text(option_text)
        if not opt_norm:
            continue
        if pred_norm == opt_norm or opt_norm in pred_norm:
            matches.append((len(opt_norm), letter))
    if not matches:
        return ''
    matches.sort()
    return matches[-1][1]


def _list_frame_files(frame_dir):
    frame_files = []
    for name in sorted(os.listdir(frame_dir)):
        path = osp.join(frame_dir, name)
        if osp.isfile(path) and osp.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            frame_files.append(path)
    return frame_files


# def _extract_single_int(text):
#     text = _normalize_text(text)
#     if not text:
#         return None
#     numbers = re.findall(r'\b\d+\b', text)
#     if len(numbers) != 1:
#         return None
#     return int(numbers[0])


def _safe_model_name(name: str | None) -> str:
    if not name:
        return 'none'
    name = str(name)
    name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)
    return name[:128] if len(name) > 128 else name


def _river_judge_user_prompt(question: str, gt: str, pred: str) -> str:
    return (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {gt}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
    )


def score_instant(task, time_error): #################
    gap_dict={'short':15,'middle':30,'long':300,'very_long':1800}
    if time_error<-gap_dict[task]/2:
        return 0
    if abs(time_error)<=gap_dict[task]/2:
        return 1
    else:
        return max(1-(abs(time_error)-gap_dict[task]/2)/gap_dict[task]/2,0.)

class RIVER_Bench(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    SYS = (
        'Carefully watch the video and answer the question using only the information visible in the video.'
    )
    POST_PROMPT = 'Answer with ONLY the single uppercase letter of the correct option.'
    JUDGE_SYS = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )
    TASK_FILES = {
        # 离线版本
        'Live-Perception':                'Live-Perception.json',
        'Retro-Memory':                   'Retro-Memory.json',
        # 'Pro-Response-Instant':           'Pro-Response-Instant.json',           #mc还是mc；单次响应时间远远提前于reference-time，分数为0
        # 'Pro-Response-Streaming':         'Pro-Response-Streaming.json',         #open-ending给分
        # 构造滑动窗口，在相应的时间点让模型回答
        'Pro-Response-Instant-Sliding':   'Pro-Response-Instant-Sliding.json',   #mc还是mc，单次响应open-ending判断一下响应对错
        'Pro-Response-Streaming-Sliding': 'Pro-Response-Streaming-Sliding.json', #和原来一样，都是open-ending给分
    }

    def __init__(
        self,
        dataset='RIVER_Bench',
        nframe=0,
        fps=-1,
        frames_limit=512,
        online_mode=False,
        max_nframe=32,
        check_extracted_frames=True,
    ):
    # def __init__(self, dataset='RIVER_Bench', nframe=16, fps=-1, frames_limit=512): ########
        self.frames_limit = frames_limit
        self.online_mode = bool(online_mode)
        self.max_nframe = int(max_nframe)
        if self.online_mode and self.max_nframe <= 0:
            raise ValueError('online_mode requires max_nframe > 0')
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames)

    @classmethod
    def supported_datasets(cls):
        return ['RIVER_Bench']

    @staticmethod
    def _sample_tsv_valid(tsv_path): #检查代码合法
        if not osp.exists(tsv_path):
            return False
        try:
            data = load(tsv_path)
        except Exception:
            return False

        required = {
            'index', 'task', 'question', 'video', 'answer', 'answer_text', 'eval_mode',
            'A', 'B', 'C', 'D', 'E',
        }
        if not required.issubset(set(data.columns)):
            return False

        sample = data.head(min(32, len(data)))
        for _, row in sample.iterrows():
            eval_mode = str(row['eval_mode']).strip() ###########
            if eval_mode == 'mcq_letter':
                answer = str(row['answer']).strip().upper()
                if answer not in string.ascii_uppercase[:5]:
                    return False
        return True

    @classmethod
    def _build_tsv(cls, ann_dir, tsv_path):
        rows = []
        index = 0
        for task, filename in cls.TASK_FILES.items():
            ann_path = osp.join(ann_dir, filename)
            if not osp.exists(ann_path):
                raise FileNotFoundError(f'RIVER_Bench annotation json not found: {ann_path}')
            with open(ann_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            if not isinstance(records, list):
                raise ValueError(f'RIVER_Bench annotation json should be a list: {ann_path}')

            for record in records:
                video_rel_path = str(record.get('video', '')).strip()
                if not video_rel_path:
                    raise ValueError(f'RIVER_Bench sample {index} in {filename} is missing `video`')

                row = dict(
                    index=index,
                    task=str(record.get('task', task)),
                    subtask=str(record.get('subtask', '')),
                    question=str(record.get('question', '')),
                    video=video_rel_path,
                    start=record.get('start'),
                    end=record.get('end'),
                    time_reference=record.get('time_reference'),
                    fps=record.get('fps'),
                    A='',
                    B='',
                    C='',
                    D='',
                    E='',
                )

                candidates = record.get('candidates')
                # if isinstance(candidates, list):
                if isinstance(candidates, list) and len(candidates): #####过去、即时、instant中多选、
                    if len(candidates) < 2 or len(candidates) > 5:
                        raise ValueError(
                            f'RIVER_Bench sample {index} has {len(candidates)} candidates, expected between 2 and 5'
                        )
                    candidate_texts = [str(x) for x in candidates]
                    for ch, text in zip('ABCDE', candidate_texts):
                        row[ch] = text
                    answer_text = str(record.get('answer', ''))
                    if answer_text not in candidate_texts:
                        raise ValueError(
                            f'RIVER_Bench sample {index} answer does not match any candidate: {answer_text!r}'
                        )
                    row['answer'] = string.ascii_uppercase[candidate_texts.index(answer_text)]
                    row['answer_text'] = answer_text
                    row['eval_mode'] = 'mcq_letter'
                else:                           #####instant单纯响应、streaming多次
                    answer_text = str(record.get('answer', ''))
                    row['answer'] = answer_text
                    row['answer_text'] = answer_text
                    row['eval_mode'] = 'open_ending'

                rows.append(row)
                index += 1

        dump(pd.DataFrame(rows), tsv_path)

    def prepare_dataset(self, dataset='RIVER_Bench', root_dir=None, ann_dir=None):
        root_dir = root_dir if root_dir is not None else _default_river_bench_root()
        ann_dir = ann_dir if ann_dir is not None else _default_river_bench_anno_dir()
        tsv_path = osp.join(ann_dir, f'{dataset}.tsv')

        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f'RIVER_Bench root directory not found: {root_dir}. '
                'Please set environment variable RIVER_ROOT.'
            )
        if not osp.isdir(ann_dir):
            raise FileNotFoundError(
                f'RIVER_Bench annotation directory not found: {ann_dir}. '
                'Please set environment variable RIVER_ANNO_DIR.'
            )

        if not self._sample_tsv_valid(tsv_path):
            self._build_tsv(ann_dir, tsv_path)

        return dict(root=root_dir, data_file=tsv_path)

    @staticmethod
    def _cache_key(task, video_rel_path, start, end, mode, source_kind):
        stem = str(video_rel_path).replace('\\', '/').rstrip('/')
        stem = stem.rsplit('.', 1)[0]
        stem = stem.replace('/', '__')
        return f'{task}__{stem}__{source_kind}__s{_float_token(start)}__e{_float_token(end)}__{mode}'

    @staticmethod
    def _candidate_media_paths(root_dir, task, video_rel_path):
        rel = str(video_rel_path).replace('\\', '/').strip().lstrip('/')
        candidates = []

        def add(path):
            if path not in candidates:
                candidates.append(path)

        task = str(task).strip()
        add(osp.join(root_dir, rel))
        # add(osp.join(root_dir, 'chunked_videos', rel))
        # add(osp.join(root_dir, task, rel))
        # add(osp.join(root_dir, task, 'chunked_videos', rel))
        # add(osp.join(root_dir, 'video', rel))
        # add(osp.join(root_dir, 'video', 'chunked_videos', rel))

        base, ext = osp.splitext(rel)
        # if not ext:
        #     for suffix in VIDEO_EXTENSIONS:
        #         add(osp.join(root_dir, rel + suffix))
        #         add(osp.join(root_dir, 'chunked_videos', rel + suffix))
        #         add(osp.join(root_dir, task, rel + suffix))
        #         add(osp.join(root_dir, task, 'chunked_videos', rel + suffix))
        #         add(osp.join(root_dir, 'video', rel + suffix))
        #         add(osp.join(root_dir, 'video', 'chunked_videos', rel + suffix))

        return candidates

    def _resolve_media_path(self, line):
        candidates = self._candidate_media_paths(self.data_root, line.get('task', ''), line['video'])
        for path in candidates:
            if osp.isfile(path):
                return path, 'file'
            if osp.isdir(path):
                return path, 'dir'
        raise FileNotFoundError(
            f'RIVER_Bench media not found for task={line.get("task", "")}, video={line["video"]}. '
            f'Tried: {candidates}'
        )

    @staticmethod
    def _resolve_clip(total_frames, source_fps, line):
        clip_start = 0
        clip_end = total_frames

        if _has_value(line.get('start')):
            start = float(line['start'])
            if source_fps and source_fps > 0:
                clip_start = max(0, int(np.floor(start * source_fps)))
            elif start.is_integer():
                clip_start = max(0, min(total_frames - 1, int(start)))

        if _has_value(line.get('end')):
            end = float(line['end'])
            if source_fps and source_fps > 0:
                clip_end = min(total_frames, int(np.ceil(end * source_fps)))
            elif end.is_integer():
                clip_end = min(total_frames, int(end))

        clip_end = max(clip_start + 1, clip_end)
        clip_end = min(total_frames, clip_end)
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

    @staticmethod
    def _save_frames_with_decord(video_path, indices, frame_paths):
        import decord
        from decord import cpu

        vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=1)
        images = [Image.fromarray(vr[i].asnumpy()) for i in indices]
        for im, pth in zip(images, frame_paths):
            if not osp.exists(pth):
                try:
                    im.save(pth)
                except Exception as e:
                    print(f'Error saving frame {pth}: {e}')
                    continue

    @staticmethod
    def _save_frames_with_av(video_path, indices, frame_paths):
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
                    pth = frame_paths[pos]
                    try:
                        if not osp.exists(pth):
                            image.save(pth)
                    except Exception as e:
                        print(f'Error saving frame {pth}: {e}')
                        # image.save(pth)
                        continue
                seen.add(frame_idx)

        missing = [pth for pth in frame_paths if not osp.exists(pth)]
        if missing:
            raise RuntimeError(f'Failed to decode all requested RIVER_Bench frames from {video_path}')

    @staticmethod
    def _save_frames_from_dir(frame_files, indices, frame_paths):
        for frame_idx, out_path in zip(indices, frame_paths):
            if osp.exists(out_path):
                continue
            try:
                image = Image.open(frame_files[frame_idx]).convert('RGB')
                image.save(out_path)
            except Exception as e:
                print(f'Error saving frame {out_path}: {e}')
                continue

    def save_video_frames(self, line, video_llm=False, verbose=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        media_path, source_kind = self._resolve_media_path(line)
        mode = f'{self.nframe}frame' if self.nframe > 0 else f'{self.fps}fps'
        if self.online_mode:
            mode = f'{mode}_online_max{self.max_nframe}'
        cache_key = self._cache_key(
            line.get('task', ''),
            line['video'],
            line.get('start'),
            line.get('end'),
            mode,
            source_kind,
        )

        if source_kind == 'file':
            use_av = False
            try:
                import decord
                from decord import cpu

                vr = decord.VideoReader(media_path, ctx=cpu(0), num_threads=1)
                source_fps = float(vr.get_avg_fps())
                total_frames = len(vr)
                if source_fps <= 0:
                    raise ValueError('invalid video fps')
            except Exception as err:
                warnings.warn(f'decord failed for {media_path}: {err}. Falling back to av.')
                use_av = True
                import av

                with av.open(media_path) as container:
                    stream = container.streams.video[0]
                    source_fps = float(stream.average_rate) if stream.average_rate else 30.0
                    total_frames = stream.frames
                    if not total_frames:
                        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
                        total_frames = max(1, int(duration * source_fps))

            clip_start, clip_end = self._resolve_clip(total_frames, source_fps, line)
            indices, clip_duration, sample_fps, clip_frames = self._sample_indices(clip_start, clip_end, source_fps)
            if self.nframe > 0 and self.fps <= 0:
                frame_paths = self._frame_paths_nframe(cache_key, len(indices))
            else:
                frame_paths = self._frame_paths_fps_mode(cache_key, len(indices), sample_fps)

            if not np.all([osp.exists(p) for p in frame_paths]):
                if use_av:
                    self._save_frames_with_av(media_path, indices, frame_paths)
                else:
                    self._save_frames_with_decord(media_path, indices, frame_paths)
        else:
            frame_files = _list_frame_files(media_path)
            if not frame_files:
                raise RuntimeError(f'RIVER_Bench frame directory is empty: {media_path}')

            source_fps = float(line['fps']) if _has_value(line.get('fps')) else None
            total_frames = len(frame_files)
            clip_start, clip_end = self._resolve_clip(total_frames, source_fps, line)
            indices, clip_duration, sample_fps, clip_frames = self._sample_indices(clip_start, clip_end, source_fps)
            if self.nframe > 0 and self.fps <= 0:
                frame_paths = self._frame_paths_nframe(cache_key, len(indices))
            else:
                frame_paths = self._frame_paths_fps_mode(cache_key, len(indices), sample_fps)

            if not np.all([osp.exists(p) for p in frame_paths]):
                self._save_frames_from_dir(frame_files, indices, frame_paths)

        video_info = {
            'source_kind': source_kind,
            'fps': source_fps,
            'n_frames': total_frames,
            'duration': (total_frames / source_fps) if source_fps else None,
            'clip_start': clip_start,
            'clip_end': clip_end,
            'clip_duration': clip_duration,
            'clip_frames': clip_frames,
            'sample_fps': sample_fps,
            'sample_n_frame': len(indices),
        }
        return frame_paths, video_info

    def _build_mcq_block(self, line):
        segments = [f"Question:\n{line['question']}", 'Options:']
        for ch in 'ABCDE':
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
                )
            )
        else:
            message.extend(dict(type='image', value=pth) for pth in frame_paths)

        if str(line.get('eval_mode', '')).strip() == 'mcq_letter':
            message.append(dict(type='text', value=self._build_mcq_block(line)))
            message.append(dict(type='text', value=self.POST_PROMPT))
            message.append(dict(type='text', value='Best option:(', role='assistant'))
        else:
            message.append(dict(type='text', value=str(line['question'])))
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
        model_name = judge_kwargs.get('model', 'qwen3-235b-a22b-thinking-2507')
        judge_kwargs['model'] = model_name
        nproc = int(judge_kwargs.pop('nproc', 1))
        force_reeval = bool(judge_kwargs.pop('force_reeval', False))

        score_file = get_intermediate_file_path(eval_file, f'_{_safe_model_name(model_name)}_score')
        tmp_file = get_intermediate_file_path(eval_file, f'_{_safe_model_name(model_name)}_tmp', 'pkl')
        acc_file = get_intermediate_file_path(eval_file, '_acc', 'json')

        if not osp.exists(score_file) or force_reeval:
            data = load(eval_file)
            scored = data.copy()
            if 'prediction' not in scored.columns:
                raise ValueError(f'Prediction column not found in evaluation file: {eval_file}')

            if model_name in (None, 'exact_matching'):
                warnings.warn(
                    'RIVER_Bench open_ending requires an LLM judge; got exact_matching/None. '
                    'Will mark open_ending samples as failed.'
                )
                judge_model = None
            else:
                try:
                    judge_model = build_judge(system_prompt=self.JUDGE_SYS, **judge_kwargs)
                except Exception as e:
                    warnings.warn(f'Failed to build judge model ({model_name}): {type(e)}: {e}')
                    warnings.warn(DEBUG_MESSAGE)
                    judge_model = None

            lines = [scored.iloc[i] for i in range(len(scored))]
            indices = [int(line['index']) if 'index' in line else int(i) for i, line in enumerate(lines)]

            cached: dict[int, dict[str, Any]] = {}
            if osp.exists(tmp_file) and not force_reeval:
                try:
                    cached = load(tmp_file)
                except Exception:
                    cached = {}

            def _judge_one(model, line):
                idx = int(line['index']) if _has_value(line.get('index')) else None
                question = str(line.get('question', '')).strip()
                gt_norm = _normalize_text(line.get('answer', ''))
                parsed = _normalize_text(line.get('prediction', ''))

                if model is None:
                    return {'hit': 0, 'raw': '', 'error': 'no_judge_model'}

                user_prompt = _river_judge_user_prompt(question, gt_norm, parsed)
                raw = model.generate([dict(type='text', value=user_prompt)], dataset='RIVER_Bench')
                parsed_resp = parse_python_dict_text(raw)
                if not parsed_resp:
                    return {'hit': 0, 'raw': raw, 'error': 'parse_failed'}
                yn = normalize_yesno(parsed_resp.get('pred', ''))
                hit = 1 if yn == 'yes' else 0
                return {'hit': hit, 'raw': raw, 'error': ''}

            pending_tups = []
            pending_indices = []
            for line, idx in zip(lines, indices):
                eval_mode = str(line.get('eval_mode', '')).strip()
                task = str(line.get('task', '')).strip()
                if eval_mode != 'open_ending' or task == 'Pro-Response-Instant':
                    continue
                if idx in cached:
                    continue
                pending_tups.append((judge_model, line))
                pending_indices.append(idx)

            if len(pending_indices):
                _ = track_progress_rich(
                    _judge_one,
                    pending_tups,
                    nproc=max(1, nproc),
                    chunksize=max(1, nproc),
                    keys=pending_indices,
                    save=tmp_file,
                )

            cached = load(tmp_file) if osp.exists(tmp_file) else cached

            parsed_outputs = []
            hits = []
            for _, row in scored.iterrows():
                task = row.get('task', '')
                eval_mode = str(row.get('eval_mode', '')).strip()
                pred = row.get('prediction', '')
                gt = row.get('answer', '')

                if eval_mode == 'mcq_letter':
                    option_map = {
                        ch: str(row.get(ch, '')).strip()
                        for ch in 'ABCDE'
                        if _has_value(row.get(ch)) and str(row.get(ch, '')).strip()
                    }
                    parsed = _extract_option_letter(pred, valid_letters=''.join(option_map.keys()))
                    if not parsed and option_map:
                        parsed = _match_option_text(pred, option_map)
                    hit = int(bool(parsed) and parsed == str(gt).strip().upper())
                elif eval_mode == 'substring_match':
                    parsed = _normalize_text(pred)
                    gt_norm = _normalize_text(gt)
                    hit = int(bool(parsed) and bool(gt_norm) and gt_norm in parsed)
                # elif eval_mode == 'integer_match':
                #     parsed_int = _extract_single_int(pred)
                #     gt_int = _extract_single_int(gt)
                #     parsed = '' if parsed_int is None else str(parsed_int)
                #     hit = int(parsed_int is not None and gt_int is not None and parsed_int == gt_int)
                elif eval_mode == 'open_ending':
                    end = float(row.get('end', ''))  #####question-time
                    question = row.get('question', '') #####
                    subtask = row.get('subtask', '') #####
                    time_reference = float(row.get('time_reference', '')) #####
                    parsed = _normalize_text(pred)
                    gt_norm = _normalize_text(gt)
                    # hit = int(bool(parsed) and bool(gt_norm) and gt_norm in parsed)
                    if task == 'Pro-Response-Instant':
                        hit = 0
                    else:
                        idx = int(row['index']) if _has_value(row.get('index')) else None
                        item = cached.get(idx, {}) if idx is not None else {}
                        hit = int(item.get('hit', 0))
                        # print(f'end-time_reference {end} {time_reference} {type(end)} {type(time_reference)}')
                        # hit *= score_instant(subtask, end-time_reference)
                else:
                    parsed = ''
                    hit = 0

                parsed_outputs.append(parsed)
                hits.append(hit)

            scored['parsed_output'] = parsed_outputs
            scored['hit'] = hits
            dump(scored, score_file)

        scored = load(score_file)

        # subtask_acc = self._group_acc(scored, 'subtask')
        task_acc = self._group_acc(scored, 'task')
        result = {
            'overall_accuracy': float(np.mean(list(task_acc.values()))) if task_acc else 0.0,
            'task_accuracy': task_acc,
            # 'subtask_accuracy': subtask_acc,
            # 'subtask_macro_accuracy': float(np.mean(list(subtask_acc.values()))) if subtask_acc else 0.0,
        }
        dump(result, acc_file)
        return result
