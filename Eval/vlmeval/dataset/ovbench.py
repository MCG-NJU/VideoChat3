# Environment variables used by this dataset:
# - OVBENCH_ROOT: OVBench media root directory.
# - OVBENCH_ANNO_JSON: path to the flat OVBench split annotation JSON.
# - OVBENCH_ANNO_DIR: legacy annotation directory; standard split configs use OVBENCH_ANNO_JSON.

# It is recommended to use OVBenchNew or OVBenchBBox1000New

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
from ..utils import read_frames_auto
from .video_base import VideoBaseDataset


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpeg', '.mpg', '.m4v', '.gif')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _default_ovbench_root():
    return os.environ.get(
        'OVBENCH_ROOT',
        '/disk/zdata1/home/zhangqingyu/data/img/OVBench',
    ).strip()


def _default_ovbench_anno_dir():
    return os.environ.get(
        'OVBENCH_ANNO_DIR',
        '/disk/zdata1/home/zhangqingyu/work/StreamForest-main/anno/eval/OVBench/json',
    ).strip()


def _default_ovbench_anno_json():
    return os.environ.get(
        'OVBENCH_ANNO_JSON',
        '/disk/zdata1/home/zhangqingyu/work/StreamForest-main/anno/eval/OVBench/ovbench_split.json',
    ).strip()


def _default_ovbench_streamforest_anno_json():
    return '/disk/zdata1/home/zhangqingyu/work/StreamForest-main/anno/eval/OVBench/ovbench_split_with_fps.json'


def _default_ovbench_qwen_anno_json():
    return '/disk/zdata1/home/zhangqingyu/work/StreamForest-main/anno/eval/OVBench/ovbench_split_with_fps_ava_raw_frame.json'


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


def _normalize_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    text = str(text).replace('\n', ' ').replace('\t', ' ')
    text = text.strip().strip('"').strip("'").strip('(').strip(')')
    text = re.sub(r'[^0-9a-zA-Z]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


SPLIT_RECORD_FIELDS = {
    'id', 'video_id', 'question', 'answer', 'answer_type', 'sub_answer_type', 'options', 'start', 'end', 'video_time',
}


def _load_json_records(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f'OVBench annotation json should be a list: {json_path}')
    return records


def _is_split_annotation_record(record):
    return isinstance(record, dict) and SPLIT_RECORD_FIELDS.issubset(set(record.keys()))


def _is_split_annotation_json(json_path):
    if not osp.isfile(json_path):
        return False
    try:
        records = _load_json_records(json_path)
    except Exception:
        return False
    if not records:
        return False
    return _is_split_annotation_record(records[0])


def _resolve_split_annotation_json(ann_json=None, ann_dir=None, dataset='OVBench'):
    env_ann_json = os.environ.get('OVBENCH_ANNO_JSON', '').strip()
    explicit_candidates = []
    if ann_json is not None:
        explicit_candidates.append(str(ann_json).strip())
    elif env_ann_json:
        explicit_candidates.append(env_ann_json)

    default_candidates = []
    if not explicit_candidates:
        if dataset in {'OVBench_2fps', 'OVBench_4fps'}:
            default_candidates.append(_default_ovbench_qwen_anno_json())
        if dataset == 'OVBench_StreamForest':
            default_candidates.append(_default_ovbench_streamforest_anno_json())
        default_candidates.append(_default_ovbench_anno_json())
    if ann_dir:
        ann_dir = str(ann_dir).strip()
        ann_parent_dir = osp.dirname(ann_dir.rstrip('/'))
        if dataset in {'OVBench_2fps', 'OVBench_4fps'}:
            default_candidates.extend(
                [
                    osp.join(ann_dir, 'ovbench_split_with_fps_ava_raw_frame.json'),
                    osp.join(ann_parent_dir, 'ovbench_split_with_fps_ava_raw_frame.json'),
                ]
            )
        if dataset == 'OVBench_StreamForest':
            default_candidates.extend(
                [
                    osp.join(ann_dir, 'ovbench_split_with_fps.json'),
                    osp.join(ann_parent_dir, 'ovbench_split_with_fps.json'),
                ]
            )
        default_candidates.extend(
            [
                osp.join(ann_dir, 'ovbench_split.json'),
                osp.join(ann_parent_dir, 'ovbench_split.json'),
            ]
        )

    checked = set()
    for candidate in explicit_candidates + default_candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        if _is_split_annotation_json(candidate):
            return candidate

        if candidate in explicit_candidates and osp.isfile(candidate):
            raise ValueError(
                'OVBench now expects the flat split annotation json with fields '
                f'{sorted(SPLIT_RECORD_FIELDS)}. Please point OVBENCH_ANNO_JSON to a valid OVBench split json.'
            )

    raise FileNotFoundError(
        'OVBench split annotation json not found. Please set OVBENCH_ANNO_JSON to a valid OVBench split json.'
    )


def _strip_option_prefix(text):
    text = str(text).strip()
    match = re.match(r'^[A-E]\.\s*(.*)$', text)
    return match.group(1).strip() if match else text


def _infer_subset(video_rel_path):
    rel = str(video_rel_path).replace('\\', '/').strip().lstrip('/')
    parts = [part for part in rel.split('/') if part]
    return parts[0] if parts else ''


def _parse_question_idx_from_sample_id(sample_id):
    if not _has_value(sample_id):
        return None
    match = re.search(r'__q(\d+)$', str(sample_id).strip())
    return int(match.group(1)) if match else None


def _cache_safe_token(value, fallback='na'):
    if not _has_value(value):
        return fallback
    token = str(value).replace('\\', '/').strip()
    token = token.replace('/', '__')
    token = re.sub(r'[^0-9A-Za-z._-]+', '_', token)
    return token or fallback


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


class OVBench(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    SYS = (
        'Carefully watch the video and answer the multiple-choice question.\n'
        'Select the single best option based on the visual evidence.'
    )
    POST_PROMPT = 'Answer with ONLY the single uppercase letter of the correct option.'
    LEGACY_SUBSET_FILES = {
        'ava': 'ava.json',
        'coin': 'coin.json',
        'hirest': 'hirest.json',
        'tao': 'tao.json',
    }

    def __init__(self, dataset='OVBench', nframe=0, fps=-1, frames_limit=4096, check_extracted_frames=True):
        self.frames_limit = frames_limit
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames)

    @classmethod
    def supported_datasets(cls):
        return ['OVBench', 'OVBench_1fps', 'OVBench_2fps', 'OVBench_4fps', 'OVBench_StreamForest']

    @staticmethod
    def _sample_tsv_valid(tsv_path, require_directory_fps=False):
        if not osp.exists(tsv_path):
            return False
        try:
            data = load(tsv_path)
        except Exception:
            return False

        required = {
            'index', 'sample_id', 'subset', 'question', 'video', 'video_id', 'question_idx', 'answer', 'answer_text',
            'answer_type', 'sub_answer_type', 'A', 'B', 'C', 'D', 'E', 'start', 'end', 'video_time',
        }
        if not required.issubset(set(data.columns)):
            return False

        sample = data.head(min(32, len(data)))
        for _, row in sample.iterrows():
            answer = str(row['answer']).strip().upper()
            if answer not in string.ascii_uppercase[:5]:
                return False
            video_id = str(row['video_id']).strip()
            if not video_id or str(row['video']).strip() != video_id:
                return False
            if str(row['subset']).strip() != _infer_subset(video_id):
                return False
            try:
                question_idx = int(row['question_idx'])
            except Exception:
                return False
            if question_idx < 0:
                return False
            try:
                question_start = float(row['start'])
                question_end = float(row['end'])
                video_time = float(row['video_time'])
            except Exception:
                return False
            if not str(row['sample_id']).strip():
                return False
            if not (question_start <= question_end <= video_time):
                return False
            if require_directory_fps:
                video_path = str(row.get('video', '')).strip()
                video_ext = osp.splitext(video_path)[1].lower()
                if video_ext not in VIDEO_EXTENSIONS:
                    try:
                        fps_value = float(row['fps'])
                    except Exception:
                        return False
                    if fps_value <= 0:
                        return False
        return True

    @classmethod
    def _build_tsv_from_json(cls, ann_json, tsv_path):
        if not osp.exists(ann_json):
            raise FileNotFoundError(f'OVBench annotation json not found: {ann_json}')

        records = _load_json_records(ann_json)
        if not records:
            raise ValueError(f'OVBench split annotation json is empty: {ann_json}')
        if not _is_split_annotation_record(records[0]):
            raise ValueError(
                'OVBench now expects the flat split annotation json with fields '
                f'{sorted(SPLIT_RECORD_FIELDS)}. Received incompatible file: {ann_json}'
            )

        rows = []
        per_video_question_counter = defaultdict(int)
        for record in records:
            video_rel_path = str(record.get('video_id', '')).strip()
            if not video_rel_path:
                raise ValueError('OVBench split record is missing `video_id`')
            subset = _infer_subset(video_rel_path)
            if not subset:
                raise ValueError(f'OVBench split record has invalid `video_id`: {video_rel_path!r}')

            sample_id = str(record.get('id', '')).strip()
            question_idx = _parse_question_idx_from_sample_id(sample_id)
            if question_idx is None:
                question_idx = per_video_question_counter[video_rel_path]
            per_video_question_counter[video_rel_path] = max(per_video_question_counter[video_rel_path], question_idx + 1)

            options = record.get('options', [])
            if not isinstance(options, list):
                raise ValueError(f'OVBench split record {sample_id or video_rel_path} has invalid `options`: {type(options)}')
            if len(options) < 2 or len(options) > 5:
                raise ValueError(
                    f'OVBench split record {sample_id or video_rel_path} has {len(options)} options, '
                    'expected between 2 and 5'
                )

            option_map = {ch: '' for ch in 'ABCDE'}
            stripped_options = [_strip_option_prefix(x) for x in options]
            for ch, text in zip('ABCDE', stripped_options):
                option_map[ch] = text

            answer_letter = str(record.get('answer', '')).strip().upper()
            if answer_letter not in option_map or not option_map[answer_letter]:
                raise ValueError(
                    f'OVBench split record {sample_id or video_rel_path} has invalid answer letter: {answer_letter!r}'
                )

            try:
                question_start = float(record.get('start'))
                question_end = float(record.get('end'))
                video_time = float(record.get('video_time'))
            except Exception as err:
                raise ValueError(
                    f'OVBench split record {sample_id or video_rel_path} is missing valid start/end/video_time'
                ) from err

            if not (question_start <= question_end <= video_time):
                raise ValueError(
                    f'OVBench split record {sample_id or video_rel_path} has invalid time range: '
                    f'start={question_start}, end={question_end}, video_time={video_time}'
                )

            rows.append(
                dict(
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
                )
            )

        dump(pd.DataFrame(rows), tsv_path)

    @classmethod
    def _build_tsv_from_legacy_dir(cls, ann_dir, tsv_path):
        raise ValueError(
            'OVBench no longer supports the legacy per-subset annotation directory. '
            'Please use the split question-level annotation json `ovbench_split.json`.'
        )

    @classmethod
    def _build_tsv(cls, tsv_path, ann_json=None, ann_dir=None):
        if ann_json is not None:
            return cls._build_tsv_from_json(ann_json, tsv_path)
        raise ValueError('OVBench now expects the split annotation json `ovbench_split.json`.')

    def prepare_dataset(self, dataset='OVBench', root_dir=None, ann_json=None, ann_dir=None):
        root_dir = root_dir if root_dir is not None else _default_ovbench_root()
        tsv_path = osp.join(root_dir, f'{dataset}.tsv')
        ann_dir = ann_dir if ann_dir is not None else _default_ovbench_anno_dir()
        resolved_ann_json = _resolve_split_annotation_json(ann_json=ann_json, ann_dir=ann_dir, dataset=dataset)

        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f'OVBench root directory not found: {root_dir}. '
                'Please set environment variable OVBENCH_ROOT.'
            )

        require_directory_fps = dataset in {'OVBench_StreamForest', 'OVBench_2fps', 'OVBench_4fps'}
        if not self._sample_tsv_valid(tsv_path, require_directory_fps=require_directory_fps):
            self._build_tsv(tsv_path, ann_json=resolved_ann_json)

        return dict(root=root_dir, data_file=tsv_path)

    @staticmethod
    def _cache_key(subset, video_rel_path, question_idx, sample_id, start, end, mode, source_kind, extra_token=None):
        stem = str(video_rel_path).replace('\\', '/').rstrip('/')
        stem = stem.rsplit('.', 1)[0]
        stem = stem.replace('/', '__')
        question_token = (
            _cache_safe_token(sample_id)
            if _has_value(sample_id)
            else (f'q{int(question_idx):04d}' if _has_value(question_idx) else 'qna')
        )
        cache_key = (
            f'{subset}__{stem}__{source_kind}__qstartendv2__{question_token}__'
            f's{_float_token(start)}__e{_float_token(end)}__{mode}'
        )
        if _has_value(extra_token):
            cache_key = f'{cache_key}__{_cache_safe_token(extra_token)}'
        return cache_key

    @staticmethod
    def _candidate_media_paths(root_dir, subset, video_rel_path):
        rel = str(video_rel_path).replace('\\', '/').strip().lstrip('/')
        candidates = []

        def add(path):
            if path not in candidates:
                candidates.append(path)

        subset = str(subset).strip()
        add(osp.join(root_dir, rel))
        if subset and not rel.startswith(f'{subset}/'):
            add(osp.join(root_dir, subset, rel))

        base, ext = osp.splitext(rel)
        if not ext:
            for suffix in VIDEO_EXTENSIONS:
                add(osp.join(root_dir, rel + suffix))
                if subset and not rel.startswith(f'{subset}/'):
                    add(osp.join(root_dir, subset, rel + suffix))

        return candidates

    def _resolve_media_path(self, line):
        candidates = self._candidate_media_paths(self.data_root, line.get('subset', ''), line['video'])
        for path in candidates:
            if osp.isfile(path):
                return path, 'file'
            if osp.isdir(path):
                return path, 'dir'
        raise FileNotFoundError(
            f'OVBench media not found for subset={line.get("subset", "")}, video={line["video"]}. '
            f'Tried: {candidates}'
        )

    @staticmethod
    def _resolve_effective_window(line):
        if not _has_value(line.get('start')):
            raise ValueError('OVBench sample is missing `start` for question-level sampling.')
        if not _has_value(line.get('end')):
            raise ValueError('OVBench sample is missing `end` for question-level sampling.')

        question_start = float(line['start'])
        question_end = float(line['end'])
        if question_end < question_start:
            raise ValueError(
                f'OVBench sample has invalid time range: start={question_start}, end={question_end}'
            )
        if _has_value(line.get('video_time')):
            video_time = float(line['video_time'])
            if question_end > video_time:
                raise ValueError(
                    'OVBench sample end exceeds `video_time`: '
                    f'start={question_start}, end={question_end}, video_time={video_time}'
                )
        return question_start, question_end

    @staticmethod
    def _resolve_clip(total_frames, source_fps, start_sec, end_sec):
        clip_start = 0
        clip_end = total_frames

        if not source_fps or source_fps <= 0:
            raise ValueError('OVBench question-level sampling requires a positive source fps.')

        clip_start = max(0, int(np.floor(float(start_sec) * source_fps)))
        clip_end = min(total_frames, int(np.ceil(float(end_sec) * source_fps)))

        clip_end = max(clip_start + 1, clip_end)
        clip_end = min(total_frames, clip_end)
        return clip_start, clip_end

    @staticmethod
    def _time_to_frame_index(time_sec, source_fps, total_frames):
        frame_idx = int(np.floor(float(time_sec) * source_fps + 1e-6))
        return min(max(0, frame_idx), total_frames - 1)

    @staticmethod
    def _uniform_positions(total_points, keep_points):
        if total_points <= 0:
            return []
        if keep_points >= total_points:
            return list(range(total_points))
        if keep_points <= 1:
            return [0]

        positions = [0]
        for i in range(1, keep_points - 1):
            pos = int(round(i * (total_points - 1) / (keep_points - 1)))
            min_pos = positions[-1] + 1
            max_pos = (total_points - 1) - (keep_points - 1 - i)
            pos = min(max(pos, min_pos), max_pos)
            positions.append(pos)
        positions.append(total_points - 1)
        return positions

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

    def _frame_metadata_path(self, cache_key):
        frame_root = osp.join(self.frame_root, cache_key)
        os.makedirs(frame_root, exist_ok=True)
        return osp.join(frame_root, 'sample.json')

    @staticmethod
    def _infer_video_read_type_from_source(source_kind, media_path):
        if source_kind == 'dir':
            return 'img'
        suffix = osp.splitext(str(media_path))[1].lower()
        if suffix == '.gif':
            return 'gif'
        if suffix == '.avi':
            return 'av'
        return 'decord'

    @staticmethod
    def _cached_frame_paths_from_info(metadata_path, video_info):
        frame_root = osp.dirname(metadata_path)
        output_frames = video_info.get('output_frames', [])
        frame_paths = [osp.join(frame_root, str(item.get('output_name', '')).strip()) for item in output_frames]
        if not frame_paths or not np.all([pth and osp.exists(pth) for pth in frame_paths]):
            return None
        return frame_paths

    @staticmethod
    def _sample_output_name(position, total, source_kind, original_ref):
        if source_kind == 'dir':
            return f'frame-{position:04d}-of-{total:04d}-orig-name-{_cache_safe_token(original_ref)}.jpg'
        return f'frame-{position:04d}-of-{total:04d}-orig-idx-{int(original_ref):010d}.jpg'

    def _build_output_frames(self, media_path, source_kind, frame_indices, sample_times_sec):
        frame_root_items = None
        if source_kind == 'dir':
            frame_root_items = _list_frame_files(media_path)

        total = len(frame_indices)
        output_frames = []
        for position, (frame_idx, time_sec) in enumerate(zip(frame_indices, sample_times_sec), start=1):
            item = {
                'position': int(position),
                'total_frames': int(total),
                'time_sec': float(time_sec),
                'orig_index': int(frame_idx),
            }
            if source_kind == 'dir':
                if frame_root_items is None or frame_idx < 0 or frame_idx >= len(frame_root_items):
                    raise IndexError(
                        f'OVBench sampled frame index {frame_idx} is out of range for directory-backed media: {media_path}'
                    )
                original_name = osp.basename(frame_root_items[frame_idx])
                original_stem = osp.splitext(original_name)[0]
                item['orig_name'] = original_name
                item['orig_stem'] = original_stem
                output_name = self._sample_output_name(position, total, source_kind, original_stem)
            else:
                output_name = self._sample_output_name(position, total, source_kind, frame_idx)
            item['output_name'] = output_name
            output_frames.append(item)
        return output_frames

    @staticmethod
    def _save_frames_from_arrays(frames, frame_paths):
        for frame, out_path in zip(frames, frame_paths):
            if osp.exists(out_path):
                continue
            Image.fromarray(frame).save(out_path)

    @staticmethod
    def _load_cached_video_info(metadata_path):
        if not osp.exists(metadata_path):
            return None
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                video_info = json.load(f)
        except Exception:
            return None
        required = {
            'metadata_version', 'source_fps', 'effective_start_sec', 'effective_end_sec', 'clip_duration',
            'clip_frames', 'sample_fps', 'sample_n_frame', 'sample_times_sec', 'output_frames',
        }
        if not required.issubset(set(video_info.keys())):
            return None
        return video_info

    @staticmethod
    def _dump_video_info(metadata_path, video_info):
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(video_info, f, ensure_ascii=False, indent=2)
            f.write('\n')

    def _use_streamforest_sampling(self):
        return self.dataset_name in {'OVBench_2fps', 'OVBench_4fps'}

    def _use_streamforest_style_prompt(self):
        return self.dataset_name in {'OVBench_2fps', 'OVBench_4fps', 'OVBench_StreamForest'}

    def _max_frames_num(self):
        value = os.environ.get('OVBENCH_MAX_FRAMES_NUM', '').strip()
        if value:
            return max(1, int(value))
        return max(1, int(self.frames_limit))

    def _save_video_frames_streamforest_style(self, line):
        media_path, source_kind = self._resolve_media_path(line)
        effective_start_sec, effective_end_sec = self._resolve_effective_window(line)
        max_frames = self._max_frames_num()
        mode = f'{self.fps}fps'
        cache_key = self._cache_key(
            line.get('subset', ''),
            line['video'],
            line.get('question_idx'),
            line.get('sample_id'),
            effective_start_sec,
            effective_end_sec,
            mode,
            source_kind,
            extra_token=f'max{max_frames}-samplev2',
        )
        metadata_path = self._frame_metadata_path(cache_key)

        cached_info = self._load_cached_video_info(metadata_path)
        if cached_info is not None:
            frame_paths = self._cached_frame_paths_from_info(metadata_path, cached_info)
            if frame_paths is not None:
                return frame_paths, cached_info

        reader_kwargs = dict(
            video_path=media_path,
            num_frames=max_frames,
            sample='middle',
            min_num_frames=4,
            max_num_frames=max_frames,
            clip=[effective_start_sec, effective_end_sec],
            local_num_frames=1,
            dynamic_fps=float(self.fps),
        )
        if source_kind == 'dir' and _has_value(line.get('fps')):
            reader_kwargs['fps'] = float(line['fps'])

        frames, frame_indices, source_fps, reader_duration = read_frames_auto(**reader_kwargs)
        frame_indices = [int(idx) for idx in frame_indices]
        sample_times_sec = [float(idx) / float(source_fps) for idx in frame_indices]
        clip_duration = max(0.0, float(effective_end_sec) - float(effective_start_sec))
        if clip_duration > 0 and len(frame_indices) > 1:
            sample_fps = round((len(frame_indices) - 1) / clip_duration, 6)
        else:
            sample_fps = float(self.fps if self.fps > 0 else source_fps)

        output_frames = self._build_output_frames(media_path, source_kind, frame_indices, sample_times_sec)
        frame_root = osp.dirname(metadata_path)
        frame_paths = [osp.join(frame_root, item['output_name']) for item in output_frames]
        if not np.all([osp.exists(p) for p in frame_paths]):
            self._save_frames_from_arrays(frames, frame_paths)

        clip_start = min(frame_indices)
        clip_end = max(frame_indices) + 1
        clip_frames = max(1, clip_end - clip_start)
        video_info = {
            'metadata_version': 'ovbench_qwen_streamforest_sample_v2',
            'dataset_name': str(self.dataset_name),
            'sample_id': str(line.get('sample_id', '')).strip(),
            'video_id': str(line.get('video_id', line.get('video', ''))).strip(),
            'media_path': str(media_path),
            'source_kind': source_kind,
            'video_read_type': self._infer_video_read_type_from_source(source_kind, media_path),
            'fps': float(source_fps),
            'source_fps': float(source_fps),
            'n_frames': None,
            'duration': float(reader_duration),
            'clip_start': int(clip_start),
            'clip_end': int(clip_end),
            'effective_start_sec': float(effective_start_sec),
            'effective_end_sec': float(effective_end_sec),
            'clip_duration': float(clip_duration),
            'clip_frames': int(clip_frames),
            'sample_fps': float(sample_fps),
            'sample_n_frame': len(frame_indices),
            'sample_times_sec': sample_times_sec,
            'frame_indices': frame_indices,
            'max_frames_num': int(max_frames),
            'dynamic_fps': float(self.fps),
            'min_num_frames': 4,
            'max_num_frames': int(max_frames),
            'output_frames': output_frames,
        }
        self._dump_video_info(metadata_path, video_info)
        return frame_paths, video_info

    def _sample_indices(self, start_sec, end_sec, source_fps, total_frames):
        source_fps = float(source_fps) if source_fps and source_fps > 0 else float(self.fps if self.fps > 0 else 1.0)
        clip_duration = max(0.0, float(end_sec) - float(start_sec))

        if self.nframe > 0 and self.fps <= 0:
            if self.nframe == 1:
                sample_times = [float(end_sec)]
            else:
                sample_times = np.linspace(float(start_sec), float(end_sec), self.nframe).tolist()
            indices = [self._time_to_frame_index(t, source_fps, total_frames) for t in sample_times]
            sample_fps = ((len(sample_times) - 1) / clip_duration) if clip_duration > 0 and len(sample_times) > 1 else source_fps
        elif self.fps > 0:
            step_sec = 1.0 / self.fps
            n_steps = max(0, int(np.floor(clip_duration * self.fps + 1e-9)))
            sample_times = [float(start_sec) + i * step_sec for i in range(n_steps + 1)]
            if not sample_times:
                sample_times = [float(start_sec)]
            if not np.isclose(sample_times[-1], float(end_sec)):
                sample_times.append(float(end_sec))

            truncated = len(sample_times) > self.frames_limit
            if truncated:
                positions = self._uniform_positions(len(sample_times), self.frames_limit)
                sample_times = [sample_times[pos] for pos in positions]

            indices = [self._time_to_frame_index(t, source_fps, total_frames) for t in sample_times]
            if truncated:
                sample_fps = ((len(sample_times) - 1) / clip_duration) if clip_duration > 0 and len(sample_times) > 1 else self.fps
            else:
                sample_fps = self.fps
        else:
            raise ValueError('nframe and fps should be set with at least one valid value')

        clip_start = min(indices)
        clip_end = max(indices) + 1
        clip_frames = max(1, clip_end - clip_start)
        actual_sample_times = [float(idx) / source_fps for idx in indices]
        return indices, clip_duration, sample_fps, clip_frames, clip_start, clip_end, actual_sample_times

    @staticmethod
    def _save_frames_with_decord(video_path, indices, frame_paths):
        import decord
        from decord import cpu

        vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=1)
        images = [Image.fromarray(vr[i].asnumpy()) for i in indices]
        for im, pth in zip(images, frame_paths):
            if not osp.exists(pth):
                im.save(pth)

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
                    if not osp.exists(pth):
                        image.save(pth)
                seen.add(frame_idx)

        missing = [pth for pth in frame_paths if not osp.exists(pth)]
        if missing:
            raise RuntimeError(f'Failed to decode all requested OVBench frames from {video_path}')

    @staticmethod
    def _save_frames_from_dir(frame_files, indices, frame_paths):
        for frame_idx, out_path in zip(indices, frame_paths):
            if osp.exists(out_path):
                continue
            image = Image.open(frame_files[frame_idx]).convert('RGB')
            image.save(out_path)

    def save_video_frames(self, line, video_llm=False, verbose=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self._use_streamforest_sampling():
            return self._save_video_frames_streamforest_style(line)

        media_path, source_kind = self._resolve_media_path(line)
        effective_start_sec, effective_end_sec = self._resolve_effective_window(line)
        mode = f'{self.nframe}frame' if self.nframe > 0 else f'{self.fps}fps'
        cache_key = self._cache_key(
            line.get('subset', ''),
            line['video'],
            line.get('question_idx'),
            line.get('sample_id'),
            effective_start_sec,
            effective_end_sec,
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

            indices, clip_duration, sample_fps, clip_frames, clip_start, clip_end, sample_times_sec = self._sample_indices(
                effective_start_sec, effective_end_sec, source_fps, total_frames
            )
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
                raise RuntimeError(f'OVBench frame directory is empty: {media_path}')

            source_fps = float(line['fps']) if _has_value(line.get('fps')) else None
            total_frames = len(frame_files)
            if (not source_fps or source_fps <= 0) and _has_value(line.get('video_time')):
                video_time = float(line['video_time'])
                if video_time > 0:
                    source_fps = total_frames / video_time
            if not source_fps or source_fps <= 0:
                raise ValueError(
                    'OVBench frame directory sample requires a positive fps or valid `video_time` for '
                    f'question-level sampling: {line["video"]}'
                )
            indices, clip_duration, sample_fps, clip_frames, clip_start, clip_end, sample_times_sec = self._sample_indices(
                effective_start_sec, effective_end_sec, source_fps, total_frames
            )
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
            'effective_start_sec': effective_start_sec,
            'effective_end_sec': effective_end_sec,
            'clip_duration': clip_duration,
            'clip_frames': clip_frames,
            'sample_fps': sample_fps,
            'sample_n_frame': len(indices),
            'sample_times_sec': sample_times_sec,
        }
        return frame_paths, video_info

    @staticmethod
    def _build_streamforest_time_msg(video_info):
        sample_times_sec = video_info.get('sample_times_sec') if video_info is not None else None
        if not sample_times_sec:
            return ''
        first_sec = float(sample_times_sec[0])
        last_sec = float(sample_times_sec[-1])
        span = max(0.0, last_sec - first_sec)
        return (
            f'The video contains {len(sample_times_sec)} frames sampled from the past {span:.1f} seconds ago '
            f'({first_sec:.1f}s of the entire video) up to the present moment ({last_sec:.1f}s of the entire video).'
        )

    def _build_question_block(self, line, video_info=None, include_segment=True, time_msg=None):
        segments = []
        start_value = video_info.get('effective_start_sec') if video_info is not None else line.get('start')
        end_value = video_info.get('effective_end_sec') if video_info is not None else line.get('end')
        start_text = _format_seconds(start_value)
        end_text = _format_seconds(end_value)
        if include_segment and start_text is not None and end_text is not None:
            segments.append(f'Video segment: {start_text} to {end_text}')
        if time_msg:
            segments.append(time_msg)
        segments.append(f"Question:\n{line['question']}")
        segments.append('Options:')
        for ch in 'ABCDE':
            value = line.get(ch, '')
            if _has_value(value) and str(value).strip():
                segments.append(f'{ch}. {value}')
        return '\n'.join(segments)

    @staticmethod
    def _build_streamforest_native_question_block(line):
        segments = [str(line['question'])]
        for ch in 'ABCDE':
            value = line.get(ch, '')
            if _has_value(value) and str(value).strip():
                segments.append(f'{ch}. {value}')
        return '\n'.join(segments)

    def _build_streamforest_style_question_block(self, line, video_info=None):
        segments = []
        time_msg = self._build_streamforest_time_msg(video_info)
        if time_msg:
            segments.append(time_msg)
        segments.append(self._build_streamforest_native_question_block(line))
        return '\n'.join(segments)

    def _build_streamforest_native_video_item(self, line):
        media_path, source_kind = self._resolve_media_path(line)
        effective_start_sec, effective_end_sec = self._resolve_effective_window(line)
        video_item = dict(
            type='video',
            value=media_path,
            start=effective_start_sec,
            end=effective_end_sec,
            sample_id=str(line.get('sample_id', '')).strip(),
            video_id=str(line.get('video_id', line.get('video', ''))).strip(),
            question_idx=line.get('question_idx'),
            index=line.get('index'),
        )
        if source_kind == 'dir':
            video_item['video_read_type'] = 'img'
            if _has_value(line.get('fps')):
                video_item['fps'] = float(line['fps'])
            return video_item
        if source_kind == 'file':
            suffix = osp.splitext(str(media_path))[1].lower()
            if suffix == '.gif':
                video_item['video_read_type'] = 'gif'
            else:
                video_item['video_read_type'] = 'decord'
            return video_item
        raise ValueError(
            f'OVBench_StreamForest expected a file or directory media source, got '
            f'source_kind={source_kind!r} for media_path={media_path}'
        )

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        use_system_prompt = not (
            self._use_streamforest_style_prompt()
            or (video_llm and self.dataset_name == 'OVBench_1fps')
        )
        message = [dict(type='text', value=self.SYS, role='system')] if use_system_prompt else []
        if self.dataset_name == 'OVBench_StreamForest' and video_llm:
            message.append(self._build_streamforest_native_video_item(line))
            question_block = self._build_streamforest_native_question_block(line)
        else:
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

            if self.dataset_name in {'OVBench_2fps', 'OVBench_4fps'}:
                question_block = self._build_streamforest_style_question_block(line, video_info=video_info)
            elif self.dataset_name == 'OVBench_1fps' and video_llm:
                question_block = self._build_question_block(
                    line,
                    video_info=video_info,
                    include_segment=False,
                    time_msg=self._build_streamforest_time_msg(video_info),
                )
            else:
                question_block = self._build_question_block(line, video_info=video_info)

        if self._use_streamforest_style_prompt():
            message.append(
                dict(
                    type='text',
                    value=f'{question_block}\n\nAnswer with the option\'s letter from the given choices directly.',
                )
            )
        else:
            message.append(dict(type='text', value=question_block))
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
        judge_name = judge_kwargs.get('model', 'exact_matching')
        if judge_name not in [None, 'exact_matching']:
            warnings.warn(f'OVBench uses rule-based evaluation. Ignoring judge model `{judge_name}`.')

        score_file = get_intermediate_file_path(eval_file, '_score')
        acc_file = get_intermediate_file_path(eval_file, '_acc', 'json')

        data = load(eval_file)
        scored = data.copy()
        letters = 'ABCDE'
        parsed_options = []
        hits = []

        for _, row in scored.iterrows():
            option_map = {
                ch: str(row.get(ch, '')).strip()
                for ch in letters
                if _has_value(row.get(ch)) and str(row.get(ch, '')).strip()
            }
            pred_letter = _extract_option_letter(row.get('prediction', ''), valid_letters=''.join(option_map.keys()))
            if not pred_letter and option_map:
                pred_letter = _match_option_text(row.get('prediction', ''), option_map)

            gt = str(row.get('answer', '')).strip().upper()
            parsed_options.append(pred_letter)
            hits.append(int(bool(pred_letter) and pred_letter == gt))

        scored['parsed_option'] = parsed_options
        scored['hit'] = hits
        dump(scored, score_file)

        result = {
            'overall_accuracy': float(np.mean(scored['hit'])) if len(scored) else 0.0,
            'answer_type_accuracy': self._group_acc(scored, 'answer_type'),
            'sub_answer_type_accuracy': self._group_acc(scored, 'sub_answer_type'),
        }
        dump(result, acc_file)
        return result
