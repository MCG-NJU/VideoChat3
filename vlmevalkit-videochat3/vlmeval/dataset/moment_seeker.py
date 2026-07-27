import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset
from .utils.video_pyav import ffprobe_video_info, get_video_decode_backend, save_frames_by_indices_pyav
from .vidi_vue_tr import _extract_spans_from_prediction, _overlap_ratio
from .utils.vtg_reason import REASON_PROMPT_SUFFIX, extract_answer_from_tags, wrap_vtg_prompt_with_reason
from .video_meta_cache import (
    ensure_video_meta_json,
    load_video_meta,
    sample_fps_from_meta,
    sample_frame_count_from_meta,
)


@dataclass(frozen=True)
class _MomentSeekerPaths:
    root: str
    videos_dir: str
    gt_json: str
    tsv_path: str


def _default_moment_seeker_root() -> str:
    return os.environ.get(
        'MOMENT_SEEKER_ROOT',
        '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/MomentSeeker',
    )


def _video_stem_from_src_path(src_video_path: str) -> str:
    base = os.path.basename(str(src_video_path))
    if base.lower().endswith('.mp4'):
        return base[:-4]
    return base


def _index_video_files(videos_dir: str) -> Dict[str, str]:
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f'[MomentSeeker] videos_dir not found: {videos_dir}')
    files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    mp: Dict[str, str] = {}
    for f in files:
        prefix = f.split('.')[0]
        if prefix not in mp or len(f) < len(mp[prefix]) or (len(f) == len(mp[prefix]) and f < mp[prefix]):
            mp[prefix] = f
    return mp


def _ensure_moment_seeker_tsv(paths: _MomentSeekerPaths) -> None:
    if os.path.exists(paths.tsv_path):
        try:
            df = load(paths.tsv_path)
            if 'question' in df.columns and 'video' in df.columns:
                try:
                    video_map = _index_video_files(paths.videos_dir)
                    uniq_vids = set(str(x) for x in df['video_id'].dropna().tolist())
                    missing = [v for v in uniq_vids if v not in video_map]
                    if len(missing) == 0:
                        return
                except Exception:
                    pass
        except Exception:
            pass

    if not os.path.exists(paths.gt_json):
        raise FileNotFoundError(f'[MomentSeeker] annotation not found: {paths.gt_json}')

    gts = json.load(open(paths.gt_json, 'r', encoding='utf-8'))
    if not isinstance(gts, list):
        raise ValueError(f'[MomentSeeker] annotation must be a list, got {type(gts)}')

    video_map = _index_video_files(paths.videos_dir)
    duration_cache: Dict[str, float] = {}

    def get_duration(stem: str) -> float:
        if stem in duration_cache:
            return duration_cache[stem]
        fname = video_map.get(stem)
        if fname is None:
            duration_cache[stem] = 0.0
            return 0.0
        try:
            _, _, duration = ffprobe_video_info(os.path.join(paths.videos_dir, fname))
            duration_cache[stem] = float(duration)
        except Exception:
            duration_cache[stem] = 0.0
        return duration_cache[stem]

    rows = []
    for i, item in enumerate(gts):
        stem = _video_stem_from_src_path(item['src_video_path'])
        if stem not in video_map:
            continue
        intervals = item.get('answering_time_interval') or []
        if not intervals:
            continue
        qry = str(item.get('qry_text', ''))
        rows.append(
            dict(
                index=0,
                video=stem,
                video_id=stem,
                question=qry,
                query=qry,
                duration=get_duration(stem),
                gt=json.dumps(intervals),
                task=str(item.get('task', '')),
                src_video_path=str(item.get('src_video_path', '')),
            )
        )

    df = pd.DataFrame(rows).reset_index(drop=True)
    df['index'] = np.arange(len(df), dtype=int)
    os.makedirs(os.path.dirname(paths.tsv_path), exist_ok=True)
    dump(df, paths.tsv_path)


class MomentSeeker(VideoBaseDataset):
    """
    MomentSeeker: text-to-video temporal grounding (t2v.json + local mp4 under videos/).
    Metrics: mIoU (mean multi-span IoU, same as VUE_TR `_overlap_ratio`) and R1@θ
    (fraction of samples with IoU >= θ). Default θ list: [0.3, 0.5, 0.7], overridable via
    `judge_kwargs['r1_thresholds']`.
    """

    TYPE = 'Video-VTG'
    MODALITY = 'VIDEO'

    PROMPT_VIDEO = (
        "You are given a video.\n"
        "Task: temporal grounding.\n"
        "Given the query: \"{query}\", return ALL time spans (in seconds) where the query is grounded in the video.\n"
        "Output format MUST be a JSON array of [start, end] pairs, e.g. [[0, 3.5], [10, 12]].\n"
    )

    PROMPT_FRAMES_PREFIX = (
        "You are given a video as a sequence of frames.\n"
        "The timestamp (in seconds) is shown before each frame.\n"
        "Task: temporal grounding.\n"
    )

    def __init__(
        self,
        dataset='MomentSeeker',
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
        reason=False,
    ):
        self.reason = reason
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )
        self._video_file_cache: Dict[str, str] = {}
        self._video_meta = load_video_meta(getattr(self, '_video_meta_path', None))
        self._video_duration_cache: Dict[str, float] = {}

        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos

    @classmethod
    def supported_datasets(cls):
        return ['MomentSeeker']

    def prepare_dataset(self, dataset):
        root = _default_moment_seeker_root()
        videos_dir = os.path.join(root, 'videos')
        paths = _MomentSeekerPaths(
            root=root,
            videos_dir=videos_dir,
            gt_json=os.path.join(root, 't2v.json'),
            tsv_path=os.path.join(root, 'MomentSeeker.tsv'),
        )
        meta_path = paths.tsv_path.replace('.tsv', '_video_meta.json')
        self._video_meta_path = meta_path
        _ensure_moment_seeker_tsv(paths)
        try:
            df = load(paths.tsv_path)
            video_ids = sorted(set(str(x) for x in df['video_id'].dropna().tolist()))
            video_map = _index_video_files(paths.videos_dir)
            ensure_video_meta_json(
                meta_path,
                video_ids,
                lambda video_id: os.path.join(paths.videos_dir, video_map[str(video_id)]),
                prefer_pyav=get_video_decode_backend() == 'pyav',
            )
        except Exception as e:
            warnings.warn(f'[MomentSeeker] failed to ensure video meta json: {e}')
        self._videos_dir = paths.videos_dir
        return dict(root=paths.videos_dir, data_file=paths.tsv_path)

    def _resolve_video_file(self, video_id: str) -> str:
        if video_id in self._video_file_cache:
            return self._video_file_cache[video_id]
        p1 = os.path.join(self._videos_dir, f'{video_id}.mp4')
        if os.path.exists(p1):
            self._video_file_cache[video_id] = p1
            return p1
        cands = [x for x in os.listdir(self._videos_dir) if x.startswith(video_id) and x.endswith('.mp4')]
        if len(cands) == 1:
            p = os.path.join(self._videos_dir, cands[0])
            self._video_file_cache[video_id] = p
            return p
        if len(cands) > 1:
            cands.sort(key=lambda x: (len(x), x))
            p = os.path.join(self._videos_dir, cands[0])
            self._video_file_cache[video_id] = p
            return p
        raise FileNotFoundError(
            f'[MomentSeeker] video file not found for video_id={video_id} under {self._videos_dir}'
        )

    def _video_duration(self, video_id: str, fallback: float | None = None) -> float:
        if video_id in self._video_duration_cache:
            return self._video_duration_cache[video_id]
        meta = getattr(self, '_video_meta', {}).get(str(video_id), {})
        if meta:
            try:
                duration = float(meta.get('duration', 0.0))
                if duration > 0:
                    self._video_duration_cache[video_id] = duration
                    return duration
            except Exception:
                pass
        duration = float(fallback or 0.0)
        vid_path = self._resolve_video_file(video_id)
        backend = get_video_decode_backend()
        try:
            if backend != 'pyav':
                import decord

                vr = decord.VideoReader(vid_path, num_threads=1)
                total = int(len(vr))
                video_fps = float(vr.get_avg_fps())
                duration = float(total / video_fps) if video_fps > 0 else duration
            else:
                raise RuntimeError('force pyav')
        except Exception:
            try:
                _, _, probed_duration = ffprobe_video_info(vid_path)
                duration = float(probed_duration)
            except Exception:
                pass
        if duration > 0:
            self._video_duration_cache[video_id] = duration
        return duration

    def _sample_indices_and_timestamps(self, vid_path: str):
        backend = get_video_decode_backend()
        use_pyav = backend == 'pyav'

        vr = None
        try:
            if backend != 'pyav':
                import decord

                vr = decord.VideoReader(vid_path, num_threads=1)
                total = int(len(vr))
                video_fps = float(vr.get_avg_fps())
                duration = float(total / video_fps) if video_fps > 0 else 0.0
            else:
                raise RuntimeError('force pyav')
        except Exception:
            total, video_fps, duration = ffprobe_video_info(vid_path)
            use_pyav = True

        video_info = {
            'fps': float(video_fps),
            'n_frames': int(total),
            'duration': float(duration),
            'backend': 'pyav' if use_pyav else 'decord',
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = total / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            indices = [min(max(0, x), total - 1) for x in indices]
        elif self.fps > 0:
            required_frames = int(duration * self.fps) if duration > 0 else 0
            if required_frames > self.frames_limit:
                step_size = total / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                if backend != 'decord':
                    video_info['backend'] = 'pyav'
            else:
                step_size = video_fps / self.fps
                indices = [int(i * step_size) for i in range(required_frames)]
            indices = [min(max(0, x), total - 1) for x in indices]
        else:
            raise ValueError('fps and nframe should be set at least one valid value')
        timestamps = [idx / video_fps for idx in indices]
        return vr, indices, timestamps, video_info

    def _format_video_prompt(self, query: str) -> str:
        prompt = self.PROMPT_VIDEO.format(query=query)
        if self.reason:
            prompt = wrap_vtg_prompt_with_reason(prompt)
        return prompt

    def _video_llm_cached_prompt(self, video_id: str, query: str, line):
        meta = getattr(self, '_video_meta', {}).get(str(video_id), {})
        if not meta or self.fps <= 0:
            return None
        try:
            total = sample_frame_count_from_meta(meta, self.fps, self.nframe, self.frames_limit)
        except Exception:
            return None
        if total <= 0:
            return None
        frame_paths = self.frame_paths_fps(video_id, total)

        return [
            dict(
                type='video',
                value=frame_paths,
                sample_fps=sample_fps_from_meta(meta, len(frame_paths), self.fps, self.frames_limit),
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            ),
            dict(type='text', value=self._format_video_prompt(query)),
        ]

    def save_video_frames(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        video_id = str(line['video_id'])
        vid_path = self._resolve_video_file(video_id)

        vr, indices, _, video_info = self._sample_indices_and_timestamps(vid_path)

        if self.fps > 0:
            frame_paths = self.frame_paths_fps(video_id, len(indices))
        else:
            frame_paths = self.frame_paths(video_id)

        need_extract = self.check_extracted_frames and (not np.all([osp.exists(p) for p in frame_paths]))
        if need_extract:
            print(f'video ({video_id}) is not extracted')
            if video_info.get('backend') == 'pyav' or vr is None:
                save_frames_by_indices_pyav(
                    vid_path=vid_path,
                    indices=[int(x) for x in indices],
                    frame_paths=frame_paths,
                    total_frames=video_info.get('n_frames', None),
                    desc=f'Extracting (pyav): {video_id}',
                )
            else:
                try:
                    for frame_idx, pth in zip(indices, frame_paths):
                        if osp.exists(pth):
                            continue
                        arr = vr[int(frame_idx)].asnumpy()
                        Image.fromarray(arr).save(pth)
                except Exception:
                    save_frames_by_indices_pyav(
                        vid_path=vid_path,
                        indices=[int(x) for x in indices],
                        frame_paths=frame_paths,
                        total_frames=video_info.get('n_frames', None),
                        desc=f'Extracting (pyav): {video_id}',
                    )
        return frame_paths

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        query = str(line['query'])
        video_id = str(line['video_id'])

        if video_llm and not self.check_extracted_frames:
            cached = self._video_llm_cached_prompt(video_id, query, line)
            if cached is not None:
                return cached

        vid_path = self._resolve_video_file(video_id)
        vr, indices, timestamps, video_info = self._sample_indices_and_timestamps(vid_path)
        duration = float(video_info.get('duration', 0.0))

        if self.fps > 0:
            frame_paths = self.frame_paths_fps(video_id, len(indices))
        else:
            frame_paths = self.frame_paths(video_id)

        need_extract = self.check_extracted_frames and (not np.all([osp.exists(p) for p in frame_paths]))
        if need_extract:
            if video_info.get('backend') == 'pyav' or vr is None:
                save_frames_by_indices_pyav(
                    vid_path=vid_path,
                    indices=[int(x) for x in indices],
                    frame_paths=frame_paths,
                    total_frames=video_info.get('n_frames', None),
                    desc=f'Extracting (pyav): {video_id}',
                )
            else:
                try:
                    for frame_idx, pth in zip(indices, frame_paths):
                        if osp.exists(pth):
                            continue
                        arr = vr[int(frame_idx)].asnumpy()
                        Image.fromarray(arr).save(pth)
                except Exception:
                    save_frames_by_indices_pyav(
                        vid_path=vid_path,
                        indices=[int(x) for x in indices],
                        frame_paths=frame_paths,
                        total_frames=video_info.get('n_frames', None),
                        desc=f'Extracting (pyav): {video_id}',
                    )

        if video_llm:
            assert self.fps > 0
            actual_fps = (
                (self.frames_limit / duration) if (len(frame_paths) == self.frames_limit and duration > 0) else self.fps
            )
            return [
                dict(
                    type='video',
                    value=frame_paths,
                    sample_fps=actual_fps,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    total_pixels=self.total_pixels,
                ),
                dict(type='text', value=self._format_video_prompt(query)),
            ]

        msg = [dict(type='text', value=self.PROMPT_FRAMES_PREFIX)]
        for t, p in zip(timestamps, frame_paths):
            msg.append(dict(type='text', value=f'{t:.2f}s'))
            msg.append(dict(type='image', value=p))
        msg.append(dict(type='text', value=f'Query: "{query}"\n'))
        if self.reason:
            msg.append(dict(type='text', value=REASON_PROMPT_SUFFIX.lstrip('\n')))
        else:
            msg.append(dict(type='text', value='Output ONLY JSON array of [start, end] pairs.'))
        return msg

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Writes a JSON score file with:
          - `overall`: qa_count, num (same as qa_count), tIoU, and R1@θ for each threshold.
          - `by_task`: per `task` from annotations — qa_count, tIoU, R1@θ.
        tIoU is the mean multi-span IoU (`_overlap_ratio`, same as VUE_TR).
        R1@θ is the fraction of samples with tIoU >= θ.
        Optional `judge_kwargs['r1_thresholds']` (list of floats), default [0.3, 0.5, 0.7].
        Optional LLM parsing (same pattern as VUE_TR) via judge_kwargs['model'].
        If the eval file has no `task` column, tasks are merged from MomentSeeker.tsv under
        MOMENT_SEEKER_ROOT (mount videogpu to access).
        """
        r1_thresholds: List[float] = judge_kwargs.get('r1_thresholds', [0.3, 0.5, 0.7])
        if not isinstance(r1_thresholds, (list, tuple)) or not r1_thresholds:
            r1_thresholds = [0.3, 0.5, 0.7]

        model_name = judge_kwargs.get('model', 'exact_matching')
        force_llm = bool(judge_kwargs.get('force_llm_judge', False))
        parse_on_fail_only = bool(judge_kwargs.get('llm_parse_on_fail_only', True))

        judge_model = None
        if model_name not in [None, 'exact_matching']:
            if gpt_key_set():
                try:
                    from .utils import build_judge, DEBUG_MESSAGE

                    judge_model = build_judge(**judge_kwargs)
                    if hasattr(judge_model, 'working') and (not judge_model.working()):
                        warnings.warn(
                            '[MomentSeeker] Judge model is not working properly, will use rule-based parsing only.'
                        )
                        warnings.warn(DEBUG_MESSAGE)
                        judge_model = None
                except Exception as e:
                    warnings.warn(
                        f'[MomentSeeker] Failed to build judge model ({model_name}), rule-based only: {type(e)}: {e}'
                    )
                    judge_model = None
            else:
                warnings.warn('[MomentSeeker] API key is not set, will use rule-based parsing only.')
                judge_model = None

        if judge_model is None:
            force_llm = False
            parse_on_fail_only = True

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )
        safe_model_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(model_name))
        suffix = f'_{safe_model_name}_score' if model_name not in [None, 'exact_matching'] else '_score'
        score_file = get_intermediate_file_path(eval_file, suffix, 'json')
        judge_cache_file = get_intermediate_file_path(eval_file, f'_{safe_model_name}_judge_cache', 'pkl')

        if not os.path.exists(score_file):
            df = load(eval_file)
            if 'prediction' not in df.columns:
                raise ValueError('[MomentSeeker] eval file must contain `prediction`.')
            if 'gt' not in df.columns:
                raise ValueError('[MomentSeeker] eval file must contain `gt`.')

            df = df.copy()
            if 'task' not in df.columns:
                warnings.warn(
                    '[MomentSeeker] eval file has no `task` column; merging from MomentSeeker.tsv '
                    '(mount: bash /mnt/shared-storage-user/zhuyuhan/mount_anything.sh).'
                )
                tsv_path = os.path.join(_default_moment_seeker_root(), 'MomentSeeker.tsv')
                if os.path.isfile(tsv_path):
                    ref = load(tsv_path)
                    if 'task' in ref.columns and 'index' in ref.columns and 'index' in df.columns:
                        idx2task = ref.set_index('index')['task'].astype(str)
                        df['task'] = df['index'].map(idx2task).fillna('').astype(str)
                    else:
                        df['task'] = ''
                else:
                    warnings.warn(
                        f'[MomentSeeker] cannot merge task: {tsv_path} not found; '
                        'by_task breakdown will be under a single empty task key.'
                    )
                    df['task'] = ''
            else:
                df['task'] = df['task'].fillna('').astype(str)

            judge_cache = {}
            if judge_model is not None and os.path.exists(judge_cache_file):
                try:
                    judge_cache = load(judge_cache_file)
                except Exception:
                    judge_cache = {}
            if not isinstance(judge_cache, dict):
                judge_cache = {}

            def _normalize_cache_key(row, fallback_i: int):
                try:
                    if 'index' in row and (not pd.isna(row.get('index'))):
                        return int(row.get('index'))
                except Exception:
                    pass
                return int(fallback_i)

            def _llm_extract_spans(pred_text: str, index=None):
                if judge_model is None:
                    return [], False, False
                if index is not None and index in judge_cache:
                    return judge_cache[index], False, False

                prompt = (
                    'You are a parser. Extract ALL relevant time spans from the model output.\n'
                    'Output MUST be a JSON array of [start, end] pairs (numbers, seconds), e.g. [[0, 3.5], [10, 12]].\n'
                    'Do not output any extra text. If none, output: []\n'
                    f'Model output:\n{pred_text}\n'
                )
                resp = judge_model.generate(prompt)
                res_spans = []
                try:
                    js = list(extract_json_objects(resp))
                    if js:
                        for obj in reversed(js):
                            if isinstance(obj, list):
                                spans = []
                                for it in obj:
                                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                                        spans.append([float(it[0]), float(it[1])])
                                res_spans = spans
                                break
                except Exception:
                    pass

                if index is not None:
                    judge_cache[index] = res_spans
                return res_spans, True, True

            records: List[dict] = []
            used_llm = 0
            new_cache_entries = 0
            flush_every = 50

            for i, row in tqdm(df.iterrows(), total=len(df), desc='Evaluating MomentSeeker'):
                idx = _normalize_cache_key(row, i)
                gt = json.loads(row['gt']) if isinstance(row['gt'], str) else row['gt']
                gt_arr = np.array(gt, dtype=float) if gt else np.array([])

                pred_text = row['prediction']
                if self.reason:
                    pred_text = extract_answer_from_tags(pred_text)
                pred_spans: List[List[float]] = []

                if judge_model is not None and (force_llm or (not parse_on_fail_only)):
                    llm_spans, did_call, did_update = _llm_extract_spans(str(pred_text), index=idx)
                    if did_call:
                        used_llm += 1
                    if did_update:
                        new_cache_entries += 1
                    pred_spans = llm_spans if llm_spans else _extract_spans_from_prediction(pred_text)
                else:
                    pred_spans = _extract_spans_from_prediction(pred_text)
                    if judge_model is not None and (not pred_spans) and parse_on_fail_only:
                        llm_spans, did_call, did_update = _llm_extract_spans(str(pred_text), index=idx)
                        if did_call:
                            used_llm += 1
                        if did_update:
                            new_cache_entries += 1
                        if llm_spans:
                            pred_spans = llm_spans

                pred_arr = np.array(pred_spans, dtype=float) if pred_spans else np.array([])
                if pred_arr.size != 0:
                    pred_arr[:, 0] = np.floor(pred_arr[:, 0])
                    pred_arr[:, 1] = np.ceil(pred_arr[:, 1])

                tiou = _overlap_ratio(pred_arr, gt_arr)
                records.append(
                    dict(
                        tiou=tiou,
                        task=str(row.get('task', '')),
                    )
                )

                if judge_model is not None and new_cache_entries > 0 and (new_cache_entries % flush_every == 0):
                    dump(judge_cache, judge_cache_file)

            if judge_model is not None and new_cache_entries > 0:
                dump(judge_cache, judge_cache_file)

            tiou_arr = np.array([r['tiou'] for r in records], dtype=float)
            mean_tiou = float(np.mean(tiou_arr)) if len(tiou_arr) else 0.0

            r1_metrics = {}
            for th in r1_thresholds:
                key = f'R1@{float(th):g}'
                r1_metrics[key] = float(np.mean(tiou_arr >= float(th))) if len(tiou_arr) else 0.0

            n_all = len(records)

            def group_metrics(filter_fn) -> Dict[str, object]:
                sub = [r for r in records if filter_fn(r)]
                n = len(sub)
                if not n:
                    out: Dict[str, object] = dict(qa_count=0, mIoU=0.0)
                    for th in r1_thresholds:
                        out[f'R1@{float(th):g}'] = 0.0
                    return out
                arr = np.array([x['tiou'] for x in sub], dtype=float)
                out = dict(qa_count=n, mIoU=float(np.mean(arr)))
                for th in r1_thresholds:
                    out[f'R1@{float(th):g}'] = float(np.mean(arr >= float(th)))
                return out

            tasks = sorted({str(r.get('task', '') or '') for r in records})

            res = dict(
                overall=dict(
                    qa_count=n_all,
                    num=n_all,
                    mIoU=mean_tiou,
                    **r1_metrics,
                ),
                r1_thresholds=list(map(float, r1_thresholds)),
                by_task={k: group_metrics(lambda r, kk=k: str(r.get('task', '') or '') == kk) for k in tasks},
                num_llm_parsed=int(used_llm),
            )
            dump(res, score_file)

        return load(score_file)