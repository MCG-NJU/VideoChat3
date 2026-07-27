import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset
from .utils.video_pyav import ffprobe_video_info, get_video_decode_backend, save_frames_by_indices_pyav
from .utils.vtg_reason import REASON_PROMPT_SUFFIX, extract_answer_from_tags, wrap_vtg_prompt_with_reason
from .video_meta_cache import (
    ensure_video_meta_json,
    load_video_meta,
    sample_fps_from_meta,
    sample_frame_count_from_meta,
)


FAIL_MSG = 'Failed to obtain answer via API.'


@dataclass(frozen=True)
class _VUE_TR_Paths:
    root: str
    videos_dir: str
    gt_json: str
    tsv_path: str


def _default_vue_tr_root() -> str:
    return os.environ.get('VUE_TR_ROOT', '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/vidi/VUE_TR')


def _default_vue_tr_videos_dir_under_root(root: str) -> str:
    """VUE_TR videos are stored under `<VUE_TR_ROOT>/videos`."""
    return os.path.join(root, 'videos')


def _index_video_files(videos_dir: str) -> Dict[str, str]:
    """
    Build a best-effort mapping: video_id -> filename under videos_dir.
    Supports both:
      - <video_id>.mp4
      - <video_id>.<suffix>.mp4 (e.g., "<id>.f399.mp4")
    """
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f'[VUE_TR] videos_dir not found: {videos_dir}')
    files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    mp: Dict[str, str] = {}
    for f in files:
        prefix = f.split('.')[0]
        # Prefer deterministic shortest name (same as _resolve_video_file)
        if prefix not in mp or len(f) < len(mp[prefix]) or (len(f) == len(mp[prefix]) and f < mp[prefix]):
            mp[prefix] = f
    return mp


def _merge_time_spans(intervals: np.ndarray) -> np.ndarray:
    if len(intervals) == 0:
        return np.array([])
    intervals = intervals[np.argsort(intervals[:, 0])]
    merged = [intervals[0].copy()]
    for current in intervals[1:]:
        prev_end = merged[-1][1]
        curr_start, curr_end = current[0], current[1]
        if curr_start <= prev_end:
            merged[-1][1] = max(prev_end, curr_end)
        else:
            merged.append(current.copy())
    return np.array(merged)


def _overlap_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    IoU between two (multi-span) sets, ported from VUE_TR/qa_eval.py
    """
    if len(gt) == 0 or gt.shape[0] == 0:
        return 1.0 if (len(pred) == 0 or pred.shape[0] == 0) else 0.0
    if len(pred) == 0 or pred.shape[0] == 0:
        return 0.0

    pred = _merge_time_spans(pred)
    pred = pred[pred[:, 0] <= pred[:, 1]]
    if pred.size == 0:
        return 0.0

    len_gt = float(np.sum(gt[:, 1] - gt[:, 0]))
    len_pred = float(np.sum(pred[:, 1] - pred[:, 0]))
    union = len_pred + len_gt

    intersect = 0.0
    for i in range(len(pred)):
        for j in range(len(gt)):
            sta = max(pred[i][0], gt[j][0])
            end = min(pred[i][1], gt[j][1])
            intersect += max(0.0, end - sta)
    union -= intersect
    iou = intersect / (union + 1e-16)
    iou = float(max(min(1.0, iou), 0.0))
    return iou


def _per_sample_ious(results: List[dict]) -> np.ndarray:
    if not results:
        return np.array([], dtype=float)
    return np.array(
        [_overlap_ratio(r['answer'], r['gt']) for r in results],
        dtype=float,
    )


def _compute_miou_and_r1(
    results: List[dict],
    r1_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
) -> Tuple[float, Dict[str, float]]:
    ious = _per_sample_ious(results)
    if len(ious) == 0:
        return 0.0, {f'R1@{float(th):g}': 0.0 for th in r1_thresholds}
    miou = float(np.mean(ious))
    r1 = {f'R1@{float(th):g}': float(np.mean(ious >= float(th))) for th in r1_thresholds}
    return miou, r1


def _success_overlap(results: List[dict]) -> Tuple[np.ndarray, float]:
    thres = np.linspace(0, 1, 101)
    n_query = len(results)
    success = np.zeros(len(thres))
    ious = _per_sample_ious(results)
    for i, t in enumerate(thres):
        success[i] = float(np.sum(ious > t) / (n_query + 1e-16))
    auc = float(np.trapz(success, thres))
    return success, auc


def _interval_intersection(intervals1: List[List[float]], intervals2: List[List[float]]) -> List[Tuple[float, float]]:
    i, j = 0, 0
    result = []
    intervals1 = sorted(intervals1)
    intervals2 = sorted(intervals2)
    while i < len(intervals1) and j < len(intervals2):
        a_start, a_end = intervals1[i]
        b_start, b_end = intervals2[j]
        if a_start <= b_end and b_start <= a_end:
            result.append((max(a_start, b_start), min(a_end, b_end)))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result


def _interval_union(intervals1: List[List[float]], intervals2: List[List[float]]) -> List[List[float]]:
    intervals = sorted([x[:] for x in intervals1] + [x[:] for x in intervals2])
    result: List[List[float]] = []
    if not intervals:
        return result
    cur = intervals[0]
    for it in intervals[1:]:
        if it[0] <= cur[1]:
            cur[1] = max(cur[1], it[1])
        else:
            result.append(cur)
            cur = it
    result.append(cur)
    return result


def _compute_precision_recall(results: List[dict], avg: bool = True) -> Tuple[float, float]:
    gt_all, pred_all, inter_all = [], [], []
    for item in results:
        gt = [[min(x), max(x)] for x in item['gt'].tolist() if len(x) == 2]
        pred = [[min(x), max(x)] for x in item['answer'].tolist() if len(x) == 2]
        inter = _interval_intersection(gt, pred)
        uni = _interval_union(gt, pred)
        gt_all.append(sum([x[1] - x[0] for x in gt]))
        pred_all.append(sum([x[1] - x[0] for x in pred]))
        inter_all.append(sum([x[1] - x[0] for x in inter]))
        # union not needed for pre/rec here (same as official)
        _ = uni

    gt_all = np.array(gt_all, dtype=float)
    pred_all = np.array(pred_all, dtype=float)
    inter_all = np.array(inter_all, dtype=float)

    recall = np.array([i / g for i, g in zip(inter_all, gt_all) if g != 0], dtype=float)
    precision = np.array([i / p for i, p in zip(inter_all, pred_all) if p != 0], dtype=float)

    if not avg:
        return float(np.mean(precision)) if len(precision) else 0.0, float(np.mean(recall)) if len(recall) else 0.0

    thres = np.linspace(0, 1, 101)
    precision_thres = np.zeros(len(thres))
    recall_thres = np.zeros(len(thres))
    for i, t in enumerate(thres):
        precision_thres[i] = float(np.mean(precision >= t)) if len(precision) else 0.0
        recall_thres[i] = float(np.mean(recall >= t)) if len(recall) else 0.0
    precision_auc = float(np.trapz(precision_thres, thres))
    recall_auc = float(np.trapz(recall_thres, thres))
    return precision_auc, recall_auc


def _extract_spans_from_prediction(pred: object) -> List[List[float]]:
    """
    Best-effort span extraction from model output.
    Preferred: JSON list like [[s,e], [s,e], ...]
    Fallback: regex pairs like "s - e" or "s to e"
    """
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return []

    # if already structured
    if isinstance(pred, list):
        spans = []
        for it in pred:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                spans.append([float(it[0]), float(it[1])])
        return spans

    s = str(pred).strip()
    if not s:
        return []

    # try JSON extraction first
    try:
        json_objs = list(extract_json_objects(s))
        for obj in json_objs:
            if isinstance(obj, list):
                spans = []
                for it in obj:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        spans.append([float(it[0]), float(it[1])])
                if spans:
                    return spans
    except Exception:
        pass

    # try bracket pairs: [a, b]
    pair_pat = re.compile(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')
    pairs = pair_pat.findall(s)
    if pairs:
        return [[float(a), float(b)] for a, b in pairs]

    # try "a - b" or "a to b"
    span_pat = re.compile(r'(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)')
    pairs2 = span_pat.findall(s.lower())
    if pairs2:
        return [[float(a), float(b)] for a, b in pairs2]

    # fallback: take all numbers and pair them
    nums = re.findall(r'-?\d+(?:\.\d+)?', s)
    nums = nums[: len(nums) // 2 * 2]
    return [[float(nums[i]), float(nums[i + 1])] for i in range(0, len(nums), 2)]


def _ensure_vue_tr_tsv(paths: _VUE_TR_Paths) -> None:
    if os.path.exists(paths.tsv_path):
        try:
            df = load(paths.tsv_path)
            # `VideoBaseDataset` requires `question` and `video`
            if 'question' in df.columns and 'video' in df.columns:
                # TSV may be generated before all videos are downloaded; validate against local videos_dir.
                try:
                    video_map = _index_video_files(paths.videos_dir)
                    vid_col = 'video_id' if 'video_id' in df.columns else 'video'
                    uniq_vids = set(str(x) for x in df[vid_col].dropna().tolist())
                    missing = [v for v in uniq_vids if v not in video_map]
                    if len(missing) == 0:
                        return
                except Exception:
                    # If validation fails for any reason, regenerate TSV.
                    pass
        except Exception:
            # fallthrough to regenerate
            pass
    if not os.path.exists(paths.gt_json):
        raise FileNotFoundError(f'[VUE_TR] ground truth not found: {paths.gt_json}')

    gts = json.load(open(paths.gt_json, 'r', encoding='utf-8'))
    if not isinstance(gts, list):
        raise ValueError(f'[VUE_TR] ground truth must be a list, got {type(gts)}')

    # Filter to locally available videos (since some youtube videos may be missing).
    video_map = _index_video_files(paths.videos_dir)

    rows = []
    for item in gts:
        query = str(item['query'])
        video_id = str(item['video_id'])
        if video_id not in video_map:
            # skip missing videos silently to allow running on partial downloads
            continue
        rows.append(
            dict(
                index=int(item['query_id']),
                query_id=int(item['query_id']),
                video=video_id,  # required by VideoBaseDataset
                video_id=video_id,
                duration=float(item['duration']),
                query=query,
                question=query,  # compatibility alias
                gt=json.dumps(item['gt']),
                task=str(item.get('task', 'temporal_retrieval')),
                duration_category=str(item.get('duration_category', '')),
                query_format=str(item.get('query_format', '')),
                query_modality=str(item.get('query_modality', '')),
            )
        )
    df = pd.DataFrame(rows).sort_values('index').reset_index(drop=True)
    os.makedirs(os.path.dirname(paths.tsv_path), exist_ok=True)
    dump(df, paths.tsv_path)


class VUE_TR(VideoBaseDataset):
    """
    VUE-TR: Video temporal retrieval benchmark.
    Ground truth is provided in `VUE-TR_ground_truth.json`.
    Videos are expected under `VUE_TR_VIDEOS_DIR`.
    Note: videos may be partially downloaded; dataset building will not fail on missing videos,
    but `build_prompt` will raise if a requested video file is missing.
    """

    TYPE = 'Video-VTG'
    MODALITY = 'VIDEO'

    PROMPT_VIDEO = (
        "You are given a video.\n"
        "Task: temporal retrieval.\n"
        "Given the query: \"{query}\", return ALL time spans (in seconds) where the query is relevant.\n"
        "Output format MUST be a JSON array of [start, end] pairs, e.g. [[0, 3.5], [10, 12]].\n"
    )

    PROMPT_FRAMES_PREFIX = (
        "You are given a video as a sequence of frames.\n"
        "The timestamp (in seconds) is shown before each frame.\n"
        "Task: temporal retrieval.\n"
    )

    def __init__(
        self,
        dataset='VUE_TR',
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28*28,
        max_pixels=448*448,
        total_pixels=32000*2*4*14*14,
        check_extracted_frames=True,
        reason=False,
    ):
        self.reason = reason
        self.frames_limit = frames_limit  # cap frames per video for FPS sampling
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )
        self._video_file_cache: Dict[str, str] = {}
        self._video_meta = load_video_meta(getattr(self, '_video_meta_path', None))
        self._video_duration_cache: Dict[str, float] = {}

        self.data = self.data[self.data['query_modality'] == "vision"].reset_index(drop=True)
        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos

    @classmethod
    def supported_datasets(cls):
        return ['VUE_TR']

    def prepare_dataset(self, dataset):
        root = _default_vue_tr_root()
        videos_dir = _default_vue_tr_videos_dir_under_root(root)
        paths = _VUE_TR_Paths(
            root=root,
            videos_dir=videos_dir,
            gt_json=os.path.join(root, 'VUE-TR_ground_truth.json'),
            tsv_path=os.path.join(root, 'VUE_TR.tsv'),
        )
        meta_path = paths.tsv_path.replace('.tsv', '_video_meta.json')
        self._video_meta_path = meta_path
        _ensure_vue_tr_tsv(paths)
        try:
            df = load(paths.tsv_path)
            vid_col = 'video_id' if 'video_id' in df.columns else 'video'
            video_ids = sorted(set(str(x) for x in df[vid_col].dropna().tolist()))
            video_map = _index_video_files(paths.videos_dir)
            ensure_video_meta_json(
                meta_path,
                video_ids,
                lambda video_id: os.path.join(paths.videos_dir, video_map[str(video_id)]),
                prefer_pyav=get_video_decode_backend() == 'pyav',
            )
        except Exception as e:
            warnings.warn(f'[VUE_TR] failed to ensure video meta json: {e}')
        # data_root should be videos dir for runtime resolution
        self._videos_dir = paths.videos_dir
        return dict(root=paths.videos_dir, data_file=paths.tsv_path)

    def _resolve_video_file(self, video_id: str) -> str:
        if video_id in self._video_file_cache:
            return self._video_file_cache[video_id]
        # Exact match
        p1 = os.path.join(self._videos_dir, f'{video_id}.mp4')
        if os.path.exists(p1):
            self._video_file_cache[video_id] = p1
            return p1
        # Prefix match (e.g., "<id>.f399.mp4")
        cands = [x for x in os.listdir(self._videos_dir) if x.startswith(video_id) and x.endswith('.mp4')]
        if len(cands) == 1:
            p = os.path.join(self._videos_dir, cands[0])
            self._video_file_cache[video_id] = p
            return p
        if len(cands) > 1:
            # pick shortest name deterministically
            cands.sort(key=lambda x: (len(x), x))
            p = os.path.join(self._videos_dir, cands[0])
            self._video_file_cache[video_id] = p
            return p
        raise FileNotFoundError(
            f'[VUE_TR] video file not found for video_id={video_id} under {self._videos_dir} (videos may be not fully downloaded).'
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
            if backend != "pyav":
                import decord

                vr = decord.VideoReader(vid_path, num_threads=1)
                total = int(len(vr))
                video_fps = float(vr.get_avg_fps())
                duration = float(total / video_fps) if video_fps > 0 else duration
            else:
                raise RuntimeError("force pyav")
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
        use_pyav = backend == "pyav"

        vr = None
        try:
            if backend != "pyav":
                import decord
                # Use single-threaded decoder for stability on some long/corrupted videos.
                vr = decord.VideoReader(vid_path, num_threads=1)
                total = int(len(vr))
                video_fps = float(vr.get_avg_fps())
                duration = float(total / video_fps) if video_fps > 0 else 0.0
            else:
                raise RuntimeError("force pyav")
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
                if backend != "decord":
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

    def _video_llm_cached_prompt_from_meta(self, video_id: str, query: str):
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
        # Override: VUE_TR videos are not necessarily named `<video>.mp4`.
        if isinstance(line, int):
            line = self.data.iloc[line]
        video_id = str(line['video_id'])
        vid_path = self._resolve_video_file(video_id)

        vr, indices, _, video_info = self._sample_indices_and_timestamps(vid_path)

        # Choose proper frame template based on mode
        if self.fps > 0:
            frame_paths = self.frame_paths_fps(video_id, len(indices))
        else:
            frame_paths = self.frame_paths(video_id)

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )
        if need_extract:
            print(f"video ({video_id}) is not extracted")
            if video_info.get('backend') == 'pyav' or vr is None:
                save_frames_by_indices_pyav(
                    vid_path=vid_path,
                    indices=[int(x) for x in indices],
                    frame_paths=frame_paths,
                    total_frames=video_info.get('n_frames', None),
                    desc=f"Extracting (pyav): {video_id}",
                )
            else:
                try:
                    # Stream decode & save to avoid holding many frames in RAM (prevents OOM on long videos).
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
                        desc=f"Extracting (pyav): {video_id}",
                    )
        return frame_paths

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        query = str(line['query'])
        video_id = str(line['video_id'])

        if video_llm and not self.check_extracted_frames:
            cached = self._video_llm_cached_prompt_from_meta(video_id, query)
            if cached is not None:
                return cached

        # For Qwen3-VL: always rely on extracted frames for video modality, so we can control sample_fps & caps.
        vid_path = self._resolve_video_file(video_id)
        vr, indices, timestamps, video_info = self._sample_indices_and_timestamps(vid_path)
        duration = float(video_info.get('duration', 0.0))

        if self.fps > 0:
            frame_paths = self.frame_paths_fps(video_id, len(indices))
        else:
            frame_paths = self.frame_paths(video_id)

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )
        if need_extract:
            if video_info.get('backend') == 'pyav' or vr is None:
                save_frames_by_indices_pyav(
                    vid_path=vid_path,
                    indices=[int(x) for x in indices],
                    frame_paths=frame_paths,
                    total_frames=video_info.get('n_frames', None),
                    desc=f"Extracting (pyav): {video_id}",
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
                        desc=f"Extracting (pyav): {video_id}",
                    )

        if video_llm:
            assert self.fps > 0
            actual_fps = (self.frames_limit / duration) if (len(frame_paths) == self.frames_limit and duration > 0) else self.fps
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

        # Multi-image mode: interleave timestamps + frames
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
        Compute VUE-TR metrics consistent with `qa_eval.py`:
          - IoU: success@thres curve AUC (success_overlap)
          - Precision/Recall: AUC over thresholded precision/recall distributions
          - mIoU and R1@θ (fraction of samples with IoU >= θ; default θ in [0.3, 0.5, 0.7])
        Also provide breakdown over:
          - duration_category: ultra-short/short/medium/long/ultra-long
          - query_format: keyword/phrase/sentence
          - query_modality: audio/vision/vision+audio
        """
        # Temporal retrieval evaluation is a strict numeric metric (IoU/AUC/Precision/Recall AUC).
        # We do NOT use LLM to judge correctness directly; however, an optional LLM can be used to
        # normalize/parse model output into a JSON array of [start, end] spans (seconds), then we compute
        # metrics as usual.
        r1_thresholds: List[float] = judge_kwargs.get('r1_thresholds', [0.3, 0.5, 0.7])
        if not isinstance(r1_thresholds, (list, tuple)) or not r1_thresholds:
            r1_thresholds = [0.3, 0.5, 0.7]
        r1_thresholds = tuple(float(x) for x in r1_thresholds)

        model_name = judge_kwargs.get('model', 'exact_matching')
        # Default behavior when judge model is specified: try rule-based parsing first,
        # with LLM as fallback if rule-based parsing fails.
        force_llm = bool(judge_kwargs.get('force_llm_judge', False))
        parse_on_fail_only = bool(judge_kwargs.get('llm_parse_on_fail_only', True))
        print(f"force_llm: {force_llm}, parse_on_fail_only: {parse_on_fail_only}")
        judge_model = None
        if model_name not in [None, 'exact_matching']:
            if gpt_key_set():
                try:
                    from .utils import build_judge, DEBUG_MESSAGE
                    judge_model = build_judge(**judge_kwargs)
                    if hasattr(judge_model, 'working') and (not judge_model.working()):
                        warnings.warn('[VUE_TR] Judge model is not working properly, will use rule-based parsing only.')
                        warnings.warn(DEBUG_MESSAGE)
                        judge_model = None
                except Exception as e:
                    warnings.warn(
                        f'[VUE_TR] Failed to build judge model ({model_name}), will use rule-based parsing only: '
                        f'{type(e)}: {e}'
                    )
                    try:
                        from .utils import DEBUG_MESSAGE
                        warnings.warn(DEBUG_MESSAGE)
                    except Exception:
                        pass
                    judge_model = None
            else:
                warnings.warn('[VUE_TR] API key is not set, will use rule-based parsing only.')
                judge_model = None

        # If judge is unavailable, disable forcing.
        if judge_model is None:
            force_llm = False
            parse_on_fail_only = True
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )
        safe_model_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(model_name))
        suffix = f'_{safe_model_name}_score' if model_name not in [None, 'exact_matching'] else '_score'
        score_file = get_intermediate_file_path(eval_file, suffix, 'json')
        # 即使 score_file 存在，如果 prediction 更新了，我们可能需要重新计算。
        # 但 VLMEvalKit 通常依赖手动删除 score 文件来触发重评。
        # 这里我们修改逻辑：支持 judge 结果缓存，以便用户删除 score 文件后能快速重评。

        judge_cache_file = get_intermediate_file_path(eval_file, f'_{safe_model_name}_judge_cache', 'pkl')
        
        # 评测文件已存在时，直接复用；若需要重评，请手动删除 score_file。
        if not os.path.exists(score_file):
            df = load(eval_file)
            if 'prediction' not in df.columns:
                raise ValueError('[VUE_TR] eval file must contain `prediction`.')
            if 'gt' not in df.columns:
                raise ValueError('[VUE_TR] eval file must contain `gt`.')

            # Load cache if exists
            judge_cache = {}
            if judge_model is not None and os.path.exists(judge_cache_file):
                try:
                    judge_cache = load(judge_cache_file)
                except Exception:
                    judge_cache = {}
            if not isinstance(judge_cache, dict):
                judge_cache = {}

            def _normalize_cache_key(row, fallback_i: int):
                """
                Prefer stable int index if available; otherwise fallback to loop index.
                """
                try:
                    if 'index' in row and (not pd.isna(row.get('index'))):
                        return int(row.get('index'))
                except Exception:
                    pass
                return int(fallback_i)

            def _llm_extract_spans(pred_text: str, index=None):
                """
                Use LLM to normalize/parse model output into JSON array of [start, end] spans in seconds.
                Returns: (spans, used_llm: bool, cache_updated: bool)
                """
                if judge_model is None:
                    return [], False, False
                
                # Check cache
                if index is not None and index in judge_cache:
                    return judge_cache[index], False, False

                prompt = (
                    "You are a parser. Extract ALL relevant time spans from the model output.\n"
                    "Output MUST be a JSON array of [start, end] pairs (numbers, seconds), e.g. [[0, 3.5], [10, 12]].\n"
                    "Do not output any extra text. If none, output: []\n"
                    f"Model output:\n{pred_text}\n"
                )
                resp = judge_model.generate(prompt)
                
                res_spans = []
                try:
                    js = list(extract_json_objects(resp))
                    if js:
                        # find a list in extracted json objects
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
                
                # Update cache if we got a result (even if empty list, it's a valid result from LLM)
                if index is not None:
                    judge_cache[index] = res_spans
                
                return res_spans, True, True

            # Build evaluation records
            records: List[dict] = []
            used_llm = 0  # count REAL LLM calls (cache miss only)
            new_cache_entries = 0
            flush_every = 50  # simple/robust: periodically persist cache to avoid losing progress on interruption

            for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating VUE_TR"):
                idx = _normalize_cache_key(row, i)
                gt = json.loads(row['gt']) if isinstance(row['gt'], str) else row['gt']
                gt_arr = np.array(gt, dtype=float) if gt else np.array([])

                pred_text = row['prediction']
                if self.reason:
                    pred_text = extract_answer_from_tags(pred_text)
                pred_spans: List[List[float]] = []

                if judge_model is not None and (force_llm or (not parse_on_fail_only)):
                    # LLM-first (forced)
                    llm_spans, did_call, did_update = _llm_extract_spans(str(pred_text), index=idx)
                    if did_call:
                        used_llm += 1
                    if did_update:
                        new_cache_entries += 1
                    
                    if llm_spans:
                        pred_spans = llm_spans
                    else:
                        # fallback to rule-based parsing
                        pred_spans = _extract_spans_from_prediction(pred_text)
                else:
                    # rule-first; optionally fallback to LLM if parsing fails
                    pred_spans = _extract_spans_from_prediction(pred_text)
                    if (judge_model is not None) and (not pred_spans) and parse_on_fail_only:
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
                rec = dict(
                    answer=pred_arr,
                    gt=gt_arr,
                    duration_category=str(row.get('duration_category', '')),
                    query_format=str(row.get('query_format', '')),
                    query_modality=str(row.get('query_modality', '')),
                )
                records.append(rec)

                # Periodic cache flush for robustness
                if judge_model is not None and new_cache_entries > 0 and (new_cache_entries % flush_every == 0):
                    dump(judge_cache, judge_cache_file)

            # Final cache flush
            if judge_model is not None and new_cache_entries > 0:
                dump(judge_cache, judge_cache_file)

            # Overall
            _, iou_auc = _success_overlap(records)
            pre_auc, rec_auc = _compute_precision_recall(records, avg=True)
            miou, r1_metrics = _compute_miou_and_r1(records, r1_thresholds=r1_thresholds)

            def _empty_group_metrics() -> Dict[str, float]:
                out = dict(precision_auc=0.0, recall_auc=0.0, iou_auc=0.0, mIoU=0.0, num=0)
                for th in r1_thresholds:
                    out[f'R1@{float(th):g}'] = 0.0
                return out

            def group_metrics(filter_fn) -> Dict[str, float]:
                sub = [r for r in records if filter_fn(r)]
                if not sub:
                    return _empty_group_metrics()
                _, iou = _success_overlap(sub)
                p, rr = _compute_precision_recall(sub, avg=True)
                sub_miou, sub_r1 = _compute_miou_and_r1(sub, r1_thresholds=r1_thresholds)
                return dict(
                    precision_auc=p,
                    recall_auc=rr,
                    iou_auc=iou,
                    mIoU=sub_miou,
                    num=len(sub),
                    **sub_r1,
                )

            # Breakdown
            duration_cats = ["ultra-short", "short", "medium", "long", "ultra-long"]
            query_formats = ["keyword", "phrase", "sentence"]
            query_modalities = ["vision"]

            res = dict(
                overall=dict(
                    precision_auc=pre_auc,
                    recall_auc=rec_auc,
                    iou_auc=iou_auc,
                    mIoU=miou,
                    num=len(records),
                    **r1_metrics,
                ),
                r1_thresholds=list(r1_thresholds),
                by_duration={k: group_metrics(lambda r, kk=k: r['duration_category'] == kk) for k in duration_cats},
                by_query_format={k: group_metrics(lambda r, kk=k: r['query_format'] == kk) for k in query_formats},
                by_query_modality={k: group_metrics(lambda r, kk=k: r['query_modality'] == kk) for k in query_modalities},
                num_llm_parsed=int(used_llm),
            )
            dump(res, score_file)
        return load(score_file)