# Environment variables used by this dataset:
# - OVOBENCH_ROOT: OVOBench video root directory.
# - OVOBENCH_ONLINE_JSON (optional): override path to the ovo_timing online annotation JSON file.

# Download the JSON file from https://huggingface.co/datasets/ZZQ987/VideoChat3_Eval_Online_Json
# Place it under VLMEvalkit/json_data/ovo_timing
# Please ensure the following directory structure:
# 1
# VLMEvalKit/json_data/ovobench/
# ├── ovobench_timing.json
# 2
# export OVOBENCH_ONLINE_JSON="VLMEvalkit/json_data/ovobench/ovobench_timing.json"

import json
import os
import re
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..smp import dump, get_file_extension, load, osp
from .ovobench import OVOBench, VIDEO_EXTENSIONS, _default_ovobench_root


def _default_ovobench_online_json():
    return os.environ.get(
        'OVOBENCH_ONLINE_JSON',
        '/mnt/petrelfs/zhangzhiqiu/Online/VLMEvalkit/json_data/ovobench_online.json',
    ).strip()


STREAMING_SYS = """You are a helpful assistant specializing in streaming video analysis.
You will receive input frame by frame, each labeled with absolute time intervals
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
   - No relevant event has started, OR
   - The current input is irrelevant to the given question.

2. Use </Standby> when:
   - An event is in progress but has not yet completed, OR
   - The current input is relevant but the question cannot yet be answered.

3. Use </Response> only when:
   - An event has fully concluded, OR
   - The available information is sufficient to fully answer the question.
   Provide a complete description at this point.

Do not provide partial answers or speculate beyond the given information.
Whenever you deliver an answer, begin with </Response>.
"""


class OVOTiming(OVOBench):
    TYPE = 'Video-Streaming'
    MODALITY = 'VIDEO'

    def __init__(
        self,
        dataset='OVO_Timing',
        nframe=0,
        fps=2.0,
        streaming_fps: float | None = None,
        frames_limit: int = 4096,
        max_num_frames: int = 32,
        check_extracted_frames: bool = True,
    ):
        self.streaming_fps = float(streaming_fps) if streaming_fps is not None else float(fps)
        self.max_num_frames = max_num_frames
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, frames_limit=frames_limit, check_extracted_frames=check_extracted_frames)

    @classmethod
    def supported_datasets(cls):
        return ['OVO_Timing']

    @staticmethod
    def _gt_timestamps(record: dict[str, Any]) -> list[float]:
        task = record['task']
        if task == 'CRR':
            return [float(record['clue_time'])]
        if task == 'REC':
            return [float(t) + 2.0 for t in record['start_times']]
        return [float(t) + 2.0 for t in record['start_time']]

    @staticmethod
    def _task_question(record: dict[str, Any]) -> str:
        task = record['task']
        if task == 'REC':
            return 'How many times does the event: {} happen?'.format(record['activity'])
        if task == 'SSR':
            return 'Describe the steps of {}'.format(
                re.sub(r'(?<!^)(?=[A-Z])', ' ', record['tutorial']).lower()
            )
        return str(record['question'])

    @staticmethod
    def _end_time(record: dict[str, Any]) -> float:
        info = record['test_info']
        return float(info[-1]['realtime'])

    @classmethod
    def _build_tsv(cls, json_path: str, tsv_path: str, root_dir: str) -> None:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        rows = []
        index = 0
        for record in raw:
            task = record.get('task')
            if task not in ('REC', 'SSR', 'CRR'):
                continue
            sample_id = int(record.get('id', index))
            # Video files on disk are named by sample id, e.g. {id}.mp4 or {id}_{chunk}.mp4 (see _resolve_media_path).
            video_rel = f'{sample_id}.mp4'

            q = cls._task_question(record)
            gt_ts = cls._gt_timestamps(record)
            ask_t = record.get('ask_time')
            rows.append(
                dict(
                    index=index,
                    sample_id=sample_id,
                    task=task,
                    video=video_rel,
                    question=q,
                    answer=str(record.get('answer', '')),
                    gt_timestamp=json.dumps(gt_ts),
                    ask_time=ask_t if ask_t is not None else np.nan,
                    test_info=json.dumps(record['test_info']),
                    start_time=json.dumps(record['start_time']) if task == 'SSR' else '',
                    activity=str(record.get('activity', '')),
                    tutorial=str(record.get('tutorial', '')),
                    end_time=cls._end_time(record),
                )
            )
            index += 1

        dump(pd.DataFrame(rows), tsv_path)

    def prepare_dataset(self, dataset='OVO_Timing', root_dir=None, online_json=None):
        root_dir = root_dir if root_dir is not None else _default_ovobench_root()
        json_path = online_json if online_json is not None else _default_ovobench_online_json()
        tsv_path = json_path.replace('.json', '.tsv')

        # tsv_path = '/mnt/petrelfs/zhangzhiqiu/Online/VLMEvalkit/json_data/ovobench_online_debug.tsv'

        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f'OVO_Timing root directory not found: {root_dir}. Set OVOBENCH_ROOT.'
            )
        if not osp.isfile(json_path):
            raise FileNotFoundError(f'OVO_Timing online json not found: {json_path}')

        if not osp.exists(tsv_path):
            self._build_tsv(json_path, tsv_path, root_dir)
        return dict(root=root_dir, data_file=tsv_path)

    @staticmethod
    def _id_search_roots(root_dir: str, task: str) -> list[str]:
        """Dirs under OVOBench root where id-based videos may live (aligned with _candidate_media_paths)."""
        t = str(task).strip()
        roots = [root_dir]
        for sub in ('chunked_videos', t, osp.join(t, 'chunked_videos'), 'video', osp.join('video', 'chunked_videos')):
            roots.append(osp.join(root_dir, sub))
        seen: set[str] = set()
        out: list[str] = []
        for r in roots:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    @classmethod
    def _find_id_based_video_path(cls, root_dir: str, task: str, sample_id) -> str | None:
        """Resolve {id}.mp4, or if {id}_k.mp4 chunks exist, pick the file with the largest k."""
        if sample_id is None or (isinstance(sample_id, float) and pd.isna(sample_id)):
            return None
        sid = int(sample_id)
        prefix = f'{sid}_'
        best_path: str | None = None
        best_k = -1
        for base in cls._id_search_roots(root_dir, task):
            if not osp.isdir(base):
                continue
            try:
                names = os.listdir(base)
            except OSError:
                continue
            for name in names:
                stem, ext = osp.splitext(name)
                if ext.lower() not in VIDEO_EXTENSIONS:
                    continue
                if not stem.startswith(prefix):
                    continue
                suf = stem[len(prefix) :]
                if not suf.isdigit():
                    continue
                k = int(suf)
                path = osp.join(base, name)
                if osp.isfile(path) and k > best_k:
                    best_k = k
                    best_path = path
        if best_path is not None:
            return best_path
        rel = f'{sid}.mp4'
        for path in OVOBench._candidate_media_paths(root_dir, task, rel):
            if osp.isfile(path):
                return path
        return None

    def _resolve_media_path(self, line):
        sample_id = line.get('sample_id')
        path = self._find_id_based_video_path(self.data_root, str(line.get('task', '')), sample_id)
        if path is not None:
            return path, 'file'
        return super()._resolve_media_path(line)

    def save_forward_stream_frames(self, line) -> tuple[list[str], list[float]]:
        """Decode and cache JPEGs on disk (same pipeline spirit as OVOBench.save_video_frames)."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        task = str(line['task'])
        test_info = json.loads(line['test_info']) if isinstance(line['test_info'], str) else line['test_info']
        end_time = float(test_info[-1]['realtime'])
        if task in ('REC', 'SSR'):
            start_time = 0.0
        else:
            start_time = float(line['ask_time'])

        dt = 1.0 / self.streaming_fps if self.streaming_fps > 0 else 0.5
        stream_times: list[float] = []
        curr = float(start_time)
        while curr <= end_time + 1e-6:
            stream_times.append(curr)
            curr += dt

        if not stream_times:
            raise ValueError('empty stream timeline')

        line_key = line if isinstance(line, pd.Series) else pd.Series(line)
        media_path, source_kind = self._resolve_media_path(line_key)

        if source_kind != 'file':
            raise NotImplementedError(
                f'OVO_Timing pre-extract requires a video file; got directory stream: {media_path}'
            )

        import decord
        from decord import cpu

        vr = decord.VideoReader(media_path, ctx=cpu(0), num_threads=1)
        source_fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        if source_fps <= 0:
            raise ValueError(f'invalid video fps for {media_path}')

        frame_indices: list[int] = []
        for t in stream_times:
            fi = int(t * source_fps)
            fi = min(max(0, fi), max(0, total_frames - 1))
            frame_indices.append(fi)

        mode = f'forward_{str(self.streaming_fps).replace(".", "p")}fps'
        cache_key = self._cache_key(
            line_key.get('task', ''),
            line_key['video'],
            start_time,
            end_time,
            mode,
            source_kind,
        )
        cache_root = osp.join(self.frame_root, cache_key)
        os.makedirs(cache_root, exist_ok=True)
        n = len(stream_times)
        frame_paths = [osp.join(cache_root, f'stream-{i:06d}-of-{n}.jpg') for i in range(n)]

        pos_by_idx: dict[int, list[int]] = defaultdict(list)
        for pos, fi in enumerate(frame_indices):
            pos_by_idx[fi].append(pos)

        for fi in sorted(pos_by_idx.keys()):
            arr = vr[fi].asnumpy()
            im = Image.fromarray(arr)
            for pos in pos_by_idx[fi]:
                pth = frame_paths[pos]
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths, stream_times

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        frame_paths, stream_times = self.save_forward_stream_frames(line)

        task = str(line['task'])
        test_info = json.loads(line['test_info']) if isinstance(line['test_info'], str) else line['test_info']
        end_time = float(test_info[-1]['realtime'])
        start_time = 0.0 if task in ('REC', 'SSR') else float(line['ask_time'])

        meta = {
            'frame_paths': frame_paths,
            'stream_times': stream_times,
            'streaming_fps': self.streaming_fps,
            'start_time': start_time,
            'end_time': end_time,
            'history_length': 5,
            'history_fps': 1,
            'proposal_update_duration': 9999,
            'max_num_frames': self.max_num_frames,
            'max_new_tokens': 128,
            'question': str(line['question']),
            'task': task,
            'sample_id': int(line['sample_id']) if not pd.isna(line.get('sample_id', np.nan)) else int(line['index']),
            'test_info': test_info,
            'gt_timestamp': json.loads(line['gt_timestamp'])
            if isinstance(line['gt_timestamp'], str)
            else list(line['gt_timestamp']),
            'prebuilt_frames': True,
        }
        return [
            dict(type='text', value=STREAMING_SYS, role='system'),
            dict(type='text', value='[OVOBench online streaming]', ovo_online_meta=meta),
        ]

    @staticmethod
    def _f1_stats(gt_times: list[float], det_times: list[float], tol: float = 2.0) -> tuple[float, float]:
        det_copy = list(det_times)
        correct = 0
        for gt in gt_times:
            for i, t in enumerate(det_copy):
                if abs(float(t) - float(gt)) <= tol:
                    correct += 1
                    det_copy.pop(i)
                    break
        recall = correct / len(gt_times) if gt_times else 0.0
        precision = correct / len(det_times) if det_times else 0.0
        return recall, precision

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'Evaluation file should be in xlsx/json/tsv format'
        )
        judge_name = judge_kwargs.get('model', 'exact_matching')
        if judge_name not in [None, 'exact_matching']:
            warnings.warn(f'OVO_Timing uses rule-based F1. Ignoring judge model `{judge_name}`.')

        from ..smp.file import get_intermediate_file_path

        score_file = get_intermediate_file_path(eval_file, '_score')
        acc_file = get_intermediate_file_path(eval_file, '_acc', 'json')

        data = load(eval_file)
        scored = data.copy()

        recalls_crr, precs_crr = [], []
        recalls_rec, precs_rec = [], []
        recalls_ssr, precs_ssr = [], []
        parsed = []

        for _, row in scored.iterrows():
            pred_raw = row.get('prediction', '')
            gt_list = row.get('gt_timestamp', '[]')
            if isinstance(gt_list, str):
                gt_list = json.loads(gt_list)
            gt_list = [float(x) for x in gt_list]

            det_times: list[float] = []
            if isinstance(pred_raw, str) and pred_raw.strip().startswith('{'):
                try:
                    obj = json.loads(pred_raw)
                    if isinstance(obj.get('checked_detections'), list) and obj['checked_detections']:
                        det_times = [float(x['time']) for x in obj['checked_detections']]
                    elif isinstance(obj.get('detections'), list):
                        det_times = [float(x) for x in obj['detections']]
                    elif isinstance(obj.get('response_events'), list):
                        det_times = [float(x['time']) for x in obj['response_events']]
                except json.JSONDecodeError:
                    det_times = []
            parsed.append(json.dumps(det_times) if det_times else '')

            r, p = self._f1_stats(gt_list, det_times, tol=2.0)
            task = str(row.get('task', ''))
            if task == 'CRR':
                recalls_crr.append(r)
                precs_crr.append(p)
            elif task == 'REC':
                recalls_rec.append(r)
                precs_rec.append(p)
            elif task == 'SSR':
                recalls_ssr.append(r)
                precs_ssr.append(p)

        scored['parsed_detections'] = parsed
        dump(scored, score_file)

        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        rr_crr, pr_crr = _mean(recalls_crr), _mean(precs_crr)
        rr_rec, pr_rec = _mean(recalls_rec), _mean(precs_rec)
        rr_ssr, pr_ssr = _mean(recalls_ssr), _mean(precs_ssr)

        f1_crr = 2 * rr_crr * pr_crr / (rr_crr + pr_crr) if (rr_crr + pr_crr) > 0 else 0.0
        f1_rec = 2 * rr_rec * pr_rec / (rr_rec + pr_rec) if (rr_rec + pr_rec) > 0 else 0.0
        f1_ssr = 2 * rr_ssr * pr_ssr / (rr_ssr + pr_ssr) if (rr_ssr + pr_ssr) > 0 else 0.0
        avg_rr = (rr_crr + rr_rec + rr_ssr) / 3.0
        avg_pr = (pr_crr + pr_rec + pr_ssr) / 3.0
        avg_f1 = (f1_crr + f1_rec + f1_ssr) / 3.0

        result = {
            'CRR': {'recall_at_2s': rr_crr, 'precision_at_2s': pr_crr, 'f1_at_2s': f1_crr, 'n': len(recalls_crr)},
            'REC': {'recall_at_2s': rr_rec, 'precision_at_2s': pr_rec, 'f1_at_2s': f1_rec, 'n': len(recalls_rec)},
            'SSR': {'recall_at_2s': rr_ssr, 'precision_at_2s': pr_ssr, 'f1_at_2s': f1_ssr, 'n': len(recalls_ssr)},
            'average_recall_at_2s': avg_rr,
            'average_precision_at_2s': avg_pr,
            'average_f1_at_2s': avg_f1,
        }
        print(
            f'OVO_Timing @2s — avg recall: {avg_rr:.4f}, avg precision: {avg_pr:.4f}, avg F1: {avg_f1:.4f}',
            flush=True,
        )
        dump(result, acc_file)
        return result
