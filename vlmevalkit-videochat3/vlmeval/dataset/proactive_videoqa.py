# Environment variables used by this dataset:
# - PROACTIVE_VIDEOQA_ROOT: root containing EGO/TV/VAD/WEB task directories.
# - QWEN_API_KEY and QWEN_API_BASE: required when evaluate() calls the default Qwen judge.
# - PROACTIVE_JUDGE_MODEL (optional): judge model override.

"""ProactiveVideoQA dataset for VLMEvalKit online (streaming) evaluation.

Four sub-tasks are exposed as independent dataset names:
    ProactiveVideoQA_EGO / ProactiveVideoQA_TV
    ProactiveVideoQA_VAD / ProactiveVideoQA_WEB

Data layout expected:
    <data_root>/<TASK>/anno.json   — list of annotation dicts
    <data_root>/<TASK>/videos/     — video files

Annotation dict schema (anno.json):
    {"video": "xxx.mp4", "question_id": "...", "duration": float,
     "conversation": [{"role": "user", "content": ..., "time": ...}, ...],
     "answer": [{"content": ..., "related_timespan": [t0,t1],
                 "reply_timespan": [t0,t1]}, ...]}

Evaluation:
    Uses VLMEvalKit's standard judge interface to score each prediction,
    then computes PAUC at omega ∈ {0, 0.5, 1}.

    Required env vars:
        QWEN_API_KEY and QWEN_API_BASE — credentials for the default judge
    Optional:
        PROACTIVE_VIDEOQA_ROOT      — override data root
        PROACTIVE_JUDGE_MODEL       — judge model name

PAUC algorithm ported verbatim from
    ProactiveVideoQA/pauc/judge_llm_scores_openai.py
(no import of original project code).
"""
from __future__ import annotations

import json
import math
import os
import os.path as osp
import re
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd

from ..smp import LMUDataRoot, load, dump, get_logger
from ..smp.file import get_intermediate_file_path
from ..utils import track_progress_rich
from .utils import build_judge, DEBUG_MESSAGE
from .video_base import VideoBaseDataset

logger = get_logger('ProactiveVideoQA')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Must match STREAM_META_PREFIX in vlmeval/vlm/streaming_video_base.py
STREAM_META_PREFIX = '__PROACTIVE_STREAM_META__'

_DEFAULT_DATA_ROOT = (
    '/mnt/petrelfs/zengxiangyu/Research_yyd/A_Datasets/online_eval/ProactiveVideoQA'
)
TARGET_FPS = 1.0

_TASK_MAP = {
    'ProactiveVideoQA_EGO': 'EGO',
    'ProactiveVideoQA_TV': 'TV',
    'ProactiveVideoQA_VAD': 'VAD',
    'ProactiveVideoQA_WEB': 'WEB',
}

_REQUIRED_COLS = frozenset({
    'index', 'question_id', 'video', 'question',
    'duration', 'task', 'conversation_json', 'answer_json',
})

JUDGE_INSTRUCTION = (
    'You are an evaluator for a video question answering system. '
    'Your task is to rate whether the predicted answer covers the key points '
    'of the ground truth answer. Use the following scale to assign a score:\n'
    '- 3: Mostly covered; the predicted answer covers all key information in '
    'the ground truth answer, though it may have minor inaccuracies or rephrases.\n'
    '- 2: Partially covered; the predicted answer has some correct information, '
    'but also contains significant inaccuracies or missing key points.\n'
    '- 1: Incorrect; the predicted answer may be related to the ground truth '
    'answer, but most of the information is missing, or the predicted answer is '
    'not relevant to the question or in very poor quality.\n\n'
    'Output the score only, do not add more explanations.'
)


# ---------------------------------------------------------------------------
# PAUC helpers  (ported from ProactiveVideoQA/pauc/judge_llm_scores_openai.py)
# ---------------------------------------------------------------------------

def _area_under_line_ratio(points, max_x, max_y, omega=0.5, start_score=0.5):
    """Area enclosed between the step-function polyline and x-axis, normalised."""
    if not len(points):
        return 0.0
    points = sorted(points, key=lambda p: p[0])
    points = [(x * (1 - omega), y) for x, y in points]
    points.append([max_x, points[-1][1]])
    prev_y, prev_x, area = start_score, 0, 0.0
    max_area = max_x * max_y
    for x1, y1 in points:
        area += (x1 - prev_x) * prev_y
        prev_x, prev_y = x1, y1
    return area / max_area if max_area > 0 else 0.0


def _stat_metric(scored_examples, omega=0.5, max_score=2, start_score=0.5):
    """Compute mean PAUC across all turns in scored_examples."""
    auc_list = []
    for example in scored_examples:
        for turn in example['answer']:
            judge_scores = turn.get('judge_scores', {})
            if not len(judge_scores):
                auc_list.append(start_score / max_score)
                continue
            points = [
                [float(k) - turn['reply_timespan'][0], v]
                for k, v in judge_scores.items()
            ]
            max_x = turn['reply_timespan'][1] - turn['reply_timespan'][0]
            auc_list.append(
                _area_under_line_ratio(points, max_x, max_score, omega, start_score)
            )
    if not auc_list:
        return {'mean_auc': 0.0, 'num_videos': 0, 'num_turns': 0}
    return {
        'mean_auc': float(np.mean(auc_list)),
        'num_videos': len(scored_examples),
        'num_turns': len(auc_list),
    }


# ---------------------------------------------------------------------------
# expected_rounds  (must stay aligned with inference_streaming.py)
# ---------------------------------------------------------------------------

def _compute_expected_rounds(duration: float, answer: list, fps: float = 1.0) -> int:
    """Number of streaming rounds for one sample.

    Formula mirrors ProactiveVideoQA/online_model_inference/inference_streaming.py
    and generalises it to arbitrary fps:
        by_frames = int(duration * fps)
        by_reply  = ceil(min(duration, last_reply_end) * fps)
        return max(1, min(by_frames, by_reply))
    """
    last_reply_end = float(answer[-1]['reply_timespan'][1])
    by_frames = int(float(duration) * fps)
    by_reply = int(math.ceil(min(float(duration), last_reply_end) * fps))
    return max(1, min(by_frames, by_reply))


def _judge_proactive_prompt(judge, custom_id, prompt):
    """Run one judge prompt through the standard VLMEvalKit judge wrapper."""
    try:
        content = judge.generate(
            [dict(type='text', value=prompt)], dataset='ProactiveVideoQA'
        )
    except Exception as exc:
        logger.warning('Judge call failed (%s): %s', custom_id, exc)
        content = 'Failed to obtain answer via API.'
    return {'custom_id': custom_id, 'content': str(content)}


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ProactiveVideoQA(VideoBaseDataset):
    """ProactiveVideoQA benchmark — online streaming evaluation.

    Builds a TSV from anno.json, embeds all streaming metadata in the
    build_prompt message, and computes PAUC via a VLMEvalKit judge.
    """

    TYPE = 'Video-Proactive'
    MODALITY = 'VIDEO'

    @classmethod
    def supported_datasets(cls):
        return list(_TASK_MAP.keys())

    # ------------------------------------------------------------------
    # VideoBaseDataset required interface
    # ------------------------------------------------------------------

    def prepare_dataset(self, dataset, root_dir=None):
        task = _TASK_MAP[dataset]
        data_root = root_dir or os.environ.get(
            'PROACTIVE_VIDEOQA_ROOT', _DEFAULT_DATA_ROOT
        )
        video_dir = osp.join(data_root, task, 'videos')
        anno_path = osp.join(data_root, task, 'anno.json')

        if not osp.isdir(data_root):
            raise FileNotFoundError(
                f'ProactiveVideoQA root not found: {data_root}. '
                'Set environment variable PROACTIVE_VIDEOQA_ROOT.'
            )
        if not osp.isfile(anno_path):
            raise FileNotFoundError(
                f'ProactiveVideoQA annotation not found: {anno_path}'
            )

        tsv_dir = osp.join(LMUDataRoot(), 'ProactiveVideoQA')
        os.makedirs(tsv_dir, exist_ok=True)
        tsv_path = osp.join(tsv_dir, f'{task}.tsv')

        if not self._tsv_valid(tsv_path):
            self._build_tsv(anno_path, task, tsv_path)

        # EGO / VAD videos can be several minutes long; their per-sample JSON
        # predictions may exceed the xlsx 32 767-char cell limit.  Tell the user
        # to set PRED_FORMAT=tsv when running those tasks.
        if task in ('EGO', 'VAD') and os.environ.get('PRED_FORMAT', 'xlsx') == 'xlsx':
            logger.warning(
                f'ProactiveVideoQA_{task} has long videos (up to ~270 rounds). '
                'Prediction JSON may exceed xlsx cell limit (32767 chars). '
                'Recommended: set env var PRED_FORMAT=tsv before running.'
            )

        return {'root': video_dir, 'data_file': tsv_path}

    @staticmethod
    def _tsv_valid(tsv_path):
        if not osp.exists(tsv_path):
            return False
        try:
            data = load(tsv_path)
        except Exception:
            return False
        return _REQUIRED_COLS.issubset(set(data.columns))

    @staticmethod
    def _build_tsv(anno_path, task, tsv_path):
        with open(anno_path, encoding='utf-8') as f:
            records = json.load(f)
        rows = []
        for idx, rec in enumerate(records):
            conv = rec.get('conversation', [])
            question = conv[0]['content'] if conv else ''
            rows.append({
                'index': idx,
                'question_id': str(rec['question_id']),
                'video': str(rec['video']),
                'question': question,
                'duration': float(rec.get('duration', 0.0)),
                'task': task,
                'conversation_json': json.dumps(conv, ensure_ascii=False),
                'answer_json': json.dumps(rec['answer'], ensure_ascii=False),
            })
        dump(pd.DataFrame(rows), tsv_path)
        logger.info(f'Built ProactiveVideoQA TSV: {tsv_path} ({len(rows)} rows)')

    def build_prompt(self, line, video_llm=False):
        """Build a message list encoding all streaming metadata.

        Message structure:
            [{'type': 'video', 'value': video_path},
             {'type': 'text',  'value': '__PROACTIVE_STREAM_META__<json>'}]

        The 'video' entry lets VLMEvalKit's progress tracker see a real file.
        The 'text' entry carries all fields needed by StreamingVideoModel.
        Both types pass BaseModel's allowed_types check without modification.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
        if hasattr(line, 'to_dict'):
            line = line.to_dict()

        video_path = osp.join(self.data_root, str(line['video']))
        question = str(line['question'])
        duration = float(line['duration'])
        conversation = json.loads(str(line['conversation_json']))
        answer = json.loads(str(line['answer_json']))
        qid = str(line['question_id'])
        task = str(line.get('task', ''))

        # Extra user turns (TV subtitles, etc.)
        extra_turns = [
            {'time': float(c.get('time', 0)), 'content': c['content']}
            for c in conversation[1:]
            if c.get('role', 'user') == 'user' and c.get('content')
        ]

        target_fps = float(self.fps if getattr(self, 'fps', -1) > 0 else TARGET_FPS)
        target = _compute_expected_rounds(duration, answer, fps=target_fps)
        meta = {
            'video_path': video_path,
            'question': question,
            'extra_turns': extra_turns,
            'answer': answer,
            'duration': duration,
            'qid': qid,
            'task': task,
            'target_fps': target_fps,
            'target_rounds': target,
        }

        return [
            {'type': 'video', 'value': video_path},
            {
                'type': 'text',
                'value': STREAM_META_PREFIX + json.dumps(meta, ensure_ascii=False),
            },
        ]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, eval_file, **judge_kwargs):
        """Compute PAUC using VLMEvalKit's standard LLM judge interface.

        Steps:
          1. Parse per-step prediction records from the prediction column.
          2. Aggregate per-qid positive responses (answerable=='Yes').
          3. Generate judge prompts aligned to gold reply_timespans.
          4. Call the VLMEvalKit judge with cached progress.
          5. Fill judge_scores into scored_data for every gold qid.
          6. Compute PAUC at omega ∈ {0, 0.5, 1}.
          7. Save intermediate files and return result dict.
        """
        data = load(eval_file)

        # --- Step 1: parse per-step records from prediction column ---
        pred_dict = defaultdict(list)  # {qid: [(time, content), ...]}
        for _, row in data.iterrows():
            raw_pred = row.get('prediction', '')
            if not isinstance(raw_pred, str) or not raw_pred:
                continue
            try:
                parsed = json.loads(raw_pred)
            except json.JSONDecodeError:
                continue
            qid = str(parsed.get('question_id', row.get('question_id', '')))
            for rec in parsed.get('records', []):
                if (
                    rec.get('answerable', '').lower() == 'yes'
                    and rec.get('model_response')
                ):
                    end_time = float(rec['video_span'][1])
                    pred_dict[qid].append((end_time, rec['model_response']))

        # --- Step 2: build gold dict from TSV ---
        gold_dict = {}
        for _, row in self.data.iterrows():
            qid = str(row['question_id'])
            gold_dict[qid] = {
                'question_id': qid,
                'conversation': json.loads(str(row['conversation_json'])),
                'answer': json.loads(str(row['answer_json'])),
            }

        # --- Step 3: generate judge prompts ---
        # Mirrors create_openai_batch_input in judge_llm_scores_openai.py
        judge_prompts = []  # [(custom_id, prompt), ...]
        for qid, preds in pred_dict.items():
            if qid not in gold_dict:
                continue
            gold = gold_dict[qid]
            for turn_i, ans in enumerate(gold['answer']):
                ts0, ts1 = ans['reply_timespan']
                span_preds = [(t, s) for t, s in preds if ts0 <= t <= ts1]
                added = []
                for t, sent in span_preds:
                    if sent in added:
                        continue
                    added.append(sent)
                    user_msg = (
                        f"Question: {gold['conversation'][0]['content']}\n"
                        f"Ground Truth Answer: {ans['content']}\n"
                        f"Predicted Answer: {' '.join(added)}"
                    )
                    judge_prompts.append((f'{qid}={turn_i}={t}', user_msg))

        # --- Step 4: call the standard VLMEvalKit judge ---
        judge_model_name = judge_kwargs.pop(
            'model',
            os.environ.get(
                'PROACTIVE_JUDGE_MODEL',
                'qwen3-235b-a22b-thinking-2507',
            ),
        )
        nproc = judge_kwargs.pop('nproc', 8)
        judge_kwargs.setdefault('system_prompt', JUDGE_INSTRUCTION)
        judge_kwargs.setdefault('temperature', 0)
        judge_kwargs.setdefault('max_tokens', 32)
        judge = build_judge(model=judge_model_name, **judge_kwargs)

        if not judge_prompts:
            logger.warning('No positive predictions found; PAUC will be 0.')
            judge_results = []
        else:
            logger.info(
                f'Calling judge "{judge_model_name}" on {len(judge_prompts)} prompts '
                f'(nproc={nproc})...'
            )
            if not judge.working():
                raise RuntimeError(
                    f'Judge "{judge_model_name}" is unavailable. {DEBUG_MESSAGE}'
                )
            cache_file = get_intermediate_file_path(
                eval_file,
                f'_{judge_model_name.replace("/", "_")}_judge_cache',
                'pkl',
            )
            cache = load(cache_file) if osp.exists(cache_file) else {}
            pending = [
                task for task in judge_prompts
                if task[0] not in cache
                or 'Failed to obtain answer via API.' in cache[task[0]].get('content', '')
            ]
            if pending:
                track_progress_rich(
                    _judge_proactive_prompt,
                    tasks=[(judge, custom_id, prompt) for custom_id, prompt in pending],
                    nproc=nproc,
                    save=cache_file,
                    keys=[custom_id for custom_id, _ in pending],
                )
            cache = load(cache_file)
            judge_results = [cache[custom_id] for custom_id, _ in judge_prompts]

        # --- Step 5: build scored_data (all gold qids, empty judge_scores) ---
        scored_data = []
        for qid, ge in gold_dict.items():
            preds = pred_dict.get(qid, [])
            answers = []
            for ans in ge['answer']:
                ts0, ts1 = ans['reply_timespan']
                turn = dict(ans)
                turn['preds'] = [
                    (t, sent) for t, sent in preds if ts0 <= t <= ts1
                ]
                turn['judge_scores'] = {}
                answers.append(turn)
            entry = {
                'question_id': qid,
                'conversation': ge['conversation'],
                'answer': answers,
            }
            scored_data.append(entry)
        scored_by_qid = {e['question_id']: e for e in scored_data}

        # Fill in judge scores
        # custom_id format: "{qid}={turn_i}={pred_time}"
        # Score mapping: LLM outputs 1/2/3 → store 0/1/2 (subtract 1)
        for result in judge_results:
            custom_id = result['custom_id']
            content = result.get('content', '')
            matches = re.findall(r'(?<![0-9])([123])(?![0-9])', content)
            score = (int(matches[-1]) - 1) if matches else None

            # Decode custom_id; qid may contain '=' itself, so split from right
            parts = custom_id.rsplit('=', 2)
            if len(parts) != 3:
                continue
            qid, turn_i_str, pred_time_str = parts
            try:
                turn_i = int(turn_i_str)
                pred_time = float(pred_time_str)
            except ValueError:
                continue

            if qid not in scored_by_qid:
                continue
            entry = scored_by_qid[qid]
            if turn_i >= len(entry['answer']):
                continue
            turn = entry['answer'][turn_i]
            ts0, ts1 = turn['reply_timespan']
            if score is not None and ts0 <= pred_time <= ts1:
                turn['judge_scores'][pred_time] = score

        # Post-process: remove late-reply scores that are worse than last
        # in-related_timespan score (mirrors process_openai_batch_output)
        for entry in scored_data:
            for turn in entry['answer']:
                if len(turn['judge_scores']) <= 1:
                    continue
                related_end = turn.get(
                    'related_timespan', turn['reply_timespan']
                )[1]
                items = sorted(turn['judge_scores'].items())
                last_in_related = 0
                to_del = []
                for t, s in items:
                    if t <= related_end:
                        last_in_related = s
                    elif s < last_in_related:
                        to_del.append(t)
                for t in to_del:
                    del turn['judge_scores'][t]

        # --- Step 6: compute PAUC ---
        result_dict = {}
        for omega in [0, 0.5, 1]:
            res = _stat_metric(scored_data, omega=omega)
            result_dict[f'omega={omega}'] = round(res['mean_auc'] * 100, 2)
        result_dict['num_videos'] = len(scored_data)
        result_dict['num_turns'] = sum(len(e['answer']) for e in scored_data)

        # --- Step 7: save intermediate files ---
        judge_out = eval_file.rsplit('.', 1)[0] + '_judge.jsonl'
        scores_out = eval_file.rsplit('.', 1)[0] + '_scores.json'
        summary_out = eval_file.rsplit('.', 1)[0] + '_summary.json'

        with open(judge_out, 'w', encoding='utf-8') as f:
            for r in judge_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        with open(scores_out, 'w', encoding='utf-8') as f:
            json.dump(scored_data, f, ensure_ascii=False, indent=2)

        with open(summary_out, 'w', encoding='utf-8') as f:
            json.dump(
                {'judge': judge_model_name, 'dataset': self.dataset_name, **result_dict},
                f, ensure_ascii=False, indent=2,
            )

        task = self.dataset_name.split('_')[-1]
        logger.info(
            f'[{task}] PAUC: omega=0={result_dict["omega=0"]:.1f}  '
            f'omega=0.5={result_dict["omega=0.5"]:.1f}  '
            f'omega=1={result_dict["omega=1"]:.1f}  '
            f'({result_dict["num_videos"]} videos, {result_dict["num_turns"]} turns)'
        )
        logger.info(f'[{task}] Scores saved to: {summary_out}')

        return result_dict
