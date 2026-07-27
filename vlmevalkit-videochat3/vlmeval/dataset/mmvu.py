import json
import os
import re
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE


FAIL_MSG = 'Failed to obtain answer via API.'


def _safe_model_name(x: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', str(x))


def _normalize_text(s: str) -> str:
    if s is None:
        return ''
    if isinstance(s, float) and pd.isna(s):
        return ''
    s = str(s).strip().lower()
    # keep alnum and spaces, collapse spaces
    s = re.sub(r'[^0-9a-z\s]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _extract_after_markers(text: str, markers: List[str]) -> str:
    if not text:
        return ''
    lower = text.lower()
    for m in markers:
        idx = lower.rfind(m.lower())
        if idx >= 0:
            return text[idx + len(m):].strip()
    return text.strip()


def _extract_mcq_letter(text: str) -> str:
    """
    Robustly extract an option letter in [A-E] from model output.
    """
    if text is None:
        return ''
    if isinstance(text, float) and pd.isna(text):
        return ''
    s = str(text).strip().upper()
    if not s:
        return ''

    # Prefer the last explicit "Answer: X" pattern
    m = re.findall(r'ANSWER\s*[:：]\s*([A-E])\b', s)
    if m:
        return m[-1]

    # (A) / [A] / A. / A)
    for p in [r'\(([A-E])\)', r'\[([A-E])\]', r'\b([A-E])[\.\)]\b', r'[:\s]\s*([A-E])\b']:
        m2 = re.findall(p, s)
        if m2:
            return m2[-1]

    # fallback: any standalone A-E not inside a word
    m3 = re.findall(r'(?<![A-Z])([A-E])(?![A-Z])', s)
    return m3[-1] if m3 else ''


def _map_hf_video_url_to_local_rel(url: str) -> str:
    """
    HF URL: .../videos/<Subject>/<id>.mp4  ->  videos/<Subject>/<id>.mp4
    """
    if not isinstance(url, str):
        return ''
    # find "/videos/" segment
    m = re.search(r'/videos/(.+?\.mp4)(?:\?|$)', url)
    if m:
        return f"videos/{m.group(1)}"
    # fallback: if already a relative path
    if url.endswith('.mp4') and ('videos/' in url or 'videos\\' in url):
        idx = url.replace('\\', '/').find('videos/')
        return url.replace('\\', '/')[idx:]
    return url


class MMVU(VideoBaseDataset):
    """
    MMVU: Measuring Expert-Level Multi-Discipline Video Understanding

    This implementation assumes a local MMVU repository layout (as provided by the official repo):
    - <MMVU_ROOT>/data/validation.json
    - <MMVU_ROOT>/videos/<Subject>/<N>.mp4

    At first use, it will generate a TSV file `<dataset_name>.tsv` in the root directory.

    Open-ended evaluation (`evaluate`) must use LLM Judge: `exact_matching` is not allowed;
    default judge model is `gpt-4o-0806` when `--judge` is not specified.
    Requires `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, or `QWEN_API_KEY`.
    """

    TYPE = 'Video-VQA'
    SYS = (
        "Carefully watch the video and answer the question.\n"
        "Follow the required answer format strictly."
    )

    # Following Qwen3-VL practice: cap frames per video for FPS sampling
    def __init__(
        self,
        dataset: str = 'MMVU',
        nframe: int = 0,
        fps: float = -1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=768 * 768,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames: bool = True,
    ):
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels

    @classmethod
    def supported_datasets(cls):
        return ['MMVU']

    @staticmethod
    def _video_key(video_rel_path: str) -> str:
        # Safe directory name for cached frames
        s = (video_rel_path or '').replace('\\', '/')
        s = s.replace('/', '__')
        s = s.replace('.mp4', '')
        return s

    def prepare_dataset(
        self,
        dataset_name: str = 'MMVU',
        repo_id: Optional[str] = None,
    ):
        """
        Prepare MMVU dataset from a local checkout.
        You can set MMVU_ROOT env var to point to the official repo directory.
        """
        dataset_path = repo_id or os.environ.get('MMVU_ROOT', None) or '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/MMVU'
        dataset_path = str(dataset_path)

        if not osp.exists(dataset_path):
            raise FileNotFoundError(
                f"MMVU root directory not found: {dataset_path}. "
                f"Please set environment variable MMVU_ROOT to your MMVU repo path."
            )

        def check_integrity(pth: str) -> bool:
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            try:
                df = load(data_file)
            except Exception:
                return False
            if 'video' not in df or 'question' not in df:
                return False
            # sample-check a few videos exist
            sample = df.head(min(16, len(df)))
            for vp in sample['video']:
                if not osp.exists(osp.join(pth, vp)):
                    return False
            return True

        if not check_integrity(dataset_path):
            val_json = osp.join(dataset_path, 'data', 'validation.json')
            if not osp.exists(val_json):
                raise FileNotFoundError(
                    f"MMVU validation file not found: {val_json}. "
                    f"Please make sure the official repo data is present."
                )

            examples = json.load(open(val_json, 'r', encoding='utf-8'))
            rows = []
            for ex in examples:
                subject = (ex.get('metadata', {}) or {}).get('subject', '')
                video_rel = _map_hf_video_url_to_local_rel(ex.get('video', ''))
                # ensure local path exists; if not, still keep it (user may have different root)
                row = dict(
                    id=ex.get('id', ''),
                    subject=subject,
                    question_type=ex.get('question_type', ''),
                    question=ex.get('question', ''),
                    choices=json.dumps(ex.get('choices', {}) or {}, ensure_ascii=False),
                    answer=ex.get('answer', ''),
                    video=video_rel,
                    youtube_url=ex.get('youtube_url', ''),
                    textbook=(ex.get('metadata', {}) or {}).get('textbook', ''),
                )
                rows.append(row)

            df = pd.DataFrame(rows)
            df = df.assign(index=range(len(df)))
            tsv_path = osp.join(dataset_path, f'{dataset_name}.tsv')
            df.to_csv(tsv_path, sep='\t', index=False)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def save_video_frames(self, video_rel_path: str, video_llm: bool = False, verbose: bool = False):
        """
        Sample frames from <data_root>/<video_rel_path>.
        Returns: (frame_paths, indices, video_info)
        """
        vid_path = osp.join(self.data_root, video_rel_path)
        import decord
        vid = decord.VideoReader(vid_path)
        fps = float(vid.get_avg_fps())
        n_frames = int(len(vid))
        duration = (n_frames / fps) if fps > 0 else 0.0
        video_info = {
            'fps': fps,
            'n_frames': n_frames,
            'duration': duration,
        }

        vkey = self._video_key(video_rel_path)
        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(vkey)
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                warnings.warn(
                    f"Video `{video_rel_path}` requires {required_frames} frames at {self.fps} fps. "
                    f"Truncating to {self.frames_limit} frames."
                )
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_root = osp.join(self.frame_root, vkey)
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [
                    osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit))
                    for i in range(1, self.frames_limit + 1)
                ]
            else:
                step_size = fps / self.fps if self.fps > 0 else 1.0
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(vkey, len(indices))
        else:
            raise ValueError('Either nframe > 0 or fps > 0 must be set.')

        if len(indices) == 0:
            # Degenerate case: extract at least one middle frame
            indices = [max(0, n_frames // 2)]
            frame_paths = self.frame_paths_fps(vkey, len(indices)) if self.fps > 0 else self.frame_paths(vkey)[:1]

        # clamp indices to valid range (avoid occasional out-of-range due to rounding)
        if n_frames > 0 and len(indices) > 0:
            max_idx = int(n_frames) - 1
            indices = [min(max(0, int(x)), max_idx) for x in indices]

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )
        if need_extract:
            images = []
            for frame_idx in tqdm(indices, desc=f"Reading frames for {video_rel_path}"):
                images.append(Image.fromarray(vid[frame_idx].asnumpy()))
            for im, pth in tqdm(
                zip(images, frame_paths),
                total=len(frame_paths),
                desc=f"Saving frames for {video_rel_path}"
            ):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm: bool):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm=video_llm)

        # Start with system instruction
        message = [dict(type='text', value=self.SYS, role='system')]

        if video_llm:
            assert self.fps > 0
            actual_fps = (
                self.frames_limit / video_info['duration']
                if len(frames) == self.frames_limit and video_info['duration'] > 0
                else self.fps
            )
            message.append(dict(
                type='video', value=frames, sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            ))
        else:
            message.extend(dict(type='image', value=im) for im in frames)

        qtype = str(line.get('question_type', '')).strip().lower()
        question = str(line.get('question', '')).strip()

        if qtype == 'multiple-choice':
            try:
                choices = json.loads(line.get('choices', '{}') or '{}')
            except Exception:
                choices = {}
            # Keep A-E order
            opt_lines = []
            for k in ['A', 'B', 'C', 'D', 'E']:
                v = choices.get(k, '')
                if v is None:
                    v = ''
                opt_lines.append(f"{k}. {str(v).strip()}")
            options_str = "\n".join(opt_lines)
            prompt = (
                f"Question: {question}\n\n"
                f"Options:\n{options_str}\n\n"
                "Please answer with the option letter only (A, B, C, D, or E).\n"
                "Answer:"
            )
            # prompt = (
            #     f"Question: {question}\n\n"
            #     f"Options:\n{options_str}\n\n"
            #     "Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. "
            #     'Conclude by stating the final answer using the following format: "Therefore, the final answer is: $LETTER" (without quotes), where $LETTER is one of the options. Think step by step before answering.'
            # )
        else:
            prompt = (
                f"Question: {question}\n\n"
                "Please output the final short answer.\n"
                "Answer:"
            )
            # prompt = (
            #     f"Question: {question}\n\n"
            #     "Answer the given question step by step. Begin by explaining your reasoning process clearly.\n"
            #     'Conclude by stating the final answer using the following format: "Therefore, the final answer is: "Answer: $ANSWER" (without quotes), where $ANSWER is the final answer of the question. Think step by step before answering.'
            # )

        message.append(dict(type='text', value=prompt))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        MCQ: rule-based option extraction + exact matching.
        Open-ended: must use LLM Judge (official-style semantic grading); `exact_matching` is not allowed.
        """
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        data = load(eval_file)
        required_cols = ['prediction', 'answer', 'question_type']
        for c in required_cols:
            assert c in data.columns, f"MMVU evaluation requires column `{c}` in {eval_file}"

        # subject is optional but recommended for breakdown
        if 'subject' not in data.columns:
            data['subject'] = ''

        has_open = (
            data['question_type']
            .astype(str)
            .str.strip()
            .str.lower()
            .ne('multiple-choice')
            .any()
        )

        jk = dict(judge_kwargs)
        model_name = jk.pop('model', 'gpt-4o-0806')
        judge_model = None

        if has_open:
            if model_name in (None, '', 'exact_matching'):
                raise ValueError(
                    'MMVU 开放式评测必须使用 LLM Judge，不允许 `exact_matching`。'
                    '请通过 `--judge` / `judge_kwargs["model"]` 指定判分模型（例如 `gpt-4o-0806`、'
                    '`qwen3-235b-a22b-instruct-2507`、`qwen3-235b-a22b-thinking-2507`）；'
                    '未指定时默认使用 `gpt-4o-0806`。'
                )
            if not gpt_key_set():
                raise RuntimeError(
                    'MMVU 开放式评测需要可用的 Judge API：请设置 OPENAI_API_KEY、AZURE_OPENAI_API_KEY '
                    '或 QWEN_API_KEY 之一（与 vlmeval.smp.gpt_key_set 一致）。'
                )
            try:
                judge_model = build_judge(model=model_name, **jk)
            except TypeError as e:
                raise TypeError(
                    f'MMVU: 调用 build_judge 失败（常与重复传入 model 有关）: {e}. '
                    f'请只通过 judge 参数传入 model，勿在其它参数中重复指定。'
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"MMVU: 无法创建 LLM Judge (model={model_name!r}): {e}"
                ) from e
            if hasattr(judge_model, 'working') and (not judge_model.working()):
                raise RuntimeError(
                    'MMVU: Judge API 不可用（working() 为 False）。' + DEBUG_MESSAGE
                )

        tmp_file = get_intermediate_file_path(
            eval_file, f'_mmvu_judge_{_safe_model_name(model_name)}', 'pkl'
        )
        cache = {} if not osp.exists(tmp_file) else load(tmp_file)
        if not isinstance(cache, dict):
            cache = {}

        def llm_judge_one_openended(question: str, gt_answer: str, pred_text: str) -> dict:
            """
            Return dict(extracted_answer=str, correct=bool)
            """
            instruction = (
                "Evaluate whether the model's final answer is correct by comparing it to the ground-truth answer.\n"
                "The final answer does NOT need to match word-for-word, but must be explicitly and unambiguously equivalent.\n"
                "You MUST output a JSON object with keys: extracted_answer (string), correct (boolean).\n"
            )
            user_prompt = (
                f"Question: {question}\n\n"
                f"Ground Truth Answer: {gt_answer}\n\n"
                f"Model Response to the Question: {pred_text}\n"
            )
            resp = judge_model.generate(instruction + "\n" + user_prompt)
            try:
                jsons = list(extract_json_objects(resp))
                if len(jsons) == 0:
                    raise ValueError('no json found')
                out = jsons[-1]
                extracted = str(out.get('extracted_answer', '') or '')
                correct = bool(out.get('correct', False))
                return dict(extracted_answer=extracted, correct=correct)
            except Exception:
                # fallback: mark incorrect
                return dict(extracted_answer='', correct=False)

        scores = []
        extracted_answers = []
        parsed_mcq = []

        # def extract_mc_answer(output: str) -> str:
        #     pattern = r"Therefore,\s*the final answer is:\s*([A-Z])"
        #     match = re.search(pattern, output)
        #     return match.group(1) if match else output

        # def extract_openended_answer(output: str) -> str:
        #     pattern = r"Therefore,\s*the final answer is:\s*Answer:\s*(.+)"
        #     match = re.search(pattern, output, re.DOTALL)
        #     return match.group(1).strip() if match else output

        for _, row in data.iterrows():
            qtype = str(row.get('question_type', '')).strip().lower()
            pred = row.get('prediction', '')

            # if qtype == 'multiple-choice':
            #     pred = extract_mc_answer(pred)
            # else:
            #     pred = extract_openended_answer(pred)

            if qtype == 'multiple-choice':
                gt = str(row.get('answer', '')).strip().upper()
                pred_letter = _extract_mcq_letter(pred)
                parsed_mcq.append(pred_letter)
                extracted_answers.append(pred_letter)
                scores.append(int(pred_letter == gt and gt in ['A', 'B', 'C', 'D', 'E']))
                continue

            # open-ended (LLM Judge required; validated above)
            question = str(row.get('question', '') or '')
            gt = str(row.get('answer', '') or '')
            pred_text = str(pred) if not (isinstance(pred, float) and pd.isna(pred)) else ''

            idx = row.get('index', None)
            key = str(idx) if idx is not None else str(row.get('id', ''))
            if key in cache:
                judged = cache[key]
            else:
                judged = llm_judge_one_openended(question, gt, pred_text)
                cache[key] = judged
                dump(cache, tmp_file)
            extracted_answers.append(str(judged.get('extracted_answer', '') or ''))
            parsed_mcq.append('')
            scores.append(int(bool(judged.get('correct', False))))

        data['score'] = scores
        data['extracted_answer'] = extracted_answers
        data['parsed_option'] = parsed_mcq

        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(data, score_file)

        # Aggregate results
        def agg(df: 'pd.DataFrame') -> dict:
            total = int(len(df))
            hit = int(df['score'].sum()) if total > 0 else 0
            acc = (hit / total * 100.0) if total > 0 else 0.0
            return dict(total=total, hit=hit, acc=acc)

        res = {}
        # by subject
        for sub, sub_df in data.groupby('subject', dropna=False):
            sub_key = str(sub) if sub is not None else ''
            res[sub_key if sub_key else '__NO_SUBJECT__'] = agg(sub_df)
        # by question type
        for qt, qt_df in data.groupby('question_type', dropna=False):
            qt_key = str(qt) if qt is not None else ''
            res[f"qtype:{qt_key if qt_key else '__NO_QTYPE__'}"] = agg(qt_df)
        res['Overall'] = agg(data)

        df_out = pd.DataFrame(res)
        out_path = get_intermediate_file_path(eval_file, '_metrics', 'csv')
        dump(df_out, out_path)
        return df_out

