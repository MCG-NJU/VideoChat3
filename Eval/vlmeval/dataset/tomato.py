import json
import os
import re
import warnings
from typing import Optional, Dict, Any, List, Tuple

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE


def _safe_model_name(x: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', str(x))


def _normalize_text(s: str) -> str:
    if s is None:
        return ''
    if isinstance(s, float) and pd.isna(s):
        return ''
    s = str(s).strip().lower()
    s = re.sub(r'[^0-9a-z\s]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _ordered_choice_keys(choices: dict) -> List[str]:
    if not choices:
        return []
    return sorted(str(k).strip().upper() for k in choices.keys() if k is not None and str(k).strip())


def _extract_mcq_letter(text: str, allowed: str = '') -> str:
    """
    从模型输出中尽量鲁棒地抽取 MCQ 选项字母。
    allowed: 合法选项字母集合（如 "ABCDEFG"）；为空时默认 A-F。
    """
    if text is None:
        return ''
    if isinstance(text, float) and pd.isna(text):
        return ''
    s = str(text).strip().upper()
    if not s:
        return ''

    if not allowed:
        allowed = 'ABCDEF'
    allowed = ''.join(sorted(set(str(allowed).upper())))
    if not allowed:
        return ''

    # Prefer the last explicit "Answer: X"
    m = re.findall(rf'ANSWER\s*[:：]\s*([{allowed}])\b', s)
    if m:
        return m[-1]

    # (A) / [A] / A. / A)
    for p in [
        rf'\([{allowed}]\)',
        rf'\[[{allowed}]\]',
        rf'\b([{allowed}])[\.\)]\b',
        rf'[:\s]\s*([{allowed}])\b',
    ]:
        m2 = re.findall(p, s)
        if m2:
            return m2[-1]

    # fallback: any standalone option letter not inside a word
    m3 = re.findall(rf'(?<![A-Z])([{allowed}])(?![A-Z])', s)
    return m3[-1] if m3 else ''


def _answer_index_to_letter(ans_idx: int) -> str:
    try:
        i = int(ans_idx)
    except Exception:
        return ''
    if 0 <= i < 26:
        return chr(ord('A') + i)
    return ''


class TOMATO(VideoBaseDataset):
    """
    TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models

    该实现假设用户本地存在 TOMATO 官方 repo 的目录结构：
    - <TOMATO_ROOT>/data/*.json
    - <TOMATO_ROOT>/videos/{human,object,simulated}/*.mp4

    在第一次使用时，会根据 6 个任务 JSON 自动生成 `<TOMATO_ROOT>/TOMATO.tsv`。
    """

    TYPE = 'Video-MCQ'
    SYS = (
        "Carefully watch the video and answer the following multiple-choice question.\n"
        "You MUST answer with the option letter only (e.g., A, B, C, D, E, or F)."
    )

    # Following Qwen3-VL practice: cap frames per video for FPS sampling
    def __init__(
        self,
        dataset: str = 'TOMATO',
        nframe: int = 0,
        fps: float = -1,
        frames_limit: int = 2048,
        min_pixels: int = 28 * 28,
        max_pixels: int = 448 * 448,
        total_pixels: int = 32000 * 2 * 4 * 14 * 14,
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
        return ['TOMATO']

    @staticmethod
    def _video_key(video_rel_path: str) -> str:
        # Safe directory name for cached frames
        s = (video_rel_path or '').replace('\\', '/')
        s = s.replace('/', '__')
        s = s.replace('.mp4', '')
        return s

    @staticmethod
    def _resolve_video_rel_path(root: str, demo_type: str, key: str) -> str:
        """
        Try to resolve the mp4 path relative to TOMATO root.
        """
        demo_type = (demo_type or '').strip().lower()
        key = (key or '').strip()
        candidates = []
        if demo_type:
            candidates.append(f'videos/{demo_type}/{key}.mp4')
        # fallback search order
        for dt in ['human', 'object', 'simulated']:
            candidates.append(f'videos/{dt}/{key}.mp4')
        for rel in candidates:
            if osp.exists(osp.join(root, rel)):
                return rel
        # return the primary guess even if not found (user may have different layout)
        return candidates[0] if candidates else f'videos/{key}.mp4'

    def prepare_dataset(self, dataset_name: str = 'TOMATO', repo_id: Optional[str] = None):
        """
        Prepare TOMATO dataset from a local checkout.

        - 优先使用入参 repo_id
        - 其次读取环境变量 TOMATO_ROOT
        """
        dataset_path = (
            repo_id
            or os.environ.get('TOMATO_ROOT', None)
            or '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/TOMATO'
        )
        dataset_path = str(dataset_path)

        if not osp.exists(dataset_path):
            raise FileNotFoundError(
                f"TOMATO root directory not found: {dataset_path}. "
                f"Please set environment variable TOMATO_ROOT to your TOMATO repo path."
            )

        def check_integrity(pth: str) -> bool:
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            try:
                df = load(data_file)
            except Exception:
                return False
            if 'video' not in df or 'question' not in df or 'answer' not in df:
                return False
            # sample-check a few videos exist
            sample = df.head(min(16, len(df)))
            for vp in sample['video']:
                if not osp.exists(osp.join(pth, vp)):
                    return False
            return True

        if not check_integrity(dataset_path):
            data_dir = osp.join(dataset_path, 'data')
            if not osp.exists(data_dir):
                raise FileNotFoundError(
                    f"TOMATO data directory not found: {data_dir}. "
                    f"Please make sure the official repo meta files exist."
                )

            task_files = [
                'count.json',
                'direction.json',
                'rotation.json',
                'shape&trend.json',
                'velocity&frequency.json',
                'visual_cues.json',
            ]
            missing = [f for f in task_files if not osp.exists(osp.join(data_dir, f))]
            if missing:
                raise FileNotFoundError(f"Missing TOMATO meta files under {data_dir}: {missing}")

            rows = []
            for jf in task_files:
                task_name = osp.splitext(jf)[0]
                meta = json.load(open(osp.join(data_dir, jf), 'r', encoding='utf-8'))
                # meta is a dict keyed by numeric strings
                for local_idx, ex in meta.items():
                    if ex is None:
                        continue
                    question = str(ex.get('question', '') or '').strip()
                    options = ex.get('options', []) or []
                    # build choices mapping A-F...
                    choices = {}
                    for i, opt in enumerate(options):
                        letter = _answer_index_to_letter(i)
                        if not letter:
                            continue
                        choices[letter] = str(opt).strip()

                    ans_letter = _answer_index_to_letter(ex.get('answer', -1))
                    demo_type = ex.get('demonstration_type', '') or ''
                    key = ex.get('key', '') or ''
                    video_rel = self._resolve_video_rel_path(dataset_path, demo_type, key)

                    variation = ex.get('variation', {}) or {}
                    row = dict(
                        id=f'{task_name}:{local_idx}',
                        key=key,
                        task=task_name,
                        motion_type=str(ex.get('motion_type', '') or task_name),
                        question_type='multiple-choice',
                        question=question,
                        choices=json.dumps(choices, ensure_ascii=False),
                        answer=ans_letter,
                        video=video_rel,
                        demonstration_type=str(demo_type),
                        video_source_url=str(ex.get('video_source_url', '') or ''),
                        note=str(ex.get('note', '') or ''),
                        counterfactual=int(variation.get('counterfactual', 0) or 0),
                        composite=int(variation.get('composite', 0) or 0),
                        zoom=int(variation.get('zoom', 0) or 0),
                        first_person=int(variation.get('first_person', 0) or 0),
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
        if not osp.exists(vid_path):
            raise FileNotFoundError(f'TOMATO video not found: {vid_path}')

        import decord
        
        # 针对这个损坏的视频，直接使用 OpenCV 抽帧，因为 decord 在遇到损坏帧时会死锁
        if '0217-00.mp4' or '0803-00.mp4' in vid_path:
            import cv2
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = (n_frames / fps) if fps > 0 else 0.0
            video_info = {
                'fps': fps,
                'n_frames': n_frames,
                'duration': duration,
            }
            
            # ... (复用原有的 indices 计算逻辑)
            vkey = self._video_key(video_rel_path)
            if self.nframe > 0 and self.fps < 0:
                step_size = n_frames / (self.nframe + 1)
                indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
                frame_paths = self.frame_paths(vkey)
            elif self.fps > 0:
                required_frames = int(duration * self.fps) if duration > 0 else (
                    int((n_frames / fps) * self.fps) if fps > 0 and n_frames > 0 else 0
                )
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
                indices = [max(0, n_frames // 2)]
                frame_paths = self.frame_paths_fps(vkey, len(indices)) if self.fps > 0 else self.frame_paths(vkey)[:1]

            if n_frames > 0 and len(indices) > 0:
                max_idx = int(n_frames) - 1
                indices = [min(max(0, int(x)), max_idx) for x in indices]

            need_extract = self.check_extracted_frames and (
                not np.all([osp.exists(p) for p in frame_paths])
            )
            if need_extract:
                images = []
                last_valid_frame = None
                for frame_idx in tqdm(indices, desc=f"Reading frames for {video_rel_path} (OpenCV)"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame_bgr = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        last_valid_frame = pil_image
                        images.append(pil_image)
                    else:
                        print(f"\n[Warning] Frame {frame_idx} is corrupted or missing. Using last valid frame.")
                        if last_valid_frame is not None:
                            images.append(last_valid_frame)
                        else:
                            # 如果第一帧就坏了，创建一个黑色的图
                            images.append(Image.new('RGB', (1280, 720), (0, 0, 0)))
                
                for im, pth in tqdm(
                    zip(images, frame_paths),
                    total=len(frame_paths),
                    desc=f"Saving frames for {video_rel_path}",
                ):
                    if not osp.exists(pth):
                        im.save(pth)
            cap.release()
            return frame_paths, indices, video_info

        # 正常的视频继续使用 decord
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
            required_frames = int(duration * self.fps) if duration > 0 else (
                int((n_frames / fps) * self.fps) if fps > 0 and n_frames > 0 else 0
            )
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
                desc=f"Saving frames for {video_rel_path}",
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

        question = str(line.get('question', '')).strip()
        try:
            choices = json.loads(line.get('choices', '{}') or '{}')
        except Exception:
            choices = {}

        opt_lines = [
            f"{k}. {str(choices[k]).strip()}"
            for k in _ordered_choice_keys(choices)
        ]
        options_str = "\n".join(opt_lines)

        prompt = f"""You will be provided with {len(frames)} separate frames uniformly sampled
from a video, the frames are provided in chronological order of the
video. Analyze these frames and provide the answer to the question
about the video content. Answer the multiple-choice question about
the video content.

You must use these frames to answer the multiple-choice question; do not
rely on any external knowledge or commonsense.

<question> {question} </question>

<options> {options_str} </options>

Even if the information in these separate frames is not enough to answer
the question, PLEASE TRY YOUR BEST TO GUESS AN ANSWER WHICH YOU THINK
WOULD BE THE MOST POSSIBLE ONE BASED ON THE QUESTION.

DO NOT GENERATE ANSWER SUCH AS ‘NOT POSSIBLE TO DETERMINE.’"""
        message.append(dict(type='text', value=prompt))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        TOMATO 默认评测：MCQ 准确率（A-F 字母匹配）。
        额外输出按 task / demonstration_type / variation 的 breakdown。
        """
        from .utils.multiple_choice import extract_answer_from_item

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )
        data = load(eval_file)

        required_cols = ['prediction', 'answer']
        for c in required_cols:
            assert c in data.columns, f"TOMATO evaluation requires column `{c}` in {eval_file}"

        # Optional: LLM judge for robust option extraction.
        # If a judge model is provided and API keys are available, use it; otherwise fallback to rule-based parsing.
        judge_model_name = judge_kwargs.get('model', 'exact_matching')
        judge = None
        if judge_model_name == 'exact_matching':
            judge = None
        elif gpt_key_set():
            try:
                judge = build_judge(**judge_kwargs)
                if hasattr(judge, 'working') and (not judge.working()):
                    warnings.warn('Judge model is not working properly, will use rule-based extraction for TOMATO evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    judge = None
            except Exception as e:
                warnings.warn(f'Failed to initialize judge model ({judge_model_name}): {e}. Will use rule-based extraction.')
                judge = None
        else:
            warnings.warn('API key is not set properly, will use rule-based extraction for TOMATO evaluation')
            judge = None

        # optional breakdown columns
        for c in ['task', 'demonstration_type', 'counterfactual', 'composite', 'zoom', 'first_person', 'choices']:
            if c not in data.columns:
                data[c] = ''

        parsed_opt = []
        scores = []
        extracted = []

        judge_opts: List[str] = []
        judge_logs: List[str] = []

        for _, row in data.iterrows():
            pred = row.get('prediction', '')
            try:
                choices = json.loads(row.get('choices', '{}') or '{}')
            except Exception:
                choices = {}
            allowed_letters = ''.join(_ordered_choice_keys(choices))

            gt = str(row.get('answer', '')).strip().upper()
            gt = gt if gt in allowed_letters else gt[:1]

            pred_letter = ''

            # 1) Judge-based extraction (semantic matching) if available
            if judge is not None:
                try:
                    # Build an item compatible with `extract_answer_from_item`
                    item: Dict[str, Any] = {
                        'question': str(row.get('question', '') or ''),
                        'prediction': str(pred),
                    }
                    for k, v in (choices or {}).items():
                        kk = str(k).strip().upper()
                        if kk in allowed_letters:
                            item[kk] = str(v)
                    res = extract_answer_from_item(judge, item, dataset_name='TOMATO')
                    cand = str(res.get('opt', '') or '').strip().upper()
                    log = str(res.get('log', '') or '')
                    judge_logs.append(log)
                    # `extract_answer_from_item` may randomly guess when judge fails;
                    # treat such cases as invalid and fallback to rule-based extraction for determinism.
                    is_random_guess = ('randomly generate' in log.lower()) or ('random' in log.lower() and 'generate' in log.lower())
                    if (not is_random_guess) and (cand in allowed_letters):
                        pred_letter = cand
                        judge_opts.append(cand)
                    else:
                        judge_opts.append('')
                except Exception as e:
                    # Judge failed for this sample; fallback to rule-based extraction
                    judge_opts.append('')
                    judge_logs.append(f'JUDGE_ERROR: {e}')

            # 2) Rule-based extraction (original behavior)
            if not pred_letter:
                pred_letter = _extract_mcq_letter(pred, allowed=allowed_letters)

                # fallback: numeric index -> letter
                if not pred_letter and allowed_letters:
                    max_idx = len(allowed_letters) - 1
                    m = re.findall(rf'(?<!\d)([0-{max_idx}])(?!\d)', str(pred))
                    if m:
                        pred_letter = _answer_index_to_letter(int(m[-1]))

                # fallback: match option text
                if not pred_letter:
                    pred_norm = _normalize_text(pred)
                    best = ''
                    for k, v in (choices or {}).items():
                        if not k:
                            continue
                        if _normalize_text(v) and _normalize_text(v) in pred_norm:
                            best = str(k).strip().upper()
                    pred_letter = best

            parsed_opt.append(pred_letter)
            extracted.append(pred_letter)
            scores.append(int(bool(pred_letter) and bool(gt) and pred_letter == gt))

        data['score'] = scores
        data['parsed_option'] = parsed_opt
        data['extracted_answer'] = extracted
        if judge is not None:
            # expose judge outputs for debugging; keep optional to avoid breaking old pipelines
            data['judge_option'] = judge_opts
            data['judge_log'] = judge_logs

        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(data, score_file)

        def agg(df: 'pd.DataFrame') -> dict:
            total = int(len(df))
            hit = int(df['score'].sum()) if total > 0 else 0
            acc = (hit / total * 100.0) if total > 0 else 0.0
            return dict(total=total, hit=hit, acc=acc)

        res: Dict[str, Dict[str, Any]] = {}
        for task, sub_df in data.groupby('task', dropna=False):
            key = f"task:{str(task) if task is not None else ''}"
            res[key] = agg(sub_df)
        for dt, sub_df in data.groupby('demonstration_type', dropna=False):
            key = f"demo:{str(dt) if dt is not None else ''}"
            res[key] = agg(sub_df)

        # Variation breakdown (treat missing as 0)
        for vcol in ['counterfactual', 'composite', 'zoom', 'first_person']:
            try:
                vals = sorted(list(set([int(x) if str(x).strip() != '' else 0 for x in data[vcol]])))
            except Exception:
                vals = []
            for v in vals:
                sub_df = data[(pd.to_numeric(data[vcol], errors='coerce').fillna(0).astype(int) == int(v))]
                res[f"var:{vcol}={v}"] = agg(sub_df)

        res['Overall'] = agg(data)

        df_out = pd.DataFrame(res)
        out_path = get_intermediate_file_path(eval_file, '_metrics', 'csv')
        dump(df_out, out_path)
        return df_out


