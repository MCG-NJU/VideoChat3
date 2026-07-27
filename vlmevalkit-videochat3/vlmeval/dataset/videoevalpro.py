import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from ..utils import can_infer
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.videoevalpro import build_videoevalpro_grader_prompt, parse_videoevalpro_grader_response

FAIL_MSG = 'Failed to obtain answer via API.'


def _video_frame_key(video_rel: str) -> str:
    s = (video_rel or '').replace('\\', '/')
    return s.replace('/', '__').replace('.mp4', '').replace('.mkv', '').replace('.webm', '')


def _load_options_list(options_cell) -> list:
    if options_cell is None or (isinstance(options_cell, float) and pd.isna(options_cell)):
        return []
    if isinstance(options_cell, (list, tuple)):
        return [str(x) for x in options_cell]
    s = str(options_cell).strip()
    if not s or s == '[]':
        return []
    try:
        v = json.loads(s)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        try:
            import ast
            v = ast.literal_eval(s)
            return [str(x) for x in v] if isinstance(v, list) else []
        except Exception:
            return []


def _choices_dict_from_options(opts_list: list) -> dict:
    out = {}
    for o in opts_list:
        m = re.match(r'^([A-Za-z])\.\s*(.*)$', str(o).strip(), flags=re.DOTALL)
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
    return out


def _infer_mcq_letter(pred: str, options_cell) -> str:
    opts = _load_options_list(options_cell)
    choices = _choices_dict_from_options(opts)
    if not choices:
        return ''
    pred_s = str(pred).strip()
    if not pred_s or FAIL_MSG in pred_s:
        return ''
    inf = can_infer(pred_s, choices)
    if inf and isinstance(inf, str) and inf.upper() in choices:
        return inf.upper()
    allowed = ''.join(sorted(choices.keys()))
    m = re.search(rf'[{re.escape(allowed)}]', pred_s.upper())
    return m.group(0) if m else ''


def _grade_one(judge_model, question: str, gold: str, pred: str) -> int:
    prompt = build_videoevalpro_grader_prompt(question, gold, pred)
    out = judge_model.generate(prompt)
    letter = parse_videoevalpro_grader_response(out)
    if letter == 'A':
        return 1
    if letter in ('B', 'C'):
        return 0
    # Judge 未给出可解析的 A/B/C：按错误计分（不再做子串匹配）
    return 0


class VideoEvalPro(VideoBaseDataset):
    """
    VideoEval-Pro：长视频开放式短答评测（展示题干，不展示选项；官方判别常用 GPT-4o 类 Judge）。

    多选题（展示选项、字母作答、与 `mcq_letter` 比对）请使用 `VideoEvalProMCQ`。

    默认从本地仓库读取 `VideoEval-Pro.tsv` 或 `data/*.parquet` 与 `videos_filtered/*.mp4`。
    可通过环境变量 `VIDEOEVALPRO_ROOT` 覆盖根目录。

    开放式评测（evaluate）必须使用 LLM Judge：禁止 `model='exact_matching'`；未指定 `model` 时默认 `gpt-4o-0806`。
    需配置 `OPENAI_API_KEY` / `AZURE_OPENAI_API_KEY` 或 `QWEN_API_KEY`。
    """

    TYPE = 'Video-VQA'
    SYS = (
        'Carefully watch the video and answer the question based on what you see. '
        'Keep the answer short and concise.'
    )

    def __init__(
        self,
        dataset: str = 'VideoEval-Pro',
        nframe: int = 0,
        fps: float = -1,
        frames_limit: int = 2048,
        min_pixels: int = 28 * 28,
        max_pixels: int = 448 * 448,
        total_pixels: int = 32000 * 2 * 4 * 14 * 14,
        check_extracted_frames: bool = True,
    ):
        self.dataset_name = dataset
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )

    @classmethod
    def supported_datasets(cls):
        return ['VideoEval-Pro']

    def prepare_dataset(
        self,
        dataset_name: str = 'VideoEval-Pro',
        repo_id: Optional[str] = None,
    ):
        root = repo_id or os.environ.get(
            'VIDEOEVALPRO_ROOT',
            '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/VideoEval-Pro',
        )
        root = str(root)
        if not osp.exists(root):
            raise FileNotFoundError(
                f'VideoEval-Pro root not found: {root}. '
                f'Set VIDEOEVALPRO_ROOT to your checkout path.'
            )

        data_file = osp.join(root, f'{dataset_name}.tsv')
        videos_dir = osp.join(root, 'videos_filtered')

        def check_integrity() -> bool:
            if not osp.exists(data_file):
                return False
            try:
                df = load(data_file)
            except Exception:
                return False
            need = {'index', 'video', 'question', 'answer'}
            if not need.issubset(set(df.columns)):
                return False
            for vp in df['video'].head(min(8, len(df))):
                if not osp.exists(osp.join(root, vp)):
                    return False
            return True

        if not check_integrity():
            from glob import glob

            pq_list = sorted(glob(osp.join(root, 'data', '*.parquet')))
            if not pq_list:
                raise FileNotFoundError(
                    f'No parquet under {osp.join(root, "data")}. '
                    f'Please download VideoEval-Pro annotations (HF `data/` shards).'
                )
            if not osp.isdir(videos_dir):
                raise FileNotFoundError(
                    f'Expected video directory missing: {videos_dir}'
                )

            parts = []
            for p in pq_list:
                parts.append(pd.read_parquet(p))
            df = pd.concat(parts, ignore_index=True)
            df = df.rename(columns={'answer': 'mcq_letter', 'answer_text': 'answer'})
            df['video'] = df['video'].apply(
                lambda x: f'videos_filtered/{x}' if not str(x).startswith('videos_filtered') else str(x)
            )

            def _opts_cell(x):
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return '[]'
                if isinstance(x, str):
                    return x
                try:
                    return json.dumps([str(y) for y in list(x)], ensure_ascii=False)
                except Exception:
                    return json.dumps([], ensure_ascii=False)

            if 'options' in df:
                df['options'] = df['options'].apply(_opts_cell)
            else:
                df['options'] = '[]'

            for c in ('meta', 'source', 'qa_subtype', 'qa_type', 'mcq_letter'):
                if c not in df:
                    df[c] = ''

            df = df.assign(index=range(len(df)))
            keep = [
                'index', 'video', 'question', 'answer', 'mcq_letter', 'options',
                'meta', 'source', 'qa_subtype', 'qa_type',
            ]
            df = df[[c for c in keep if c in df.columns]]
            df.to_csv(data_file, sep='\t', index=False)

        return dict(data_file=data_file, root=root)

    def save_video_frames(self, video_rel: str, video_llm: bool = False, verbose: bool = False):
        vid_path = osp.join(self.data_root, video_rel)
        import decord
        vid = decord.VideoReader(vid_path)
        fps = float(vid.get_avg_fps())
        n_frames = int(len(vid))
        duration = (n_frames / fps) if fps > 0 else 0.0
        video_info = {'fps': fps, 'n_frames': n_frames, 'duration': duration}
        vkey = _video_frame_key(video_rel)

        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(vkey)
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                print(
                    f'Warning: Video `{video_rel}` requires {required_frames} frames at {self.fps} fps. '
                    f'Truncating to {self.frames_limit} frames.'
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

        if n_frames > 0 and len(indices) > 0:
            max_idx = int(n_frames) - 1
            indices = [min(max(0, int(x)), max_idx) for x in indices]

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )
        if need_extract:
            images = []
            for frame_idx in tqdm(indices, desc=f'Reading frames for {video_rel}'):
                images.append(Image.fromarray(vid[frame_idx].asnumpy()))
            for im, pth in tqdm(zip(images, frame_paths), total=len(frame_paths), desc=f'Saving frames for {video_rel}'):
                if not osp.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm: bool):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm=video_llm)

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
            for im in frames:
                message.append(dict(type='image', value=im))

        q = str(line['question']).strip()
        message.append(dict(type='text', value=q))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        data = load(eval_file)
        assert 'prediction' in data.columns and 'answer' in data.columns, (
            'VideoEval-Pro evaluation requires `prediction` and `answer` (open-ended gold) columns.'
        )

        jk = dict(judge_kwargs)
        nproc = int(jk.pop('nproc', 1))
        # 与官方 VideoEval-Pro 默认一致；未传 --judge 时使用 gpt-4o-2024-08-06 的映射名
        model_name = jk.pop('model', 'gpt-4o-0806')

        if model_name in (None, '', 'exact_matching'):
            raise ValueError(
                'VideoEval-Pro 开放式评测必须使用 LLM Judge，不允许 `exact_matching`。'
                '请通过 `--judge` / `judge_kwargs["model"]` 指定判分模型（例如 `gpt-4o-0806`、'
                '`qwen3-235b-a22b-instruct-2507`、`qwen3-235b-a22b-thinking-2507`）；'
                '未指定时默认使用 `gpt-4o-0806`。'
            )
        if not gpt_key_set():
            raise RuntimeError(
                'VideoEval-Pro 评测需要可用的 Judge API：请设置 OPENAI_API_KEY、AZURE_OPENAI_API_KEY '
                '或 QWEN_API_KEY 之一（与 vlmeval.smp.gpt_key_set 一致）。'
            )

        try:
            judge_model = build_judge(model=model_name, **jk)
        except TypeError as e:
            raise TypeError(
                f'VideoEval-Pro: 调用 build_judge 失败（常与重复传入 model 有关）: {e}. '
                f'请只通过 judge 参数传入 model，勿在其它参数中重复指定。'
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"VideoEval-Pro: 无法创建 LLM Judge (model={model_name!r}): {e}"
            ) from e

        if hasattr(judge_model, 'working') and (not judge_model.working()):
            raise RuntimeError(
                'VideoEval-Pro: Judge API 不可用（working() 为 False）。' + DEBUG_MESSAGE
            )

        safe_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(model_name))
        tmp_file = get_intermediate_file_path(eval_file, f'_vep_judge_{safe_name}', 'pkl')
        cache = load(tmp_file) if osp.exists(tmp_file) else {}

        scores = [None] * len(data)
        indices_list = list(data['index'])

        def work(idx_row):
            idx, row = idx_row
            pred = row.get('prediction', '')
            if pred is None or (isinstance(pred, float) and pd.isna(pred)):
                pred = ''
            pred = str(pred)
            if FAIL_MSG in pred:
                return idx, 0
            q = str(row.get('question', ''))
            gold = str(row.get('answer', ''))
            if idx in cache:
                return idx, int(cache[idx])
            s = _grade_one(judge_model, q, gold, pred)
            return idx, s

        rows_to_run = [(int(row['index']), row) for _, row in data.iterrows()]
        missing = [(i, r) for i, r in rows_to_run if i not in cache]

        if len(missing):
            if nproc <= 1:
                for tup in tqdm(missing, desc='VideoEval-Pro judge'):
                    i, s = work(tup)
                    cache[i] = s
                    dump(cache, tmp_file)
            else:
                lock = threading.Lock()

                def _wrapped(tup):
                    return work(tup)

                with ThreadPoolExecutor(max_workers=nproc) as ex:
                    futures = [ex.submit(_wrapped, t) for t in missing]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc='VideoEval-Pro judge'):
                        i, s = fut.result()
                        with lock:
                            cache[i] = s
                            dump(cache, tmp_file)

        for j, idx in enumerate(indices_list):
            scores[j] = int(cache.get(int(idx), 0))

        data['score'] = scores
        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(data, score_file)

        def agg(sub):
            tot = len(sub)
            hit = int(sub['score'].sum()) if tot else 0
            acc = (hit / tot * 100.0) if tot else 0.0
            return dict(total=tot, hit=hit, acc=round(acc, 3))

        metrics = {'Overall': agg(data)}
        if 'source' in data.columns:
            for name, sub in data.groupby('source', dropna=False):
                key = str(name) if str(name) else '__NO_SOURCE__'
                metrics[f'source:{key}'] = agg(sub)
        if 'qa_type' in data.columns:
            for name, sub in data.groupby('qa_type', dropna=False):
                key = str(name) if str(name) else '__NO_QA_TYPE__'
                metrics[f'qa_type:{key}'] = agg(sub)
        if 'qa_subtype' in data.columns:
            for name, sub in data.groupby('qa_subtype', dropna=False):
                key = str(name) if str(name) else '__NO_QA_SUBTYPE__'
                metrics[f'qa_subtype:{key}'] = agg(sub)

        out_path = get_intermediate_file_path(eval_file, '_metrics', 'csv')
        metrics_rows = [dict(metric=k, **v) for k, v in metrics.items()]
        pd.DataFrame(metrics_rows).to_csv(out_path, index=False)
        return metrics


class VideoEvalProMCQ(VideoEvalPro):
    """
    VideoEval-Pro 多选题子集：在 prompt 中展示选项，模型输出选项字母；
    评测将预测解析为字母后与 `mcq_letter` 比对（默认精确匹配，无需 LLM Judge）。

    与 `VideoEvalPro`（开放式短答 + LLM Judge）共用同一份 `VideoEval-Pro.tsv` 与视频缓存目录
    （`dataset='VideoEval-Pro'`），便于两种评测分开跑、复用已抽帧结果。
    """

    TYPE = 'Video-MCQ'
    SYS = (
        'Carefully watch the video and answer the multiple-choice question based on what you see. '
        'Respond with only the single capital letter of your choice (A-F, depending on the listed options).'
    )

    def build_prompt(self, line, video_llm: bool):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm=video_llm)

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
            for im in frames:
                message.append(dict(type='image', value=im))

        q = str(line['question']).strip()
        opts = _load_options_list(line.get('options'))
        opt_block = '\n'.join(opts) if opts else ''
        if opt_block:
            text = f'{q}\n{opt_block}\nAnswer with only the option letter.'
        else:
            text = f'{q}\nAnswer with only the option letter.'
        message.append(dict(type='text', value=text))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        data = load(eval_file)
        assert 'prediction' in data.columns and 'mcq_letter' in data.columns, (
            'VideoEval-Pro MCQ 评测需要 `prediction` 与 `mcq_letter` 列。'
        )

        jk = dict(judge_kwargs)
        model_name = jk.pop('model', 'exact_matching')

        judge_model = None
        if model_name not in (None, '', 'exact_matching'):
            if not gpt_key_set():
                raise RuntimeError(
                    'VideoEval-Pro MCQ：使用 LLM 抽取选项时需要可用的 Judge API '
                    '（OPENAI_API_KEY / AZURE_OPENAI_API_KEY / QWEN_API_KEY）。'
                )
            try:
                judge_model = build_judge(model=model_name, **jk)
            except Exception as e:
                raise RuntimeError(
                    f'VideoEval-Pro MCQ: 无法创建 LLM Judge (model={model_name!r}): {e}'
                ) from e
            if hasattr(judge_model, 'working') and (not judge_model.working()):
                raise RuntimeError(
                    'VideoEval-Pro MCQ: Judge API 不可用（working() 为 False）。' + DEBUG_MESSAGE
                )

        from .utils.videomme import extract_option

        scores = []
        for _, row in data.iterrows():
            pred = row.get('prediction', '')
            if pred is None or (isinstance(pred, float) and pd.isna(pred)):
                pred = ''
            pred = str(pred)
            gold = str(row.get('mcq_letter', '')).strip().upper()[:1]
            if not gold:
                scores.append(0)
                continue
            if FAIL_MSG in pred:
                scores.append(0)
                continue

            if judge_model is None:
                pred_letter = _infer_mcq_letter(pred, row.get('options'))
                scores.append(1 if pred_letter == gold else 0)
            else:
                item = row.to_dict()
                opts = _load_options_list(item.get('options'))
                item['candidates'] = str(opts)
                extract_pred = extract_option(judge_model, item, 'Video-MME')
                scores.append(1 if str(extract_pred).strip().upper()[:1] == gold else 0)

        data['score'] = scores
        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(data, score_file)

        def agg(sub):
            tot = len(sub)
            hit = int(sub['score'].sum()) if tot else 0
            acc = (hit / tot * 100.0) if tot else 0.0
            return dict(total=tot, hit=hit, acc=round(acc, 3))

        metrics = {'Overall': agg(data)}
        if 'source' in data.columns:
            for name, sub in data.groupby('source', dropna=False):
                key = str(name) if str(name) else '__NO_SOURCE__'
                metrics[f'source:{key}'] = agg(sub)
        if 'qa_type' in data.columns:
            for name, sub in data.groupby('qa_type', dropna=False):
                key = str(name) if str(name) else '__NO_QA_TYPE__'
                metrics[f'qa_type:{key}'] = agg(sub)
        if 'qa_subtype' in data.columns:
            for name, sub in data.groupby('qa_subtype', dropna=False):
                key = str(name) if str(name) else '__NO_QA_SUBTYPE__'
                metrics[f'qa_subtype:{key}'] = agg(sub)

        out_path = get_intermediate_file_path(eval_file, '_metrics', 'csv')
        metrics_rows = [dict(metric=k, **v) for k, v in metrics.items()]
        pd.DataFrame(metrics_rows).to_csv(out_path, index=False)
        return metrics
