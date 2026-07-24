from ..smp import *
from .video_base import VideoBaseDataset
import subprocess
import re

class LVBench(VideoBaseDataset):

    """
    LVBench: An Extreme Long Video Understanding Benchmark

    本实现假设数据组织结构与官方仓库一致：
    - 根目录下包含：
      - data/video_info.meta.jsonl
      - videos/（由 video2dataset 下载得到的视频文件）

    在第一次使用时，会根据 `video_info.meta.jsonl` 自动生成
    一个 TSV 文件 `<dataset_name>.tsv`，字段包括：
    - index: 全局题目索引
    - video: 视频的 key（YouTube ID）
    - uid: 官方题目 ID
    - video_path: 相对根目录的视频路径，例如 `videos/00000/00000000.mp4`
    - question: 含选项的完整英文问题文本
    - answer: 正确选项字母（A/B/C/D）
    - question_type: 题目能力类别列表的 JSON 字符串
    - time_reference: 时间标注字符串
    """

    TYPE = 'Video-MCQ'
    SYS = ''

    def __init__(
        self,
        dataset: str = 'LVBench',
        nframe: int = 0,
        fps: float = -1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
    ):
        self.dataset_name = dataset
        self.frames_limit = frames_limit  # following Qwen3-VL, impose a cap of 2,048 frames per video
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )

    @classmethod
    def supported_datasets(cls):
        return ['LVBench']

    def prepare_dataset(
        self,
        dataset_name: str = 'LVBench',
        repo_id: str = '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/LVBench',
    ):
        """
        准备 LVBench 数据集。
        """

        dataset_path = repo_id

        def check_integrity(pth: str) -> bool:
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False

            data = load(data_file)
            if 'video_path' not in data:
                return False

            # 简单检查：随机抽样若干条，确认视频文件存在
            sample_paths = data['video_path'][:16]
            for vp in sample_paths:
                if not osp.exists(osp.join(pth, vp)):
                    return False
            return True

        if not osp.exists(dataset_path):
            raise FileNotFoundError(
                f'LVBench root directory not found: {dataset_path}. '
                f'Please set environment variable LVBENCH_ROOT to your LVBench root.'
            )

        if not check_integrity(dataset_path):

            def _build_video_key_to_path(root: str):
                """
                尝试从 video2dataset 的 parquet 元数据中恢复
                key (YouTube ID) 到本地路径的映射。
                若 parquet 不存在，则退化为按行顺序与 mp4 排序顺序一一对应。
                """
                videos_dir = osp.join(root, 'videos')
                parquet_file = osp.join(videos_dir, '00000.parquet')
                key_to_path = {}

                import pandas as pd

                df = pd.read_parquet(parquet_file)
                if df.empty:
                    raise RuntimeError(f'LVBench parquet file is empty: {parquet_file}')

                key_to_path = {
                    row["url"].split("watch?v=")[1]: osp.join("videos/00000", f"{row['key']}.mp4")
                    for _, row in df.iterrows()
                }

                return key_to_path

            def generate_tsv(pth: str):
                import pandas as pd

                meta_file = osp.join(pth, 'data', 'video_info.meta.jsonl')
                if not osp.exists(meta_file):
                    raise FileNotFoundError(
                        f'LVBench meta file not found: {meta_file}. '
                        f'Please make sure video_info.meta.jsonl is downloaded.'
                    )

                data_file = osp.join(pth, f'{dataset_name}.tsv')

                key_to_path = _build_video_key_to_path(pth)

                rows = []
                with open(meta_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        meta = json.loads(line)
                        key = meta['key']
                        qa_list = meta.get('qa', [])

                        video_rel_path = key_to_path[key]

                        video_path = osp.join(dataset_path, video_rel_path)
                        if not osp.exists(video_path):
                            continue

                        for qa in qa_list:
                            rows.append(
                                dict(
                                    video=key,
                                    uid=str(qa['uid']),
                                    video_path=video_rel_path,
                                    question=qa['question'],
                                    answer=str(qa['answer']).strip(),
                                    question_type=json.dumps(qa.get('question_type', [])),
                                    time_reference=qa.get('time_reference', ''),
                                )
                            )

                if len(rows) == 0:
                    raise RuntimeError('No QA entries parsed from video_info.meta.jsonl')

                df = pd.DataFrame(rows)
                df = df.assign(index=range(len(df)))
                df.to_csv(data_file, sep='\t', index=False)

            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video_path: str, video_llm: bool = False, verbose: bool = False):
        """
        根据 nframe 或 fps 从视频中抽帧，支持 AV1（使用 PyAV 解码）。
        """
        def get_video_info(path):
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets,avg_frame_rate",
                "-of", "json",
                path,
            ]
            data = subprocess.check_output(cmd)
            info = json.loads(data)["streams"][0]

            frames = int(info["nb_read_packets"])
            num, den = info["avg_frame_rate"].split("/")
            fps = float(num) / float(den)

            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ]
            duration = float(subprocess.check_output(cmd).decode().strip())

            return frames, fps, duration

        import av

        vid_path = osp.join(self.data_root, video_path)

        n_frames, fps, duration = get_video_info(vid_path)
        video_info = {'fps': fps, 'n_frames': n_frames, 'duration': duration}

        # ---- 计算抽帧 index ----
        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_path[:-4])
        elif self.fps > 0:
            total_duration = n_frames / fps
            required_frames = int(total_duration * self.fps)
            if required_frames > self.frames_limit: # 不能超过帧上限
                print(f"Warning: Video `{video_path}` requires {required_frames} frames with {self.fps} fps. Truncating to {self.frames_limit} frames.")
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_root = osp.join(self.frame_root, video_path[:-4])
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit)) for i in range(1, self.frames_limit + 1)]
            else:
                step_size = fps / self.fps
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(video_path[:-4], len(indices))
        else:
            raise ValueError('Either nframe > 0 or fps > 0 must be set.')

        needed = set(indices)

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )
        if need_extract:
            container = av.open(vid_path)
            stream = container.streams.video[0]

            frame_iter = enumerate(container.decode(stream))
            for idx, frame in tqdm(
                frame_iter,
                total=n_frames,
                desc=f"Extracting frames for {video_path}",
            ):
                if idx in needed:
                    out_idx = indices.index(idx)
                    pth = frame_paths[out_idx]
                    if not osp.exists(pth):
                        frame.to_image().save(pth)

            container.close()

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm: bool):
        """
        构建给 VLM 的多模态输入：
        - 若 video_llm=True，则直接传入整段视频；
        - 否则传入若干采样帧。
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video_path'], video_llm)

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            assert self.fps > 0
            # 如果视频帧数达到限制，则使用限制帧数和视频时长计算实际 FPS
            actual_fps = self.frames_limit / video_info['duration'] if len(frames) == self.frames_limit else self.fps
            message.append(dict(
                type='video', value=frames, sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            ))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        assert '(A)' in line['question']
        question = line['question'].split("(A)", 1)[0]
        options = "(A)" + line['question'].split("(A)", 1)[1]

        prompt = "\n".join([
            "Select the best answer to the following multiple-choice question based on the video.",
            "Respond with only the letter (A, B, C, or D) of the correct option.",
            f"Question: {question}",
            "Possible answer choices:",
            f"{options}",
            "The best answer is:",
        ])
        # print(prompt)
        message.append(dict(type='text', value=prompt))
        return message

    @staticmethod
    def _polish_answer(ans: str) -> str:
        if not ans:
            return ''
        
        text = ans.strip().upper()

        # 1. 匹配格式化选项，如 (A), [A], A), A.
        medium_patterns = [
            r'\(([ABCD])\)',         # (A)
            r'\[([ABCD])\]',         # [A]
            r'\b([ABCD])[\.\)]',     # A.  或  A)
            r'[:\s]\s*([ABCD])\b',   # : A   或   空格 A
        ]
        for p in medium_patterns:
            m = re.search(p, text)
            if m:
                return m.group(1)

        # 3. 最后兜底：找独立单字母 A/B/C/D（避免匹配单词）
        # 要求前后是非字母，即不是 Apple、Cat 这种单词内部字母
        m = re.search(r'(?<![A-Z])([ABCD])(?![A-Z])', text)
        if m:
            return m.group(1)

        return ''

    # It returns a dictionary
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        直接复刻 LVBench 官方的评测逻辑：
        - 读取 `<model>_LVBench.xlsx/tsv` 或 json，其中包含 prediction 列；
        - 与 TSV 中的 answer 比较，统计整体与各 question_type 的准确率。
        同时：
        - 将归一化后的预测结果写回到 eval_file 的 `pred` 列（若存在则覆盖）；
        - 将各类别以及 overall 的准确率写入一个额外的 `_metrics.csv` 文件。
        """
        from ..smp.file import get_file_extension, get_intermediate_file_path
        from .utils import build_judge, DEBUG_MESSAGE, extract_answer_from_item
        import warnings

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        # 读取预测文件
        data = load(eval_file)
        # 兼容 Excel / TSV：确保 prediction/answer/question_type 存在
        assert 'prediction' in data and 'answer' in data and 'question_type' in data, (
            'LVBench evaluation requires `prediction`, `answer`, and `question_type` columns.'
        )

        # === 1. 生成归一化后的预测列，并写回到 eval_file ===
        # 若指定了 judge model，则使用 LLM Judge 从 prediction 中抽取最终选项；
        # 否则沿用原先的规则解析逻辑（_polish_answer）。
        model_name = judge_kwargs.get('model', 'exact_matching')
        if model_name == 'exact_matching' or model_name is None:
            judge_model = None
        else:
            judge_model = None
            try:
                judge_model = build_judge(**judge_kwargs)
                if hasattr(judge_model, 'working') and (not judge_model.working()):
                    warnings.warn('Judge model is not working properly, will use rule-based parsing for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    judge_model = None
            except Exception as e:
                warnings.warn(f'Failed to build judge model ({model_name}), will use rule-based parsing: {type(e)}: {e}')
                warnings.warn(DEBUG_MESSAGE)
                judge_model = None

        def _normalize_pred_rule(x):
            if pd.isna(x):
                return ''
            return cls._polish_answer(str(x))

        def _parse_question_options(q: str):
            """
            从包含 (A)/(B)/(C)/(D) 的题干中解析出：
            - question_text: 纯问题（不含选项）
            - options: dict, e.g. {'A': '...', 'B': '...'}
            解析失败则 options 为空 dict。
            """
            if not isinstance(q, str):
                q = '' if pd.isna(q) else str(q)
            if '(A)' not in q:
                return q.strip(), {}
            matches = list(re.finditer(r'\(([ABCD])\)', q))
            if len(matches) < 2:
                return q.strip(), {}
            question_text = q[:matches[0].start()].strip()
            options = {}
            for i, m in enumerate(matches):
                ch = m.group(1)
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(q)
                opt_text = q[start:end].strip()
                if opt_text:
                    options[ch] = opt_text
            return question_text, options

        if judge_model is None:
            # Rule-based parsing path
            data['pred'] = data['prediction'].apply(_normalize_pred_rule)
        else:
            # LLM-judge parsing path (with caching)
            assert 'question' in data, (
                'LVBench LLM-judge evaluation requires `question` column to parse (A)/(B)/(C)/(D) options.'
            )

            safe_model_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(model_name))
            judge_cache_file = get_intermediate_file_path(
                eval_file, f'_judge_{safe_model_name}', target_format='pkl'
            )
            judge_cache = {}

            preds = []
            updated_cache = False
            for _, row in data.iterrows():
                idx = row.get('index', None)
                if idx is not None and idx in judge_cache:
                    preds.append(judge_cache[idx])
                    continue

                pred_raw = row.get('prediction', '')
                if pred_raw is None or (isinstance(pred_raw, float) and pd.isna(pred_raw)) or pd.isna(pred_raw):
                    opt = ''
                else:
                    pred_text = str(pred_raw)
                    q_text, options = _parse_question_options(row.get('question', ''))
                    # 若无法解析选项，则退化为原规则解析
                    if not options:
                        opt = cls._polish_answer(pred_text)
                    else:
                        item = {'question': q_text if q_text else str(row.get('question', '')), 'prediction': pred_text}
                        for ch, txt in options.items():
                            item[ch] = txt
                        try:
                            opt = extract_answer_from_item(judge_model, item, dataset_name='LVBench')['opt']
                        except Exception:
                            opt = ''

                opt = (opt or '').strip().upper()
                # LVBench 的 GT 为 A/B/C/D；其它输出（如 Z）按“无法解析”处理，保持与旧逻辑一致（不计入统计）
                if opt not in ['A', 'B', 'C', 'D']:
                    opt = ''

                preds.append(opt)
                if idx is not None:
                    judge_cache[idx] = opt
                    updated_cache = True

            if updated_cache:
                dump(judge_cache, judge_cache_file)
            data['pred'] = preds

        # 直接覆盖保存到原 eval_file（格式由后缀决定）
        dump(data, eval_file)

        total_qa_num = 0
        right_num = 0
        category_right = defaultdict(int)
        category_total = defaultdict(int)

        for _, row in data.iterrows():
            # 使用已归一化后的 `pred` 列进行评测
            pred = row.get('pred', '')
            if not isinstance(pred, str):
                pred = str(pred) if not pd.isna(pred) else ''
            pred = pred.strip().upper()
            if not pred:
                continue

            gt = str(row['answer']).strip().upper()
            if not gt:
                continue

            equal = (pred == gt)

            qtype = row.get('question_type', '')
            # question_type 在 TSV 中以 JSON 字符串形式存储
            qtypes = []
            if isinstance(qtype, str):
                s = qtype.strip()
                if s:
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            qtypes = [str(x) for x in parsed]
                    except Exception:
                        # 退化处理：用逗号或分号分隔
                        for part in s.replace(';', ',').split(','):
                            part = part.strip().strip('"').strip("'")
                            if part:
                                qtypes.append(part)
            elif isinstance(qtype, (list, tuple)):
                qtypes = [str(x) for x in qtype]

            if len(qtypes) == 0:
                # 若没有类别信息，也至少统计到 overall 中
                category_total['__NO_CATEGORY__'] += 1
                if equal:
                    category_right['__NO_CATEGORY__'] += 1

            for c in qtypes:
                category_total[c] += 1
                if equal:
                    category_right[c] += 1

            if equal:
                right_num += 1
            total_qa_num += 1

        category_acc = {}
        for key in category_total:
            if category_total[key] > 0:
                category_acc[key] = category_right[key] / category_total[key]
            else:
                category_acc[key] = 0.0

        acc = float(right_num) / total_qa_num if total_qa_num > 0 else 0.0
        category_acc.update({'acc': acc})

        # === 2. 将最终评测结果写入一个独立的 CSV 文件 ===
        # 使用统一的中间文件命名工具，并显式指定为 csv 格式
        metrics_file = get_intermediate_file_path(eval_file, '_metrics', target_format='csv')
        metrics_rows = []
        for key, v in category_acc.items():
            # 若有统计信息，则一并写出；否则只写准确率
            metrics_rows.append(
                dict(
                    category=key,
                    accuracy=float(v),
                    right=int(category_right.get(key, 0)),
                    total=int(category_total.get(key, 0)),
                )
            )
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(metrics_file, index=False)

        return category_acc

