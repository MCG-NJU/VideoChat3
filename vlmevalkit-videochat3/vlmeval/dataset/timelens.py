import json
import os
import re
import warnings
from typing import List, Tuple

from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .video_concat_dataset import ConcatVideoDataset
from .utils.video_pyav import ffprobe_video_info
from .video_meta_cache import (
    ensure_video_meta_json,
    load_video_meta,
    sample_fps_from_meta,
    sample_frame_count_from_meta,
)
from .utils.vtg_reason import extract_answer_from_tags, wrap_vtg_prompt_with_reason


def extract_time(paragraph: str) -> List[Tuple[float, float]]:
    """
    从模型输出中提取时间戳对。
    该函数来自 TimeLens 官方仓库 timelens/utils.py
    """
    paragraph = str(paragraph).lower()
    timestamps: List[Tuple[float, float]] = []

    # 1) 首先检查 HH:MM:SS(.xx) 或 MM:SS(.xx) 格式
    time_regex = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b")
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]
    
    if time_matches:
        # 转换为秒
        converted: List[float] = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:  # HH:MM:SS.xx 格式
                h, m = map(int, parts[:2])
                s = float(parts[2])
                converted.append(float(h * 3600 + m * 60 + s))
            elif len(parts) == 2:  # MM:SS.xxx 格式
                m = int(parts[0])
                s = float(parts[1])
                converted.append(float(m * 60 + s))
        timestamps = [(converted[i], converted[i + 1]) for i in range(0, len(converted), 2)]

    # 2) 检查 "m - n" 或 "m to n" 格式
    if len(timestamps) == 0:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",  # 18.5 - 23.0
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",  # 18.5 to 23.0
        ]
        for pat in patterns:
            ms = re.findall(pat, paragraph)
            if ms:
                timestamps = [(float(s), float(e)) for s, e in ms]
                break

    # 3) 其他格式，例如:
    # Starting time: 0.8 seconds. Ending time: 1.1 seconds
    # The start time for this event is 0 seconds, and the end time is 12 seconds.
    if len(timestamps) == 0:
        num_re = re.compile(r"\b(\d+\.\d+|\d+)\b")  # 时间格式 (例如, 18, 18.5)
        nums = re.findall(num_re, paragraph)
        nums = nums[: len(nums) // 2 * 2]
        timestamps = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]

    return [(float(s), float(e)) for s, e in timestamps]


def iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """计算两个时间区间的 IoU"""
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    denom = max1 - min0
    if denom <= 0:
        return 0.0
    return max(min1 - max0, 0.0) / denom


class TimeLensBase(VideoBaseDataset):
    """
    TimeLens 时序定位数据集基类。
    用于单个子数据集的实现。
    """

    TYPE = "Video-Temporal-Grounding"
    GROUNDER_PROMPT = (
        "Please find the visual event described by the sentence '{query}', determining its starting and ending times. "
        "The format should be: 'The event happens in <start time> - <end time> seconds'."
    )

    GROUNDER_PROMPT_TEXT_TIMESTAMP = (
        "You are given a video with multiple frames. "
        "The numbers before each video frame indicate its sampling timestamp (in seconds). "
    ) + GROUNDER_PROMPT

    # 子类需要设置这个属性
    SUBSET_NAME = None

    def __init__(
        self,
        dataset,
        use_frame_time=True,
        nframe=-1,
        fps=2,
        frames_limit=2048,
        min_pixels=28*28,
        max_pixels=448*448,
        total_pixels=32000*2*4*14*14,
        check_extracted_frames=True,
        reason=False,
    ):
        """
        Args:
            dataset (str): 数据集名称
            use_frame_time (bool): 是否在提示中附带采样帧对应的时间戳。
            nframe (int): 采样帧数（与 fps 互斥）。
            fps (int): 采样帧率（与 nframe 互斥）。
            frames_limit (int): 最大帧数限制。
            min_pixels (int): 视频帧最小像素数。
            max_pixels (int): 视频帧最大像素数。
            total_pixels (int): 视频总像素数限制。
            check_extracted_frames (bool): 为 True 时检查抽帧缓存路径，若未齐则解码并补写；
                为 False 时不做存在性检查，也不解码、不写盘，假定 ``frame_paths`` 对应帧均已预抽好。
            reason (bool): 为 True 时在 prompt 末尾追加 thinking/answer 格式说明，评测时从 <answer> 标签提取。
        """
        self.reason = reason
        self.use_frame_time = use_frame_time
        self.dataset_name = dataset
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels

        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )
        self._video_meta = load_video_meta(getattr(self, '_video_meta_path', None))
        self._video_duration_cache = {}

    def _frame_paths(self, key, num_frames):
        frame_root = osp.join(self.frame_root, key)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def _frame_paths_fps(self, key, num_frames, fps):
        frame_root = osp.join(self.frame_root, key)
        os.makedirs(frame_root, exist_ok=True)
        return [
            osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, fps))
            for i in range(1, num_frames + 1)
        ]

    def _video_duration(self, video_key, fallback=None):
        if video_key in self._video_duration_cache:
            return self._video_duration_cache[video_key]
        meta = getattr(self, '_video_meta', {}).get(str(video_key), {})
        if meta:
            try:
                duration = float(meta.get('duration', 0.0))
                if duration > 0:
                    self._video_duration_cache[video_key] = duration
                    return duration
            except Exception:
                pass
        duration = float(fallback or 0.0)
        vid_path = osp.join(self.data_root, f"{video_key}.mp4")
        try:
            import decord

            vid = decord.VideoReader(vid_path)
            video_fps = float(vid.get_avg_fps())
            duration = float(len(vid) / video_fps) if video_fps > 0 else duration
        except Exception:
            try:
                _, _, probed_duration = ffprobe_video_info(vid_path)
                duration = float(probed_duration)
            except Exception:
                pass
        if duration > 0:
            self._video_duration_cache[video_key] = duration
        return duration

    def _format_grounder_prompt(self, query: str) -> str:
        prompt = self.GROUNDER_PROMPT.format(query=query)
        if self.reason:
            prompt = wrap_vtg_prompt_with_reason(prompt)
        return prompt

    def prepare_dataset(
        self,
        dataset,
        json_path=None,
        root_dir=None,
    ):
        """
        准备单个 TimeLens 子数据集
        
        Args:
            dataset: 数据集名称
            json_path: JSON 注释文件路径
            root_dir: 视频根目录
        """
        default_bench_root = os.environ.get("TIMELENS_BENCH_ROOT")
        if not default_bench_root:
            raise EnvironmentError(
                "TIMELENS_BENCH_ROOT is not set. Point it to the TimeLens-Bench directory."
            )
        
        if json_path is None:
            json_path = osp.join(default_bench_root, f"{self.SUBSET_NAME}-timelens.json")
        
        if root_dir is None:
            root_dir = osp.join(default_bench_root, "videos", self.SUBSET_NAME)
        
        tsv_path = json_path.replace(".json", "_new.tsv")
        meta_path = tsv_path.replace(".tsv", "_video_meta.json")
        self._video_meta_path = meta_path
        
        # 如果 TSV 已存在，直接使用
        if osp.exists(tsv_path):
            try:
                df = load(tsv_path)
                video_ids = sorted(set(str(x) for x in df['video'].dropna().tolist()))
                ensure_video_meta_json(
                    meta_path,
                    video_ids,
                    lambda video_key: osp.join(root_dir, f'{video_key}.mp4'),
                )
            except Exception as e:
                warnings.warn(f'[TimeLens] failed to ensure video meta json: {e}')
            return dict(data_file=tsv_path, root=root_dir)

        assert osp.exists(json_path), f"{json_path} 不存在，请检查数据路径。"
        
        # 加载 JSON 数据
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 解析数据
        rows = []
        index = 0
        for vid, anno in raw_data.items():
            duration = float(anno['duration'])
            spans = anno['spans']
            queries = anno['queries']
            
            for span, query in zip(spans, queries):
                # 清理 query
                query = re.sub(r"\s+", " ", str(query)).strip().strip(".").strip()
                
                rows.append({
                    'index': index,
                    'video': vid,
                    'duration': duration,
                    'question': query,
                    'answer': [span],  # 答案格式为 [[start, end]]
                    'gt_start': float(span[0]),
                    'gt_end': float(span[1]),
                    'task_mode': 'miou',
                })
                index += 1
        
        df = pd.DataFrame(rows)
        df.to_csv(tsv_path, sep="\t", index=False)
        try:
            video_ids = sorted(set(str(x) for x in df['video'].dropna().tolist()))
            ensure_video_meta_json(
                meta_path,
                video_ids,
                lambda video_key: osp.join(root_dir, f'{video_key}.mp4'),
            )
        except Exception as e:
            warnings.warn(f'[TimeLens] failed to ensure video meta json: {e}')
        
        return dict(data_file=tsv_path, root=root_dir)

    def save_video_frames(self, video_path, key):
        import decord

        vid_path = osp.join(self.data_root, f"{video_path}.mp4")
        vid = decord.VideoReader(vid_path)
        video_fps = vid.get_avg_fps()
        n_frames = len(vid)
        # 视频原始信息
        video_info = {
            "fps": video_fps,
            "n_frames": n_frames,
            "duration": n_frames / video_fps,
        }

        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self._frame_paths(key, len(indices))
            # 保存采样信息
            video_info["sample_fps"] = video_fps / step_size if step_size > 0 else video_fps
            video_info["sample_n_frame"] = len(indices)
        else:
            # 使用 fps 采样
            required_frames = max(1, int(video_info["duration"] * self.fps))
            if required_frames > self.frames_limit:
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_root = osp.join(self.frame_root, key)
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit)) for i in range(1, self.frames_limit + 1)]
                # 保存采样信息
                video_info["sample_fps"] = self.frames_limit / video_info["duration"]
                video_info["sample_n_frame"] = self.frames_limit
            else:
                step_size = video_fps / self.fps
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self._frame_paths_fps(key, len(indices), self.fps)
                # 保存采样信息
                video_info["sample_fps"] = self.fps
                video_info["sample_n_frame"] = required_frames

        if self.check_extracted_frames:
            all_cached = np.all([osp.exists(p) for p in frame_paths])
        else:
            # 预抽帧：不检查路径、不解码不写盘（仍用 VideoReader 算 indices 与路径，与抽帧脚本一致）
            all_cached = True

        if not all_cached:
            images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
            for im, pth in zip(images, frame_paths):
                if osp.exists(pth):
                    continue
                try:
                    im.save(pth)
                except FileExistsError:
                    print(f"Error: {pth} 已经存在")
                    continue
                except Exception as e:
                    print(f"Error: {e}")

        return frame_paths, indices, video_info

    def _video_llm_cached_prompt_from_meta(self, video_key, query):
        meta = getattr(self, '_video_meta', {}).get(str(video_key), {})
        if not meta or self.fps <= 0:
            return None
        try:
            total = sample_frame_count_from_meta(meta, self.fps, self.nframe, self.frames_limit)
            duration = float(meta.get('duration', 0.0))
        except Exception:
            return None
        if total <= 0 or duration <= 0:
            return None

        required_frames = int(duration * self.fps)
        if required_frames > self.frames_limit:
            frame_paths = self._frame_paths(video_key, self.frames_limit)
        else:
            frame_paths = self._frame_paths_fps(video_key, total, self.fps)

        return [
            dict(
                type='video',
                value=frame_paths,
                sample_fps=sample_fps_from_meta(meta, len(frame_paths), self.fps, self.frames_limit),
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            ),
            dict(type="text", value=self._format_grounder_prompt(query)),
        ]

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = line["video"]
        # 与 scripts/pre_extract_video_frames/extract_video_frames.py 中按 `video` 去重抽帧一致：
        # 缓存目录必须只依赖视频 id，不能依赖样本 index，否则同一视频的多条 QA 会指向不同子目录。
        video_key = str(video_path)

        query = line["question"]
        if video_llm and not self.check_extracted_frames:
            cached = self._video_llm_cached_prompt_from_meta(video_key, query)
            if cached is not None:
                return cached

        frames, indices, video_info = self.save_video_frames(video_path, key=video_key)
        user_prompt1 = self._format_grounder_prompt(query)
        user_prompt2 = (
            wrap_vtg_prompt_with_reason(self.GROUNDER_PROMPT_TEXT_TIMESTAMP.format(query=query))
            if self.reason
            else self.GROUNDER_PROMPT_TEXT_TIMESTAMP.format(query=query)
        )

        if video_llm:
            message = []
            actual_fps = (
                self.frames_limit / video_info["duration"]
                if len(frames) == self.frames_limit and video_info["duration"] > 0
                else self.fps
            )
            message.append(dict(
                type='video', value=frames, sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels, # The maximum number of tokens per frame
                total_pixels=self.total_pixels, # total number of video tokens did not exceed 224K
            ))
            message.append(dict(type="text", value=user_prompt1))
            return message

        # Multi-image mode: instruction + timestamp before each frame (seconds)
        vfps = float(video_info.get("fps") or 0.0)
        message = [dict(type="text", value=user_prompt2)]
        for idx, im in zip(indices, frames):
            t = (float(idx) / vfps) if vfps > 0 else 0.0
            message.append(dict(type="text", value=f'{t:.2f}s'))
            message.append(dict(type="image", value=im))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ["xlsx", "json", "tsv"], "仅支持 xlsx/json/tsv 评测文件。"

        tgt_file = get_intermediate_file_path(eval_file, "_rating", "json")
        score_file = get_intermediate_file_path(eval_file, "_score")

        data = load(eval_file)

        if "task_mode" not in data:
            data["task_mode"] = "miou"

        def compute_iou_score(pred_raw, gt_span, duration):
            """计算预测时间段与真实时间段的 IoU"""
            if pred_raw is None or (isinstance(pred_raw, float) and np.isnan(pred_raw)):
                return -1
            
            if self.reason:
                pred_raw = extract_answer_from_tags(pred_raw)
            # 使用 TimeLens 的时间提取函数
            pred_timestamps = extract_time(str(pred_raw))
            
            if len(pred_timestamps) == 0:
                # 有预测但无法解析时间戳，计为 0 分并计入分母
                return 0.0
            
            # 取第一个预测的时间段
            pred_span = pred_timestamps[0]
            
            # 与原数据集评测代码保持一致：
            # 不检查时间是否超过视频时长，不检查开始时间是否小于0
            # 如果开始时间 >= 结束时间，仍然计算 IoU（会得到 0）
            
            # gt_span 可能是字符串或列表
            if isinstance(gt_span, str):
                try:
                    gt_span = json.loads(gt_span.replace("'", '"'))
                except:
                    gt_span = eval(gt_span)
            
            if isinstance(gt_span, list) and len(gt_span) > 0:
                if isinstance(gt_span[0], list):
                    gt_span = gt_span[0]
                gt_span = (float(gt_span[0]), float(gt_span[1]))
            else:
                return -1
            
            # 计算 IoU
            return iou(pred_span, gt_span)

        data_un = data[~pd.isna(data["prediction"])].copy()
        data_pred_na = data[pd.isna(data["prediction"])].copy()

        data_pred_na["score"] = -1

        data_un["score"] = data_un.apply(
            lambda row: compute_iou_score(
                row["prediction"],
                row["answer"],
                row.get("duration", 0),
            ),
            axis=1,
        )

        data = pd.concat([data_pred_na, data_un])
        rejected = (data["score"] == -1).sum()
        scored = data[data["score"] != -1]

        print(
            f"\n{'='*60}"
            f"\n[{self.dataset_name}] Evaluation Results"
            f"\n{'='*60}"
        )
        print(
            f"Total questions: {len(data)}"
            f"\nFailed to obtain prediction: {len(data_pred_na)}"
            f"\nFailed to parse ground truth: {rejected - len(data_pred_na)}"
            f"\nScored (incl. parse-fail as 0): {len(scored)}"
        )

        dump(data, score_file)

        # 输出无预测或 GT 解析失败的样本
        fail_rows = data[data["score"] == -1]
        if not fail_rows.empty:
            print("\nExamples of unscored samples (score == -1):")
            for idx, row in fail_rows.head(5).iterrows():
                print(f"Question idx: {idx}")
                print(f"  Model answer: {row['prediction']}")
                print(f"  Reference answer: {row['answer']}")
                if 'duration' in row:
                    print(f"  Duration: {row['duration']}")
                print("-" * 40)

        # 解析失败的预测已计为 0 分；仅无预测或 GT 解析失败时排除
        overall_miou = round(scored["score"].mean(), 4) if len(scored) else 0

        # 计算不同 IoU 阈值下的 Recall@1
        r1_03 = (scored["score"] >= 0.3).sum() if len(scored) > 0 else 0
        r1_05 = (scored["score"] >= 0.5).sum() if len(scored) > 0 else 0
        r1_07 = (scored["score"] >= 0.7).sum() if len(scored) > 0 else 0

        # 构建结果字符串
        result_str = f"\nResults:\n  mIoU: {overall_miou}"

        if len(scored) > 0:
            result_str += f"\n  R1@0.3: {r1_03}/{len(scored)} ({100*r1_03/len(scored):.2f}%)"
            result_str += f"\n  R1@0.5: {r1_05}/{len(scored)} ({100*r1_05/len(scored):.2f}%)"
            result_str += f"\n  R1@0.7: {r1_07}/{len(scored)} ({100*r1_07/len(scored):.2f}%)"
        else:
            result_str += "\n  R1@0.3: 0/0 (0.00%)"
            result_str += "\n  R1@0.5: 0/0 (0.00%)"
            result_str += "\n  R1@0.7: 0/0 (0.00%)"
        
        print(result_str)
        print(f"{'='*60}\n")
        
        # 为兼容 ConcatVideoDataset，返回包含数据集名称的字典
        # ConcatVideoDataset 期望的格式：{dataset_name: {'success': x, 'overall': y}}
        # 注意：将所有 numpy 类型转换为 Python 原生类型，避免 JSON 序列化错误
        rating = {
            self.dataset_name: {
                'miou': float(overall_miou),
                'r1_03': int(r1_03),
                'r1_05': int(r1_05), 
                'r1_07': int(r1_07),
                'scored_count': int(len(scored)),
                'total_count': int(len(data)),
            }
        }

        dump(rating, tgt_file)
        return rating


# 三个子数据集类
class TimeLens_ActivityNet(TimeLensBase):
    """ActivityNet-TimeLens 子数据集"""
    SUBSET_NAME = "activitynet"
    
    @classmethod
    def supported_datasets(cls):
        return ["ActivityNet-TimeLens"]


class TimeLens_QVHighlights(TimeLensBase):
    """QVHighlights-TimeLens 子数据集"""
    SUBSET_NAME = "qvhighlights"
    
    @classmethod
    def supported_datasets(cls):
        return ["QVHighlights-TimeLens"]


class TimeLens_Charades(TimeLensBase):
    """Charades-TimeLens 子数据集"""
    SUBSET_NAME = "charades"
    
    @classmethod
    def supported_datasets(cls):
        return ["Charades-TimeLens"]


# 合并数据集类
class TimeLens(ConcatVideoDataset):
    """
    TimeLens 合并数据集，包含三个子数据集：
    - ActivityNet-TimeLens
    - QVHighlights-TimeLens
    - Charades-TimeLens
    
    使用 ConcatVideoDataset 方式组合，评估时自动分别显示每个子数据集的结果。
    """
    
    DATASET_SETS = {
        "TimeLens": ["ActivityNet-TimeLens", "QVHighlights-TimeLens", "Charades-TimeLens"]
    }
    
    @classmethod
    def supported_datasets(cls):
        return ["TimeLens"]
