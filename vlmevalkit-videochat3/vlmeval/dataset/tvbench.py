"""
TVBench 数据集：视频多选题 (Video-MCQ)。
与 MVBench 结构一致，数据来源：TVBench/json/*.json + TVBench/video/<task_subdir>/。
参考：tvbench/infer.ipynb, VLMEvalKit mvbench.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.mvbench import check_ans, check_ans_with_model, get_dimension_rating

FAIL_MSG = "Failed to obtain answer via API."


def _default_tvbench_root() -> str:
    """默认 TVBench 根目录，可通过环境变量 TVBENCH_ROOT 覆盖。"""
    return os.environ.get(
        "TVBENCH_ROOT",
        "/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/TVBench",
    )


# TVBench 任务列表，与 infer.ipynb 中 data_list 一致
TVBENCH_DATA_LIST = {
    "Action Count": ("action_count.json", "video/action_count", "video", False),
    "Object Count": ("object_count.json", "video/object_count", "video", False),
    "Action Sequence": ("action_sequence.json", "video/action_sequence", "video", True),
    "Object Shuffle": ("object_shuffle.json", "video/object_shuffle", "video", False),
    "Scene Transition": ("scene_transition.json", "video/scene_transition", "video", False),
    "Action Localization": ("action_localization.json", "video/action_localization", "video", True),
    "Action Antonym": ("action_antonym.json", "video/action_antonym", "video", False),
    "Unexpected Action": ("unexpected_action.json", "video/unexpected_action", "video", False),
    "Egocentric Sequence": ("egocentric_sequence.json", "video/egocentric_sequence", "video", False),
    "Moving Direction": ("moving_direction.json", "video/moving_direction", "video", False),
}


class TVBench(VideoBaseDataset):
    """
    TVBench: 视频多选题，与 MVBench 用法一致。
    数据目录需包含：
      - json/*.json 各任务标注
      - video/<task_subdir>/ 下对应视频文件
    可通过环境变量 TVBENCH_ROOT 指定根目录，或 prepare_dataset(root_dir=...) 传入。
    """

    TYPE = "Video-MCQ"
    MODALITY = "VIDEO"

    SYS = """Carefully watch the video and pay attention to the cause and sequence of events, \
the detail and movement of objects, and the action and pose of persons. \
Based on your observations, select the best option that accurately addresses the question.
"""

    def __init__(
        self,
        dataset="TVBench",
        nframe=0,
        fps=-1,
        frames_limit=None,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
    ):
        self.type_data_list = TVBENCH_DATA_LIST
        self.num_segments = 16  # 与 MVBench 一致，用于 bound 时采样
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )
        self.frames_limit = frames_limit  # 当 fps>0 时，超过此帧数则均匀截断
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels

    @classmethod
    def supported_datasets(cls):
        return ["TVBench"]

    def prepare_dataset(
        self,
        dataset="TVBench",
        root_dir=None,
        json_dir=None,
    ):
        """
        准备 TVBench 数据：从各任务 JSON 合并为 TSV。
        Args:
            dataset: 数据集名称
            root_dir: TVBench 根目录（默认 TVBENCH_ROOT 或 _default_tvbench_root）
            json_dir: json 目录，默认 root_dir/json
        """
        root = root_dir if root_dir is not None else _default_tvbench_root()
        if json_dir is None:
            json_dir = os.path.join(root, "json")
        data_file = os.path.join(root, f"{dataset}.tsv")

        if os.path.exists(data_file):
            return dict(root=root, data_file=data_file)

        assert os.path.isdir(root), f"TVBench 根目录不存在: {root}"
        assert os.path.isdir(json_dir), f"JSON 目录不存在: {json_dir}"

        data_list = []
        for task_type, (json_name, prefix, data_type, bound) in self.type_data_list.items():
            json_path = os.path.join(json_dir, json_name)
            if not os.path.exists(json_path):
                warnings.warn(f"TVBench 跳过缺失的 JSON: {json_path}")
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            if not isinstance(json_data, list):
                json_data = [json_data]
            video_dir = os.path.join(root, prefix)
            for data in json_data:
                video_name = data.get("video", "")
                if not video_name:
                    continue
                video_path = os.path.join(root, prefix, video_name)
                if not os.path.exists(video_path):
                    warnings.warn(f"视频不存在，跳过: {video_path}")
                    continue
                candidates = data.get("candidates", [])
                if isinstance(candidates, str):
                    try:
                        candidates = json.loads(candidates)
                    except Exception:
                        candidates = eval(candidates) if candidates else []
                video_id = f"{prefix.replace('/', '__')}__{os.path.splitext(video_name)[0]}"
                row = {
                    "task_type": task_type,
                    "prefix": prefix,
                    "data_type": data_type,
                    "bound": bound,
                    "start": data.get("start") if bound else None,
                    "end": data.get("end") if bound else None,
                    "video": video_id,
                    "video_file": video_name,
                    "question": data.get("question", ""),
                    "answer": data.get("answer", ""),
                    "candidates": json.dumps(candidates) if isinstance(candidates, list) else candidates,
                }
                data_list.append(row)

        if not data_list:
            raise FileNotFoundError(
                f"TVBench 未找到有效样本，请检查 {root} 下 json/ 与 video/ 目录及内容。"
            )
        data_df = pd.DataFrame(data_list)
        data_df["index"] = np.arange(len(data_df))
        os.makedirs(os.path.dirname(data_file) or ".", exist_ok=True)
        data_df.to_csv(data_file, sep="\t", index=False)
        return dict(root=root, data_file=data_file)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        """与 MVBench 一致：在 [start, end] 秒内均匀采 num_segments 帧。"""
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video_with_bound(self, video_path, bound=None):
        """读取视频并可选地在 [start,end] 段内采样 num_segments 帧。"""
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        images = []
        for i in frame_indices:
            i = min(max(0, int(i)), max_frame)
            images.append(Image.fromarray(vr[i].asnumpy()))
        return images

    def qa_template(self, data):
        """与 MVBench/tvbench 一致：Question + Options + answer 格式为 (X) content。"""
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data["answer"]
        candidates = data["candidates"]
        if isinstance(candidates, str):
            try:
                candidates = json.loads(candidates)
            except Exception:
                candidates = eval(candidates)
        answer_idx = -1
        for idx, c in enumerate(candidates):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}" if answer_idx >= 0 else answer
        return question, answer

    def save_video_frames(self, line, video_llm=False, verbose=False):
        """
        根据 prefix + video 定位视频，若 bound 则在 [start,end] 内采样，否则按 nframe/fps 均匀采样。
        返回 (frame_paths, video_info_dict)。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
        prefix = str(line["prefix"])
        video_id = str(line["video"])
        video_name = str(line.get("video_file", line["video"]))
        video_path = os.path.join(self.data_root, prefix, video_name)
        bound = None
        if line.get("bound") and line.get("start") is not None and line.get("end") is not None:
            bound = (float(line["start"]), float(line["end"]))

        # video_id 已在 TSV 中设为唯一 id

        if bound is not None:
            # 段内固定采 num_segments 帧，与 MVBench 行为一致
            n_seg = self.num_segments
            frame_root = osp.join(self.frame_root, video_id)
            os.makedirs(frame_root, exist_ok=True)
            if not self.check_extracted_frames:
                if self.fps <= 0:
                    frame_paths = [
                        osp.join(frame_root, self.frame_tmpl.format(i + 1, n_seg))
                        for i in range(n_seg)
                    ]
                else:
                    frame_paths = [
                        osp.join(frame_root, self.frame_tmpl_fps.format(i + 1, n_seg, self.fps))
                        for i in range(n_seg)
                    ]
                video_info = {"fps": 0, "n_frames": n_seg, "duration": 0, "backend": "decord"}
                return frame_paths, video_info

            images = self.read_video_with_bound(video_path, bound=bound)
            self.num_segments = len(images)
            if self.fps <= 0:
                frame_paths = [osp.join(frame_root, self.frame_tmpl.format(i + 1, self.num_segments)) for i in range(self.num_segments)]
            else:
                frame_paths = [osp.join(frame_root, self.frame_tmpl_fps.format(i + 1, len(images), self.fps)) for i in range(len(images))]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

            video_info = {"fps": 0, "n_frames": len(images), "duration": 0, "backend": "decord"}
            return frame_paths, video_info

        # 无 bound：按 nframe 或 fps 均匀采样（与 MotionBench 类似，用 decord）
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        n_frames = len(vr)
        fps = float(vr.get_avg_fps())
        duration = (n_frames / fps) if fps > 0 else 0.0
        video_info = {"fps": fps, "n_frames": n_frames, "duration": duration, "backend": "decord"}

        if self.nframe > 0 and self.fps <= 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_id)
        elif self.fps > 0:
            required_frames = max(1, int(duration * self.fps)) if duration > 0 else (self.nframe or 8)
            if self.frames_limit is not None and required_frames > self.frames_limit:
                warnings.warn(
                    f"Video `{video_id}` requires {required_frames} frames at {self.fps} fps. "
                    f"Truncating to {self.frames_limit} frames."
                )
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                indices = [min(max(0, i), n_frames - 1) for i in indices]
                frame_paths = self.frame_paths_fps(video_id, self.frames_limit)
            else:
                step_size = (fps / self.fps) if self.fps > 0 else (n_frames / required_frames)
                indices = [int(i * step_size) for i in range(required_frames)]
                indices = [min(max(0, i), n_frames - 1) for i in indices]
                frame_paths = self.frame_paths_fps(video_id, len(indices))
        else:
            raise ValueError("nframe 与 fps 至少设置其一且有效。")

        if n_frames > 0 and indices:
            max_idx = n_frames - 1
            indices = [min(max(0, int(i)), max_idx) for i in indices]

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )
        if need_extract:
            for i, idx in enumerate(indices):
                pth = frame_paths[i] if i < len(frame_paths) else osp.join(
                    self.frame_root, video_id, self.frame_tmpl.format(i + 1, len(indices))
                )
                if not osp.exists(pth):
                    img = Image.fromarray(vr[idx].asnumpy())
                    img.save(pth)
        return frame_paths, video_info

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        question, _ = self.qa_template(line)
        message = [dict(type="text", value=self.SYS, role="system")]
        frame_paths, video_info = self.save_video_frames(line, video_llm=video_llm)
        if video_llm:
            duration = video_info.get("duration") or 0.0
            n_frames = len(frame_paths)
            actual_fps = (n_frames / duration) if duration > 0 and n_frames > 0 else (self.fps if self.fps > 0 else 1.0)
            message.append(dict(
                type="video",
                value=frame_paths,
                sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            ))
        else:
            for p in frame_paths:
                message.append(dict(type="image", value=p))
        message.append(dict(type="text", value=question))
        message.append(dict(type="text", value="\nOnly give the best option."))
        message.append(dict(type="text", value="Best option:(", role="assistant"))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ["xlsx", "json", "tsv"], (
            "评测文件须为 xlsx/json/tsv 之一"
        )
        tmp_file = get_intermediate_file_path(eval_file, "_tmp", "pkl")
        tgt_file = get_intermediate_file_path(eval_file, "_rating", "json")
        score_file = get_intermediate_file_path(eval_file, "_score")

        if not osp.exists(score_file):
            model_name = judge_kwargs.get("model", "exact_matching")
            model = None
            if model_name == "exact_matching":
                model = None
            else:
                try:
                    if gpt_key_set():
                        model = build_judge(**judge_kwargs)
                        if hasattr(model, "working") and not model.working():
                            warnings.warn(
                                "Judge model is not working properly, will use exact matching for evaluation"
                            )
                            warnings.warn(DEBUG_MESSAGE)
                            model = None
                    else:
                        warnings.warn(
                            "OPENAI_API_KEY is not set (or judge backend not configured), "
                            "will use exact matching for evaluation"
                        )
                        model = None
                except Exception as e:
                    warnings.warn(
                        f"Failed to build judge model ({model_name}): {e}. Will use exact matching."
                    )
                    model = None

            data = load(eval_file)
            data_un = data[~pd.isna(data["prediction"])]
            if "score" not in data.columns:
                data["score"] = np.nan

            for idx in data_un["index"]:
                ans = data.loc[data["index"] == idx, "answer"].values[0]
                pred = data.loc[data["index"] == idx, "prediction"].values[0]
                candidates = data.loc[data["index"] == idx, "candidates"].values[0]
                if isinstance(candidates, str):
                    try:
                        options = json.loads(candidates)
                    except Exception:
                        options = eval(candidates)
                else:
                    options = candidates
                answer_idx = -1
                for i, c in enumerate(options):
                    if c == ans:
                        answer_idx = i
                        break
                ans_formatted = (
                    f"({chr(ord('A') + answer_idx)}) {ans}"
                    if answer_idx >= 0
                    else str(ans)
                )
                input_item = data.loc[data["index"] == idx].to_dict(orient="records")[0]
                for i, opt in enumerate(options):
                    input_item[chr(ord("A") + i)] = opt
                input_item["answer"] = chr(ord("A") + answer_idx) if answer_idx >= 0 else str(ans)

                if FAIL_MSG in str(pred):
                    data.loc[data["index"] == idx, "score"] = -1
                else:
                    hit = 0
                    if model is None:
                        hit = 1 if check_ans(pred, ans_formatted) else 0
                    else:
                        hit = 1 if check_ans_with_model(pred, ans_formatted, model, input_item, "TVBench") else 0
                    data.loc[data["index"] == idx, "score"] = hit

            rejected = [x for x in data["score"] if x == -1]
            print(
                f"Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, "
                f"failed to obtain the score for another {len(rejected)} questions. "
                f"Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating."
            )
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
