"""
MotionBench 数据集：视频多选题 (Video-MCQ)。
数据来源：motionbench_val_infos.json + video_info.meta.jsonl，
视频位于 self-collected/ 与 public-dataset/ 子目录。
"""
import json
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils.multiple_choice import mcq_vanilla_eval, extract_answer_from_item
from .utils.video_pyav import ffprobe_video_info, get_video_decode_backend, save_frames_by_indices_pyav


def _polish_answer(answer: str) -> str:
    """
    将模型输出规范为单个选项字母 (A/B/C/D/E)。
    参考 MotionBench 官方 answer_util。
    """
    if answer is None:
        return ""
    s = str(answer).strip()
    if not s:
        return ""
    s = s.split(")")[0].strip()
    if "(" in s:
        try:
            s = s.split("(")[1].strip()
        except Exception:
            pass
    s = s.split(" ")[0]
    s = s.strip()
    return s[0].upper() if len(s) > 0 else ""


def _parse_mcq_from_question(q: str) -> Tuple[str, Dict[str, str]]:
    """
    从题目文本解析题干与选项。
    题目格式通常为：题干行\\nA. ...\\nB. ...\\nC. ...
    返回 (题干, {选项字母: 选项内容})。
    """
    q = str(q).replace("\r\n", "\n").strip()
    lines = [ln.rstrip() for ln in q.split("\n") if ln.strip() != ""]
    stem_lines: List[str] = []
    opts: Dict[str, str] = {}
    opt_pat = re.compile(r"^([A-E])\s*[\.\)]\s*(.*)$")
    for ln in lines:
        m = opt_pat.match(ln.strip())
        if m:
            key = m.group(1).upper()
            val = m.group(2).strip()
            opts[key] = val
        else:
            if len(opts) == 0:
                stem_lines.append(ln)
            else:
                if opts:
                    last = sorted(opts.keys())[-1]
                    opts[last] = (opts[last] + " " + ln.strip()).strip()
    stem = "\n".join(stem_lines).strip()
    return stem, opts


def _load_meta_map(meta_jsonl: str) -> Dict[str, Dict[str, str]]:
    """
    从 video_info.meta.jsonl 构建 basename -> {question_type, video_type} 映射。
    """
    if not os.path.exists(meta_jsonl):
        return {}
    m = {}
    with open(meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vp = obj.get("video_path", "")
            if not vp:
                continue
            base = os.path.basename(vp)
            m[base] = dict(
                question_type=str(obj.get("question_type", "") or ""),
                video_type=str(obj.get("video_type", "") or ""),
            )
    return m


class MotionBench(VideoBaseDataset):
    """
    MotionBench: 细粒度视频运动理解多选题 (Video-MCQ)。
    数据目录需包含：
      - self-collected/、public-dataset/ 下的视频
      - motionbench_val_infos.json（或 motionbench_val_new.json）
      - 可选 video_info.meta.jsonl（或 data/video_info.meta.jsonl）用于 question_type / video_type
    """

    TYPE = "Video-MCQ"
    MODALITY = "VIDEO"

    SYS = "You are an AI assistant responsible for answering multiple-choice questions about a video."
    FRAMES_TMPL_SYS = (
        "You will receive {} distinct frames uniformly sampled from a video in chronological order.\n"
        "Answer the multiple-choice question based on these frames.\n"
    )
    POST_PROMPT = "Answer with ONLY the single uppercase letter of the correct option (A/B/C/D/E)."

    def __init__(
        self,
        dataset="MotionBench",
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
    ):
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )

    @classmethod
    def supported_datasets(cls):
        return ["MotionBench"]

    def prepare_dataset(
        self,
        dataset="MotionBench"
    ):
        """
        准备 MotionBench 数据：从 JSON 构建 TSV。
        Args:
            dataset: 数据集名称
            json_path: 标注 JSON 路径（默认 root/motionbench_val_infos.json，若无则 root/motionbench_val_new.json）
            root_dir: 视频根目录（默认 default_bench_root）
            meta_jsonl: 元数据 jsonl 路径（可选，默认 root/video_info.meta.jsonl 或 root/data/video_info.meta.jsonl）
        """
        root = "/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/MotionBench"

        infos_path = os.path.join(root, "motionbench_val_infos.json")
        if not os.path.exists(infos_path):
            infos_path = os.path.join(root, "motionbench_val_new.json")
        json_path = infos_path

        meta_jsonl = os.path.join(root, "video_info.meta.jsonl")
        if not os.path.exists(meta_jsonl):
            meta_jsonl = os.path.join(root, "data", "video_info.meta.jsonl")

        assert os.path.exists(json_path), f"{json_path} 不存在，请检查数据路径。"
        assert os.path.isdir(root), f"{root} 不存在，请检查数据路径。"

        tsv_path = json_path.replace(".json", "_new.tsv")
        if tsv_path == json_path:
            tsv_path = os.path.join(root, "MotionBench.tsv")

        if os.path.exists(tsv_path):
            return dict(root=root, data_file=tsv_path)

        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        if not isinstance(ann, list):
            raise ValueError(f"标注文件应为 list，当前为 {type(ann)}")

        meta_map = _load_meta_map(meta_jsonl)

        rows = []
        for idx, item in enumerate(ann):
            rel_vp = str(item.get("video_path", "")).lstrip("./")
            if not rel_vp:
                raise ValueError(f"第 {idx} 条缺少 video_path")
            abs_vp = os.path.join(root, rel_vp)
            if not os.path.exists(abs_vp):
                raise FileNotFoundError(f"视频不存在: {abs_vp}")

            q_raw = item.get("question", "")
            stem, opts = _parse_mcq_from_question(q_raw)
            answer = str(item.get("answer", "")).strip().upper()
            if answer and answer not in list("ABCDE"):
                answer = _polish_answer(answer)

            base = os.path.basename(rel_vp)
            meta = meta_map.get(base, {})

            video_id = os.path.splitext(rel_vp)[0].replace("/", "__")
            row = dict(
                index=idx,
                video=video_id,
                video_path=rel_vp,
                question=stem if stem else str(q_raw).strip(),
                A=opts.get("A", ""),
                B=opts.get("B", ""),
                C=opts.get("C", ""),
                D=opts.get("D", ""),
                E=opts.get("E", ""),
                answer=answer,
                question_type=meta.get("question_type", ""),
                video_type=meta.get("video_type", ""),
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)
        dump(df, tsv_path)
        return dict(root=root, data_file=tsv_path)

    def save_video_frames(self, line, video_llm: bool = False, verbose: bool = False):
        """
        根据 video_path 从 data_root 下读取视频并采样帧。
        返回 (frame_paths, indices, video_info)。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_id = str(line["video"])
        rel_vp = str(line["video_path"])
        vid_path = os.path.normpath(os.path.join(self.data_root, rel_vp))
        backend = get_video_decode_backend()
        use_pyav = False

        # self-collected 里面有三个视频有问题
        # 所以需要替换为新的路径
        vid_path = vid_path.replace("self-collected",'self-collected_new_originfps')


        print(f"video_id: {video_id}")
        print(f"vid_path: {vid_path}")


        vid = None
        try:
            if backend != "pyav":
                import decord
                vid = decord.VideoReader(vid_path, num_threads=1)
                fps = float(vid.get_avg_fps())
                n_frames = int(len(vid))
                duration = (n_frames / fps) if fps > 0 else 0.0
            else:
                raise RuntimeError("force pyav")
        except Exception:
            n_frames, fps, duration = ffprobe_video_info(vid_path)
            use_pyav = True

        video_info = {
            "fps": float(fps),
            "n_frames": int(n_frames),
            "duration": float(duration),
            "backend": "pyav" if use_pyav else "decord",
        }

        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_id)
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                if verbose:
                    print(
                        f"Warning: 视频 `{rel_vp}` 按 {self.fps} fps 需 {required_frames} 帧，"
                        f"已截断为 {self.frames_limit} 帧。"
                    )
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_root = osp.join(self.frame_root, video_id)
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [
                    osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit))
                    for i in range(1, self.frames_limit + 1)
                ]
            else:
                step_size = fps / self.fps if self.fps > 0 else 1.0
                indices = [int(i * step_size) for i in range(required_frames)]
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
            if vid is not None:
                images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        try:
                            im.save(pth)
                        except FileExistsError:
                            print(f"Error: {pth} 已经存在")
                            continue
                        except Exception as e:
                            print(f"Error: {e}")
            else:
                save_frames_by_indices_pyav(
                    vid_path=vid_path,
                    indices=[int(x) for x in indices],
                    frame_paths=frame_paths,
                    total_frames=video_info.get("n_frames"),
                    desc=f"Extracting (pyav): {rel_vp}",
                )
        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        opts = []
        for ch in "ABCDE":
            val = str(line.get(ch, "")).strip()
            if val and val.lower() != "nan":
                opts.append(f"{ch}. {val}")
        opt_str = "\n".join(opts)
        q = str(line["question"]).strip()

        frames, _, video_info = self.save_video_frames(line, video_llm=video_llm)
        msg = []

        if video_llm:
            assert self.fps > 0
            actual_fps = (
                self.frames_limit / video_info["duration"]
                if len(frames) == self.frames_limit and video_info.get("duration", 0) > 0
                else self.fps
            )
            prompt = f"{self.SYS}\n\nQuestion:\n{q}\n\nOptions:\n{opt_str}\n\n{self.POST_PROMPT}"
            msg = [dict(
                type="video", value=frames, sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            )]
        else:
            for im in frames:
                msg.append(dict(type="image", value=im))


        msg.append(dict(type="text", value=self.FRAMES_TMPL_SYS.format(len(frames))))
        msg.append(dict(type="text", value=f"Question:\n{q}\n\nOptions:\n{opt_str}\n\nAnswer: "))
        msg.append(dict(type="text", value=self.POST_PROMPT))

        return msg

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ["xlsx", "json", "tsv"], (
            "评测文件须为 xlsx/json/tsv 之一"
        )

        judge_name = judge_kwargs.setdefault("model", "exact_matching")
        nproc = judge_kwargs.pop("nproc", 1)

        tmp_file = get_intermediate_file_path(eval_file, f"_{judge_name}_tmp", "pkl")
        score_file = get_intermediate_file_path(eval_file, f"_{judge_name}_score")
        acc_file = get_intermediate_file_path(eval_file, f"_{judge_name}_acc", "json")

        if not os.path.exists(score_file):
            data = load(eval_file)
            if "prediction" not in data.columns:
                raise ValueError("评测文件须包含 `prediction` 列。")
            if "answer" not in data.columns:
                raise ValueError("评测文件须包含 `answer`（GT 选项字母）列。")

            meta = data[["index", "answer"]].copy()
            meta["index"] = meta["index"].astype(int)

            use_judge = judge_name not in [None, "exact_matching"]
            judge_model = None
            if use_judge and gpt_key_set():
                try:
                    from .utils import build_judge
                    judge_model = build_judge(**judge_kwargs)
                    if hasattr(judge_model, "working") and (not judge_model.working()):
                        warnings.warn("Judge 不可用，将使用精确匹配评测")
                        judge_model = None
                except Exception as e:
                    warnings.warn(f"构建 Judge 失败 ({judge_name})，回退精确匹配: {type(e).__name__}: {e}")
                    judge_model = None
            elif use_judge and (not gpt_key_set()):
                warnings.warn("未设置 OPENAI_API_KEY，将使用精确匹配评测")
                judge_model = None

            if judge_model is None:
                data = data.copy()
                data["prediction"] = [_polish_answer(x) for x in data["prediction"]]
                scored = mcq_vanilla_eval(
                    model=None,
                    data=data,
                    meta=meta,
                    nproc=nproc,
                    result_file=tmp_file,
                    dataset_name="MotionBench",
                )
            else:
                cache = load(tmp_file) if osp.exists(tmp_file) else {}
                if not isinstance(cache, dict):
                    cache = {}
                answer_map = {int(i): str(a).strip().upper() for i, a in zip(meta["index"], meta["answer"])}

                updated = False
                for _, row in data.iterrows():
                    idx = int(row["index"])
                    if idx in cache:
                        continue
                    gt = answer_map.get(idx, "")
                    item = {
                        "question": str(row.get("question", "")),
                        "prediction": "" if pd.isna(row.get("prediction", "")) else str(row["prediction"]),
                        "answer": gt,
                    }
                    for ch in "ABCDE":
                        item[ch] = "" if pd.isna(row.get(ch, "")) else str(row[ch])
                    try:
                        res = extract_answer_from_item(judge_model, item, dataset_name="MotionBench")
                        opt = str(res.get("opt", "") or "").strip().upper()
                        match_log = str(res.get("log", "") or "")
                    except Exception as e:
                        opt = ""
                        match_log = f"Exception: {type(e).__name__}: {e}"
                    hit = int(opt == gt and gt in list("ABCDE"))
                    cache[idx] = dict(hit=hit, log=f"Match Log: {match_log}. ")
                    updated = True
                if updated:
                    dump(cache, tmp_file)

                data = data[data["index"].astype(int).isin(answer_map)].copy()
                data["hit"] = [cache[int(i)]["hit"] for i in data["index"].astype(int)]
                data["log"] = [cache[int(i)]["log"] for i in data["index"].astype(int)]
                scored = data
            dump(scored, score_file)

        scored = load(score_file)
        overall = float(np.mean(scored["hit"])) if len(scored) else 0.0

        def _group_acc(col: str) -> Dict[str, float]:
            if col not in scored.columns:
                return {}
            out = {}
            vals = [x for x in scored[col] if not pd.isna(x) and str(x).strip() != ""]
            for v in sorted(set(vals)):
                sub = scored[scored[col] == v]
                out[str(v)] = float(np.mean(sub["hit"])) if len(sub) else 0.0
            return out

        res = {
            self.dataset_name: dict(
                overall_accuracy=float(overall),
                question_type_accuracy=_group_acc("question_type"),
                video_type_accuracy=_group_acc("video_type"),
                num_samples=int(len(scored)),
            )
        }
        dump(res, acc_file)

        print(
            f"\n{'='*60}\n[{self.dataset_name}] Evaluation Results\n{'='*60}"
        )
        print(f"Total: {len(scored)}, Overall Accuracy: {overall:.4f}")
        print(f"{'='*60}\n")

        return res
