"""
Minerva 数据集：视频多选题 (Video-MCQ)。
视频按标题存储，标注使用 video_id，通过 url_title_map.tsv 解析本地视频路径。
"""
import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils.multiple_choice import mcq_vanilla_eval, extract_answer_from_item


@dataclass(frozen=True)
class _MinervaPaths:
    root: str
    videos_dir: str
    ann_json: str
    url_title_map_tsv: str
    tsv_path: str


def _default_minerva_root() -> str:
    """默认 Minerva 根目录，可通过环境变量 MINERVA_ROOT 覆盖。"""
    return os.environ.get(
        "MINERVA_ROOT",
        "/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/Minerva",
    ).strip()


def _canonicalize_title(s: str) -> str:
    """规范化 unicode 与标点，便于与本地文件名匹配。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    trans = str.maketrans({
        "｜": "|", "⧸": "/", "／": "/", "：": ":", "？": "?", "＊": "*",
        "＂": '"', """: '"', """: '"', "'": "'", "'": "'",
    })
    s = s.translate(trans)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_title_for_filename(title: str) -> str:
    """近似 yt-dlp 的文件名 sanitize（替换 Windows 禁用字符）。"""
    if title is None:
        return ""
    title = str(title)
    trans = str.maketrans({
        "|": "｜", "/": "⧸", "\\": "⧸", ":": "：", "?": "？",
        "*": "＊", '"': "＂", "<": "＜", ">": "＞",
    })
    return title.translate(trans)


def _build_video_stem_index(videos_dir: str) -> Dict[str, str]:
    """构建 canonical_title -> stem 的映射。"""
    idx: Dict[str, str] = {}
    with os.scandir(videos_dir) as it:
        for ent in it:
            if not ent.is_file():
                continue
            name = ent.name
            if not name.lower().endswith(".mp4"):
                continue
            stem = name[:-4]
            key = _canonicalize_title(stem)
            if key and key not in idx:
                idx[key] = stem
    return idx


def _resolve_video_stem(
    video_id: str,
    title_raw: str,
    videos_dir: str,
    stem_index: Dict[str, str],
) -> str:
    """根据 (video_id, title_raw) 解析本地 mp4 的 stem（无扩展名）。"""
    candidates: List[str] = []
    if title_raw:
        candidates.append(title_raw)
        candidates.append(_sanitize_title_for_filename(title_raw))
        candidates.append(_sanitize_title_for_filename(title_raw).strip().rstrip("."))
    for cand in candidates:
        if not cand:
            continue
        p = os.path.join(videos_dir, f"{cand}.mp4")
        if os.path.exists(p):
            return cand
    can_key = _canonicalize_title(_sanitize_title_for_filename(title_raw))
    if can_key in stem_index:
        return stem_index[can_key]
    can_key2 = _canonicalize_title(title_raw)
    if can_key2 in stem_index:
        return stem_index[can_key2]
    raise FileNotFoundError(
        f"[Minerva] 未找到本地视频 video_id={video_id}, title={title_raw!r} 于 {videos_dir}"
    )


def _read_url_title_map(tsv_path: str) -> pd.DataFrame:
    """读取 url_title_map.tsv（支持 \\t 或真实 tab）。"""
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split("\\t")
            if len(parts) < 3:
                parts = (parts + [""] * 3)[:3]
            video_id, title, url = parts[0], parts[1], parts[2]
            rows.append(dict(
                video_id=str(video_id).strip(),
                title=str(title).strip(),
                url=str(url).strip(),
            ))
    return pd.DataFrame(rows)


def _ensure_minerva_tsv(paths: _MinervaPaths) -> None:
    """从 minerva.json + url_title_map.tsv 生成 Minerva.tsv。"""
    if os.path.exists(paths.tsv_path):
        return
    if not os.path.exists(paths.ann_json):
        raise FileNotFoundError(f"[Minerva] 标注文件不存在: {paths.ann_json}")
    if not os.path.exists(paths.url_title_map_tsv):
        raise FileNotFoundError(f"[Minerva] url_title_map.tsv 不存在: {paths.url_title_map_tsv}")
    if not os.path.isdir(paths.videos_dir):
        raise FileNotFoundError(f"[Minerva] 视频目录不存在: {paths.videos_dir}")

    url_map = _read_url_title_map(paths.url_title_map_tsv)
    id2title = {r["video_id"]: r["title"] for _, r in url_map.iterrows()}

    with open(paths.ann_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"[Minerva] minerva.json 应为 list，当前为 {type(records)}")

    stem_index = _build_video_stem_index(paths.videos_dir)
    rows = []
    for i, r in enumerate(records):
        video_id = str(r.get("video_id", ""))
        title_raw = id2title.get(video_id, "")
        video_stem = _resolve_video_stem(video_id, title_raw, paths.videos_dir, stem_index)
        answer_id = int(r.get("answer_id"))
        gt_letter = chr(ord("A") + answer_id)
        choices = [r.get(f"answer_choice_{k}") for k in range(5)]
        row = dict(
            index=i,
            key=r.get("key", ""),
            video_id=video_id,
            video=video_stem,
            question=r.get("question", ""),
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            E=choices[4],
            answer=gt_letter,
            answer_id=answer_id,
            answer_text=r.get("answer", choices[answer_id] if 0 <= answer_id < 5 else ""),
            question_type=r.get("question_type", ""),
            split=r.get("split", ""),
            category=r.get("category", ""),
            reasoning=r.get("reasoning", ""),
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(paths.tsv_path) or ".", exist_ok=True)
    dump(df, paths.tsv_path)


class Minerva(VideoBaseDataset):
    """
    Minerva: 视频推理多选题 (Video-MCQ)。
    视频按标题存储，标注用 video_id，通过 url_title_map.tsv 解析本地视频。
    使用 decord 读取与保存帧。
    """

    TYPE = "Video-MCQ"
    MODALITY = "VIDEO"

    SYS = (
        "You are an AI assistant responsible for answering questions about a video.\n"
        "Select the best option (A/B/C/D/E) based on the video.\n"
    )
    FRAMES_TMPL_SYS = (
        "You will receive {} distinct frames uniformly sampled from a video in chronological order.\n"
        "Answer the multiple-choice question based on these frames.\n"
    )
    POST_PROMPT = "Answer with ONLY the single uppercase letter of the correct option (A/B/C/D/E)."

    def __init__(
        self,
        dataset="Minerva",
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
        return ["Minerva"]

    def prepare_dataset(self, dataset="Minerva", root_dir=None):
        root = root_dir if root_dir is not None else _default_minerva_root()
        paths = _MinervaPaths(
            root=root,
            videos_dir=os.path.join(root, "videos"),
            ann_json=os.path.join(root, "minerva.json"),
            url_title_map_tsv=os.path.join(root, "url_title_map.tsv"),
            tsv_path=os.path.join(root, "Minerva.tsv"),
        )
        _ensure_minerva_tsv(paths)
        return dict(root=paths.videos_dir, data_file=paths.tsv_path)

    def _frame_paths(self, key, num_frames):
        frame_root = osp.join(self.frame_root, key)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def _frame_paths_fps(self, key, num_frames, fps_val):
        frame_root = osp.join(self.frame_root, key)
        os.makedirs(frame_root, exist_ok=True)
        return [
            osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, fps_val))
            for i in range(1, num_frames + 1)
        ]

    def _save_frames_with_decord(self, vid_path, key, video_info, indices, frame_paths):
        """使用 decord 解码并保存帧。"""
        import decord
        vid = decord.VideoReader(vid_path)
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

    def _save_frames_with_av(self, vid_path, indices, frame_paths):
        """使用 av (PyAV) 解码并保存指定索引的帧。"""
        import av
        indices_set = set(indices)
        index_to_paths = {}
        for i, idx in enumerate(indices):
            index_to_paths.setdefault(idx, []).append(frame_paths[i])
        # PyAV 14+ 的 VideoFrame 不再提供 .index，顺序解码时用枚举序号对齐 decord 的帧下标。
        with av.open(vid_path) as container:
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx in indices_set:
                    for pth in index_to_paths[frame_idx]:
                        if not osp.exists(pth):
                            try:
                                arr = frame.to_ndarray(format="rgb24")
                                Image.fromarray(arr).save(pth)
                            except Exception as e:
                                print(f"Error saving frame {frame_idx}: {e}")

    def save_video_frames(self, video_path, key, verbose=False):
        """
        使用 decord 读取视频并保存采样帧，风格与 timelens 一致。
        若 decord 出错则回退到 av (PyAV) 读取。
        video_path: 视频 stem（无 .mp4），对应 data_root 下的 {video_path}.mp4
        key: 帧缓存子目录名（通常与 video_path 一致，便于同视频多题共享帧）
        返回 (frame_paths, indices, video_info)。
        """
        vid_path = osp.join(self.data_root, f"{video_path}.mp4")
        use_av = False

        try:
            import decord
            vid = decord.VideoReader(vid_path)
            video_fps = vid.get_avg_fps()
            n_frames = len(vid)
            video_info = {
                "fps": video_fps,
                "n_frames": n_frames,
                "duration": n_frames / video_fps,
            }
        except Exception as e:
            warnings.warn(f"decord 读取视频失败 ({vid_path}): {e}，改用 av 读取。")
            use_av = True
            import av
            with av.open(vid_path) as container:
                stream = container.streams.video[0]
                video_fps = float(stream.average_rate) if stream.average_rate else 30.0
                n_frames = stream.frames if stream.frames is not None else int(
                    stream.duration * float(stream.time_base) * video_fps
                )
                if n_frames <= 0:
                    n_frames = 1
            video_info = {
                "fps": video_fps,
                "n_frames": n_frames,
                "duration": n_frames / video_fps,
            }

        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self._frame_paths(key, len(indices))
            video_info["sample_fps"] = video_fps / step_size if step_size > 0 else video_fps
            video_info["sample_n_frame"] = len(indices)
        else:
            required_frames = max(1, int(video_info["duration"] * self.fps))
            if required_frames > self.frames_limit:
                print(f"Warning: required_frames > frames_limit, required_frames: {required_frames}, frames_limit: {self.frames_limit}")
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_root = osp.join(self.frame_root, key)
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [
                    osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit))
                    for i in range(1, self.frames_limit + 1)
                ]
                video_info["sample_fps"] = self.frames_limit / video_info["duration"]
                video_info["sample_n_frame"] = self.frames_limit
            else:
                step_size = video_fps / self.fps
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self._frame_paths_fps(key, len(indices), self.fps)
                video_info["sample_fps"] = self.fps
                video_info["sample_n_frame"] = required_frames

        if self.check_extracted_frames and not np.all([osp.exists(p) for p in frame_paths]):
            if use_av:
                self._save_frames_with_av(vid_path, indices, frame_paths)
            else:
                self._save_frames_with_decord(vid_path, key, video_info, indices, frame_paths)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = line["video"]
        # 同一视频多题共享帧缓存，key 用 video stem
        message = []
        frames, indices, video_info = self.save_video_frames(video_path, key=video_path)

        q = str(line["question"])
        opts = [
            ("A", str(line.get("A", ""))),
            ("B", str(line.get("B", ""))),
            ("C", str(line.get("C", ""))),
            ("D", str(line.get("D", ""))),
            ("E", str(line.get("E", ""))),
        ]
        opt_str = "\n".join([f"{k}. {v}" for k, v in opts])

        msg= []

        if video_llm:
            actual_fps = (
                self.frames_limit / video_info["duration"]
                if len(frames) == self.frames_limit and video_info.get("duration", 0) > 0
                else self.fps
            )
            prompt = f"{self.SYS}\n\nQuestion:\n{q}\n\nOptions:\n{opt_str}\n\n{self.POST_PROMPT}"
            msg = [
                dict(
                    type="video", value=frames, sample_fps=actual_fps,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    total_pixels=self.total_pixels,
                ),
            ]
        else:
            for im in frames:
                msg.append(dict(type="image", value=im))

        msg.append(dict(type="text", value=self.FRAMES_TMPL_SYS.format(len(frames))))
        msg.append(dict(type="text", value=f"Question:\n{q}\n\nOptions:\n{opt_str}\n\nAnswer: "))
        msg.append(dict(type="text", value=self.POST_PROMPT))
        return msg

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ["xlsx", "json", "tsv"], "评测文件须为 xlsx/json/tsv 之一"

        judge_name = judge_kwargs.setdefault("model", "exact_matching")
        nproc = judge_kwargs.pop("nproc", 1)

        tmp_file = get_intermediate_file_path(eval_file, f"_{judge_name}_tmp", "pkl")
        score_file = get_intermediate_file_path(eval_file, f"_{judge_name}_score")
        acc_file = get_intermediate_file_path(eval_file, f"_{judge_name}_acc", "json")

        if not os.path.exists(score_file):
            data = load(eval_file)
            if not all(c in data.columns for c in list("ABCDE")):
                for i, ch in enumerate("ABCDE"):
                    src = f"answer_choice_{i}"
                    if src in data.columns:
                        data[ch] = data[src]
            if "answer" not in data.columns:
                raise ValueError("[Minerva] 评测文件须包含 `answer` 列。")

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
                scored = mcq_vanilla_eval(
                    model=None,
                    data=data,
                    meta=meta,
                    nproc=nproc,
                    result_file=tmp_file,
                    dataset_name="Minerva",
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
                        res = extract_answer_from_item(judge_model, item, dataset_name="Minerva")
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
            for v in sorted(set([x for x in scored[col] if not pd.isna(x)])):
                sub = scored[scored[col] == v]
                out[str(v)] = float(np.mean(sub["hit"])) if len(sub) else 0.0
            return out

        res = dict(
            overall_accuracy=overall,
            split_accuracy=_group_acc("split"),
            question_type_accuracy=_group_acc("question_type"),
            category_accuracy=_group_acc("category"),
        )
        dump(res, acc_file)

        print(f"\n{'='*60}\n[{self.dataset_name}] Evaluation Results\n{'='*60}")
        print(f"Total: {len(scored)}, Overall Accuracy: {overall:.4f}")
        print(f"{'='*60}\n")

        return res
