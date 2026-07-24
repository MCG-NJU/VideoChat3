#!/usr/bin/env python3
"""Aggregate VLMEvalKit benchmark results from an output folder into a summary CSV."""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

BENCHMARKS = [
    ("Video-MME (short)", "video_mme_short"),
    ("Video-MME (medium)", "video_mme_medium"),
    ("Video-MME (long)", "video_mme_long"),
    ("Video-MME (overall)", "video_mme_overall"),
    ("LongVideoBench", "longvideobench"),
    ("LVBench", "lvbench"),
    ("TOMATO", "tomato"),
    ("MotionBench", "motionbench"),
    ("TVBench", "tvbench"),
    ("TempCompass", "tempcompass"),
    ("VideoMMMU", "videommmu"),
    ("MMVU (overall)", "mmvu"),
    ("Minerva", "minerva"),
    ("TimeLens-Charades", "timelens_charades"),
    ("TimeLens-ActivityNet", "timelens_activitynet"),
    ("TimeLens-QVHighlights", "timelens_qvhighlights"),
    ("VUE-TR", "vue_tr"),
    ("VUE-TR-V2", "vue_tr_v2"),
    ("MomentSeeker", "momentseeker"),
    ("MMBench_DEV_EN_V11", "mmbench"),
    ("RealWorldQA", "realworldqa"),
    ("AI2D_TEST(w. M.)", "ai2d"),
    ("MMStar", "mmstar"),
    ("MMMU_DEV_VAL", "mmmu"),
    ("OCRBench", "ocrbench"),
    ("ChartQA_TEST", "chartqa"),
    ("DocVQA_VAL", "docvqa"),
    ("InfoVQA_VAL", "infovqa"),
    ("MathVista_MINI", "mathvista"),
    ("MathVision_MINI", "mathvision"),
]

# (substring in filename, exclude substrings)
FILE_PATTERNS = {
    "video_mme_short": (["Video-MME_short", "Video-MME-short"], []),
    "video_mme_medium": (["Video-MME_medium", "Video-MME-medium"], []),
    "video_mme_long": (["Video-MME_long", "Video-MME-long"], []),
    "video_mme_overall": (
        ["Video-MME"],
        ["Video-MME_short", "Video-MME-short", "Video-MME_medium",
         "Video-MME-medium", "Video-MME_long", "Video-MME-long"],
    ),
    "longvideobench": (["LongVideoBench"], []),
    "lvbench": (["LVBench"], []),
    "tomato": (["TOMATO"], []),
    "motionbench": (["MotionBench"], []),
    "tvbench": (["TVBench"], []),
    "tempcompass": (["TempCompass"], []),
    "videommmu": (["VideoMMMU"], []),
    "mmvu": (["MMVU"], []),
    "minerva": (["Minerva"], []),
    "timelens_charades": (["TimeLens_Charades", "TimeLens-Charades"], []),
    "timelens_activitynet": (["TimeLens_ActivityNet", "TimeLens-ActivityNet"], []),
    "timelens_qvhighlights": (["TimeLens_QVHighlights", "TimeLens-QVHighlights"], []),
    "vue_tr": (["VUE_TR"], ["VUE_TR_V2"]),
    "vue_tr_v2": (["VUE_TR_V2"], []),
    "momentseeker": (["MomentSeeker"], []),
    "mmbench": (["MMBench_DEV_EN_V11"], []),
    "realworldqa": (["RealWorldQA"], []),
    "ai2d": (["AI2D_TEST"], []),
    "mmstar": (["MMStar"], []),
    "mmmu": (["MMMU_DEV_VAL"], []),
    "ocrbench": (["OCRBench"], []),
    "chartqa": (["ChartQA_TEST"], []),
    "docvqa": (["DocVQA_VAL"], []),
    "infovqa": (["InfoVQA_VAL"], []),
    "mathvista": (["MathVista_MINI"], []),
    "mathvision": (["MathVision_MINI"], []),
}

RESULT_SUFFIXES = (".json", ".csv")


def collect_files(root: Path) -> list[Path]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(RESULT_SUFFIXES):
                files.append(Path(dirpath) / name)
    return files


def match_file(files: list[Path], includes: list[str], excludes: list[str]) -> Path | None:
    candidates = []
    for f in files:
        name = f.name
        if not any(pat in name for pat in includes):
            continue
        if any(pat in name for pat in excludes):
            continue
        candidates.append(f)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return float(val)
    s = str(val).strip().replace("%", "")
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_percent(val: float | None) -> float | None:
    if val is None:
        return None
    if 0 <= val <= 1:
        return val * 100
    return val


def format_score(val: float | None, *, integer: bool = False) -> str:
    if val is None:
        return ""
    if integer:
        return str(int(round(val)))
    return f"{val:.1f}"


def read_csv_rows(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def find_csv_value(rows: list[list[str]], row_key: str, col_key: str = "Overall") -> float | None:
    if not rows:
        return None
    header = [h.strip().strip('"') for h in rows[0]]
    try:
        col_idx = header.index(col_key)
    except ValueError:
        col_idx = len(header) - 1

    for row in rows[1:]:
        if not row:
            continue
        cells = [c.strip().strip('"') for c in row]
        if cells[0].lower() == row_key.lower():
            if col_idx < len(cells):
                return parse_float(cells[col_idx])
    return None


def find_csv_last_metric_row(path: Path, metric_col: str = "Overall") -> float | None:
    """For metrics CSV with header row and percentage row at the end (TOMATO, VideoMMMU)."""
    rows = read_csv_rows(path)
    if len(rows) < 2:
        return None
    header = [h.strip().strip('"') for h in rows[0]]
    try:
        col_idx = header.index(metric_col)
    except ValueError:
        return None
    for row in reversed(rows[1:]):
        cells = [c.strip().strip('"') for c in row]
        if col_idx < len(cells):
            val = parse_float(cells[col_idx])
            if val is not None:
                return val
    return None


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_video_mme_duration(path: Path, duration: str) -> float | None:
    data = load_json(path)
    block = data.get(duration, {})
    return parse_float(block.get("overall"))


def extract_video_mme_overall(files: list[Path], root: Path) -> float | None:
    includes, excludes = FILE_PATTERNS["video_mme_overall"]
    f = match_file(files, includes, excludes)
    if f is not None:
        data = load_json(f)
        val = parse_float(data.get("overall", {}).get("overall"))
        if val is not None:
            return val

    parts = []
    for key, duration in [
        ("video_mme_short", "short"),
        ("video_mme_medium", "medium"),
        ("video_mme_long", "long"),
    ]:
        inc, exc = FILE_PATTERNS[key]
        pf = match_file(files, inc, exc)
        if pf is None:
            continue
        v = extract_video_mme_duration(pf, duration)
        if v is not None:
            parts.append(v)
    if parts:
        return sum(parts) / len(parts)
    return None


def extract_timelens_miou(path: Path) -> float | None:
    data = load_json(path)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and "miou" in v:
                return parse_float(v["miou"])
    return None


def extract_iou_json(path: Path, keys=("miou", "tIoU", "mIoU")) -> float | None:
    data = load_json(path)

    def search(obj):
        if isinstance(obj, dict):
            if "overall" in obj and isinstance(obj["overall"], dict):
                for k in keys:
                    if k in obj["overall"]:
                        return parse_float(obj["overall"][k])
            for k in keys:
                if k in obj:
                    return parse_float(obj[k])
            for v in obj.values():
                r = search(v)
                if r is not None:
                    return r
        return None

    return search(data)


def extract_benchmark(key: str, path: Path, all_files: list[Path]) -> float | None:
    name = path.name

    if key == "video_mme_short":
        return extract_video_mme_duration(path, "short")
    if key == "video_mme_medium":
        return extract_video_mme_duration(path, "medium")
    if key == "video_mme_long":
        return extract_video_mme_duration(path, "long")
    if key == "video_mme_overall":
        return extract_video_mme_overall(all_files, path.parent)

    if key == "longvideobench":
        data = load_json(path)
        block = data.get("overall", {})
        return parse_float(block.get("overall"))

    if key == "lvbench":
        rows = read_csv_rows(path)
        for row in rows[1:]:
            cells = [c.strip().strip('"') for c in row]
            if cells and cells[0].lower() == "acc":
                return parse_float(cells[1] if len(cells) > 1 else None)
        return None

    if key in ("tomato", "videommmu", "mmvu"):
        col = "Overall"
        return find_csv_last_metric_row(path, col)

    if key == "motionbench":
        data = load_json(path)
        block = data.get("MotionBench", data)
        if isinstance(block, dict):
            return parse_float(block.get("overall_accuracy"))
        return None

    if key == "tvbench":
        data = load_json(path)
        overall = data.get("overall")
        if isinstance(overall, list) and len(overall) >= 3:
            return parse_float(overall[2])
        return None

    if key == "tempcompass":
        rows = read_csv_rows(path)
        header = [h.strip().strip('"') for h in rows[0]] if rows else []
        acc_idx = header.index("acc") if "acc" in header else -1
        for row in rows[1:]:
            cells = [c.strip().strip('"') for c in row]
            if cells and cells[0].lower() == "overall" and acc_idx >= 0 and acc_idx < len(cells):
                return parse_float(cells[acc_idx])
        return None

    if key == "minerva":
        data = load_json(path)
        return parse_float(data.get("overall_accuracy"))

    if key in ("timelens_charades", "timelens_activitynet", "timelens_qvhighlights"):
        return extract_timelens_miou(path)

    if key in ("vue_tr", "vue_tr_v2", "momentseeker"):
        return extract_iou_json(path)

    if key == "ocrbench":
        data = load_json(path)
        if "Final Score" in data:
            return parse_float(data["Final Score"])
        return None

    if key == "mmmu":
        rows = read_csv_rows(path)
        return find_csv_value(rows, "validation", "Overall")

    if key == "chartqa":
        rows = read_csv_rows(path)
        if len(rows) >= 2:
            header = [h.strip().strip('"') for h in rows[0]]
            if "Overall" in header:
                idx = header.index("Overall")
                if idx < len(rows[1]):
                    return parse_float(rows[1][idx])
        return None

    if key in ("docvqa", "infovqa"):
        rows = read_csv_rows(path)
        v = find_csv_value(rows, "val", "Overall") or find_csv_value(rows, "none", "Overall")
        if v is not None:
            return v
        if len(rows) >= 2:
            header = [h.strip().strip('"') for h in rows[0]]
            if "Overall" in header:
                idx = header.index("Overall")
                if idx < len(rows[1]):
                    return parse_float(rows[1][idx])
        return None

    if key in ("mathvista", "mathvision"):
        rows = read_csv_rows(path)
        header = [h.strip().strip('"') for h in rows[0]] if rows else []
        acc_idx = header.index("acc") if "acc" in header else -1
        for row in rows[1:]:
            cells = [c.strip().strip('"') for c in row]
            if cells and cells[0] == "Overall" and acc_idx >= 0 and acc_idx < len(cells):
                return parse_float(cells[acc_idx])
        return None

    # Default: *_acc.csv with split / Overall
    rows = read_csv_rows(path)
    return (
        find_csv_value(rows, "none", "Overall")
        or find_csv_value(rows, "dev", "Overall")
        or find_csv_value(rows, "val", "Overall")
    )


def get_score(key: str, files: list[Path]) -> tuple[float | None, bool]:
    """Return (raw_value, is_ocr_integer)."""
    includes, excludes = FILE_PATTERNS[key]
    if key == "video_mme_overall":
        val = extract_video_mme_overall(files, Path("."))
        return (to_percent(val), False)

    path = match_file(files, includes, excludes)
    if path is None:
        return (None, key == "ocrbench")

    raw = extract_benchmark(key, path, files)
    if key == "ocrbench":
        return (raw, True)
    return (to_percent(raw), False)


def write_summary(input_dir: Path, output_path: Path) -> None:
    files = collect_files(input_dir)
    rows_out = [["dataset", "score"]]

    for display_name, key in BENCHMARKS:
        raw, is_ocr = get_score(key, files)
        rows_out.append([display_name, format_score(raw, integer=is_ocr)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out) - 1} benchmarks to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize VLMEvalKit benchmark results into a CSV table."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Output folder, e.g. outputs/VideoChat3_4B_train_stage4_128k_long_v1",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_dir>/summary.csv)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    output = args.output
    if output is None:
        output = input_dir / f"summary.csv"
    else:
        output = output.resolve()

    write_summary(input_dir, output)


if __name__ == "__main__":
    main()
