#!/usr/bin/env python3
"""Aggregate VLMEvalKit video grounding benchmark results into a summary CSV."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

# (display_name, key, metric)
BENCHMARKS = [
    ("TimeLens-Charades", "timelens_charades", "miou"),
    ("TimeLens-ActivityNet", "timelens_activitynet", "miou"),
    ("TimeLens-QVHighlights", "timelens_qvhighlights", "miou"),
    ("Vidi-Bench (IoU auc)", "vidi_bench", "iou_auc"),
    ("Vidi-Bench-v2 (IoU auc)", "vidi_bench_v2", "iou_auc"),
    ("MomentSeeker", "momentseeker", "miou"),
    ("Ego4D-NLQ", "ego4d_nlq", "miou"),
]

FILE_PATTERNS = {
    "timelens_charades": (["TimeLens_Charades", "TimeLens-Charades"], []),
    "timelens_activitynet": (["TimeLens_ActivityNet", "TimeLens-ActivityNet"], []),
    "timelens_qvhighlights": (["TimeLens_QVHighlights", "TimeLens-QVHighlights"], []),
    "vidi_bench": (["VUE_TR"], ["VUE_TR_V2"]),
    "vidi_bench_v2": (["VUE_TR_V2"], []),
    "momentseeker": (["MomentSeeker"], []),
    "ego4d_nlq": (["Ego4D-NLQ"], []),
}

RATING_SUFFIX = "_rating.json"
SCORE_SUFFIX = "_score.json"
RESULT_SUFFIXES = (RATING_SUFFIX, SCORE_SUFFIX, ".json", ".csv")


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


def format_score(val: float | None) -> str:
    if val is None:
        return ""
    return f"{val:.1f}"


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_timelens_miou(path: Path) -> float | None:
    data = load_json(path)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and "miou" in v:
                return parse_float(v["miou"])
    return None


def extract_overall_metric(path: Path, metric: str) -> float | None:
    data = load_json(path)
    if not isinstance(data, dict):
        return None

    overall = data.get("overall")
    if isinstance(overall, dict):
        for key in (metric, metric.lower(), metric.upper()):
            if key in overall:
                return parse_float(overall[key])
        if metric == "miou":
            for key in ("mIoU", "miou", "tIoU", "iou_auc"):
                if key in overall:
                    return parse_float(overall[key])

    for key in (metric, metric.lower(), metric.upper()):
        if key in data:
            return parse_float(data[key])
    return None


def extract_score(key: str, metric: str, path: Path) -> float | None:
    if key.startswith("timelens_"):
        if metric != "miou":
            return None
        return extract_timelens_miou(path)

    if key in ("vidi_bench", "vidi_bench_v2", "momentseeker", "ego4d_nlq"):
        return extract_overall_metric(path, metric)

    return None


def get_score(key: str, metric: str, files: list[Path]) -> float | None:
    includes, excludes = FILE_PATTERNS[key]
    path = match_file(files, includes, excludes)
    if path is None:
        return None
    raw = extract_score(key, metric, path)
    return to_percent(raw)


def write_summary(input_dir: Path, output_path: Path) -> None:
    files = collect_files(input_dir)
    rows_out = [["dataset", "score"]]

    for display_name, key, metric in BENCHMARKS:
        score = get_score(key, metric, files)
        rows_out.append([display_name, format_score(score)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out) - 1} benchmarks to {output_path}")
    for row in rows_out[1:]:
        print(f"  {row[0]}: {row[1] or 'N/A'}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize VLMEvalKit video grounding benchmark results into a CSV table."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Output folder, e.g. outputs/Qwen3-VL-4B_lvdb_35k_..._ego4d",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_dir>/grounding_summary.csv)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    output = args.output
    if output is None:
        output = input_dir / "grounding_summary.csv"
    else:
        output = output.resolve()

    write_summary(input_dir, output)


if __name__ == "__main__":
    main()
