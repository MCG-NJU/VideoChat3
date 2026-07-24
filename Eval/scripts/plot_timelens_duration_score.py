#!/usr/bin/env python3
"""Plot score vs ground-truth segment duration for TimeLens result files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def binned_mean_curve(x: np.ndarray, y: np.ndarray, n_bins: int = 30):
    """Return bin centers and mean y per bin for a smooth summary curve."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if i == n_bins - 1:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        means.append(np.mean(y[m]) if np.any(m) else np.nan)
    means = np.array(means)
    valid = np.isfinite(means)
    return centers[valid], means[valid]


def plot_one(path: str, title: str, out_path: Path) -> None:
    df = pd.read_excel(path)
    gt_dur = (df["gt_end"] - df["gt_start"]).astype(float)
    score = df["score"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.scatter(gt_dur, score, alpha=0.25, s=12, c="steelblue", edgecolors="none", label="samples")

    cx, cy = binned_mean_curve(gt_dur.values, score.values, n_bins=40)
    if len(cx):
        ax.plot(cx, cy, color="darkred", linewidth=2, label="binned mean (40 bins, 1–99% x range)")

    ax.set_xlabel("Ground truth duration (gt_end − gt_start, seconds)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TimeLens score against ground-truth segment duration."
    )
    parser.add_argument("files", type=Path, nargs="+", help="TimeLens *_score.xlsx files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: next to each input file).",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for p in args.files:
        if not p.is_file():
            print(f"Skip (missing): {p}")
            continue
        out_dir = args.output_dir or p.parent
        out_path = out_dir / f"{p.stem}_duration_vs_score.png"
        plot_one(str(p), p.stem, out_path)


if __name__ == "__main__":
    main()
