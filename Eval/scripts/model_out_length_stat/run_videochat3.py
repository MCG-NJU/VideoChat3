#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count model-output tokens in VLMEvalKit Excel result files.

Example:

python -u scripts/model_out_length_stat/run_videochat3.py \
  --model_path MCG-NJU/VideoChat3-4B \
  --excel /path/to/result.xlsx
"""

import argparse
import os
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer

def _pick_output_column(df: pd.DataFrame) -> str:
    # 目前你给的三个数据集都是 prediction
    preferred = ["prediction", "pred", "model_output", "output", "response", "generated", "text"]
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for key in preferred:
        if key in cols_lower:
            return str(cols_lower[key])
    raise KeyError(
        f"找不到模型输出列。现有列：{list(df.columns)}。"
        f"请确认输出列名（目前支持优先匹配：{preferred}）"
    )


def _iter_batches(items: List[str], batch_size: int) -> Iterable[Tuple[int, List[str]]]:
    for i in range(0, len(items), batch_size):
        yield i, items[i : i + batch_size]


def count_output_tokens_in_excel(
    excel_path: str,
    tokenizer,
    sheet_name: int | str = 0,
    output_col: Optional[str] = None,
    batch_size: int = 128,
    add_special_tokens: bool = False,
) -> Tuple[List[int], str, int]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if output_col is None:
        output_col = _pick_output_column(df)

    # 只统计“模型输出”本身的 token 数（不包含任何 chat 模板前后缀）。
    texts_series = df[output_col]
    texts: List[str] = []
    for x in texts_series.tolist():
        if x is None or (isinstance(x, float) and pd.isna(x)):
            texts.append("")
        else:
            texts.append(str(x))

    lengths: List[int] = [0] * len(texts)
    for start, batch in _iter_batches(texts, batch_size=batch_size):
        enc = tokenizer(
            batch,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        batch_lens = [len(ids) for ids in enc["input_ids"]]
        lengths[start : start + len(batch_lens)] = batch_lens

    return lengths, output_col, len(df)


def _print_stats(name: str, lens: List[int], n_rows: int, excel_path: str, output_col: str) -> None:
    if n_rows == 0:
        print(f"[{name}] 空表：{excel_path}")
        return

    total = sum(lens)
    mx = max(lens) if lens else 0
    mn = min(lens) if lens else 0
    mean = total / max(1, len(lens))

    lens_without_repeat = [x for x in lens if x < 4000]
    mean_without_repeat = sum(lens_without_repeat) / max(1, len(lens_without_repeat))

    print("=" * 90)
    # print(sorted(lens))
    print(f"[{name}] {os.path.basename(excel_path)}")
    print(f"- excel_path: {excel_path}")
    print(f"- output_col: {output_col}")
    print(f"- rows: {n_rows}")
    # print(f"- tokens_total: {total}")
    print(f"- tokens_mean: {mean:.2f}")
    print(f"- tokens_mean_without_repeat: {mean_without_repeat:.2f}")
    print(f"- tokens_min: {mn}")
    print(f"- tokens_max: {mx}")
    print(f"- repeat_ratio: {(1-len(lens_without_repeat)/len(lens))*100:.2f}%")

def main() -> None:
    parser = argparse.ArgumentParser(description="Count model-output tokens in .xlsx result files")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("VIDEOCHAT3_MODEL_PATH", "").strip() or "MCG-NJU/VideoChat3-4B",
        help="Local tokenizer path or Hugging Face model ID (default: MCG-NJU/VideoChat3-4B).",
    )
    parser.add_argument(
        "--excel",
        type=str,
        nargs="+",
        required=True,
        help="One or more VLMEvalKit .xlsx result files.",
    )
    parser.add_argument("--sheet", type=str, default="0", help="sheet 名或 index（默认 0）")
    parser.add_argument("--output_col", type=str, default=None, help="强制指定输出列名（默认自动探测）")
    parser.add_argument("--batch_size", type=int, default=128, help="tokenize batch size（默认 128）")
    parser.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="是否在计数时加入 special tokens（默认不加，更贴近“输出文本本身”的 token 数）",
    )
    args = parser.parse_args()

    # sheet 参数：允许传 "0" / "1" 等数字字符串，也允许传 sheet 名
    sheet: int | str
    if isinstance(args.sheet, str) and args.sheet.strip().isdigit():
        sheet = int(args.sheet.strip())
    else:
        sheet = args.sheet

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    for p in args.excel:
        if not os.path.exists(p):
            print(f"[SKIP] excel 不存在：{p}")
            continue
        lens, output_col, n_rows = count_output_tokens_in_excel(
            excel_path=p,
            tokenizer=tokenizer,
            sheet_name=sheet,
            output_col=args.output_col,
            batch_size=args.batch_size,
            add_special_tokens=args.add_special_tokens,
        )
        _print_stats(name="token_count", lens=lens, n_rows=n_rows, excel_path=p, output_col=output_col)


if __name__ == "__main__":
    main()
