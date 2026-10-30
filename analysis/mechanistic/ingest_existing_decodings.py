# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Validate and summarize the scored greedy decodes at the token caps."""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from mechanistic_common import (
    DATASETS,
    DEFAULT_CACHE_ROOT,
    GLR_CHECKPOINT,
    SFT_CHECKPOINT,
    extract_assistant_turn,
    generation_finished,
    load_dataset,
    marked_correct,
    read_jsonl,
)

BUDGETS = {
    "gsm8k": (64, 128, 256, 512, 1024, 2048),
    "math500": (128, 256, 512, 1024, 2048, 4096),
    "svamp": (64, 128, 256, 512, 1024, 2048),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_CACHE_ROOT / "rendered"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizers = {
        "CoT-SFT": AutoTokenizer.from_pretrained(SFT_CHECKPOINT),
        "GLR K=10": AutoTokenizer.from_pretrained(GLR_CHECKPOINT),
    }
    details: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        gold = load_dataset(dataset)
        sources = {
            "CoT-SFT": (spec.sft_decode, "decoded_output"),
            "GLR K=10": (spec.glr_k10_decode, "predicted"),
        }
        for method, (relative, text_field) in sources.items():
            path = ANALYSIS_DIR.parents[1] / relative
            rows = list(read_jsonl(path))
            if len(rows) != len(gold):
                raise ValueError(f"{path}: {len(rows)} rows != {len(gold)} test rows")
            tokenizer = tokenizers[method]
            for index, (record, target) in enumerate(zip(rows, gold)):
                source_prompt = record.get("q", record.get("question"))
                if source_prompt != target["q"]:
                    raise ValueError(f"{path}:{index+1}: prompt/order mismatch")
                text = str(record.get(text_field, ""))
                assistant = extract_assistant_turn(text)
                token_count = len(tokenizer.encode(assistant, add_special_tokens=False))
                finished = generation_finished(text)
                marker_method = "sft" if method == "CoT-SFT" else "glr"
                original_correct, correct_key = marked_correct(
                    dataset, record, marker_method
                )
                details.append(
                    {
                        "dataset": dataset,
                        "example_id": str(target.get("id", index)),
                        "method": method,
                        "token_count": token_count,
                        "cap": spec.cap,
                        "correct_key": correct_key,
                        "original_correct": original_correct,
                        "finished": finished,
                        "correct_at_cap": original_correct and finished and token_count <= spec.cap,
                        "hit_cap": token_count > spec.cap,
                        "source": str(path),
                    }
                )
    frame = pd.DataFrame(details)
    frame.to_parquet(
        args.output_dir / "existing_decodings_at_caps.parquet", index=False
    )
    for key, group in frame.groupby(["dataset", "method"], sort=False):
        summary.append(
            {
                "dataset": key[0],
                "method": key[1],
                "n": len(group),
                "cap": int(group.cap.iloc[0]),
                "full_accuracy": float(group.original_correct.mean()),
                "accuracy_at_cap": float(group.correct_at_cap.mean()),
                "median_tokens": float(group.token_count.median()),
                "median_tokens_correct_at_cap": float(
                    group[group.correct_at_cap].token_count.median()
                ),
                "over_cap_rate": float(group.hit_cap.mean()),
            }
        )
    table = pd.DataFrame(summary)
    table.to_csv(args.output_dir / "existing_decodings_at_caps.csv", index=False)
    budget_rows: list[dict[str, Any]] = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=False):
        for budget in BUDGETS[dataset]:
            budget_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "budget": budget,
                    "accuracy": float(
                        (group.original_correct & group.finished & (group.token_count <= budget)).mean()
                    ),
                    "n": len(group),
                }
            )
    budget_table = pd.DataFrame(budget_rows)
    budget_table.to_csv(
        args.output_dir / "existing_accuracy_by_budget.csv", index=False
    )
    lines = [
        "| Dataset | Method | Cap | Acc. at cap | Full acc. | Median tokens | Over cap | N |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.dataset.upper()} | {row.method} | {row.cap} | "
            f"{100*row.accuracy_at_cap:.1f}% | {100*row.full_accuracy:.1f}% | "
            f"{row.median_tokens:.0f} | {100*row.over_cap_rate:.1f}% | {row.n} |"
        )
    (args.output_dir / "existing_decodings_at_caps.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    budget_lines = [
        "| Dataset | Method | Budget | Accuracy | N |",
        "|---|---|---:|---:|---:|",
    ]
    for row in budget_table.itertuples():
        budget_lines.append(
            f"| {row.dataset.upper()} | {row.method} | {row.budget} | "
            f"{100*row.accuracy:.1f}% | {row.n} |"
        )
    (args.output_dir / "existing_accuracy_by_budget.md").write_text(
        "\n".join(budget_lines) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
