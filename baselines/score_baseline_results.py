#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Score the baselines with the repository metrics and the budget rule.

This script is intentionally experiment-local.  It does not modify the baseline
decoders or the metric implementations in ``src/metric``.
"""

import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# svamp_score imports its sibling module amc23_score by name.
sys.path.insert(0, str(REPO_ROOT / "src" / "metric"))

from glr_paths import get  # noqa: E402
from src.metric import gsm8k_accuracy, minerva_math500_score, svamp_score  # noqa: E402

EXPECTED = {"gsm8k": 1319, "math500": 500, "svamp": 300}
BUDGETS = {
    "gsm8k": (512, 1024),
    "math500": (1024, 2048),
    "svamp": (1024,),
}


@dataclass(frozen=True)
class Run:
    family: str
    model_size: str
    benchmark: str
    path: Path
    pred_key: str
    latent_steps: int

    @property
    def label(self) -> str:
        return f"{self.family} ({self.model_size})"


SOFT_ROOT = Path(get("baseline_dir")) / "soft_thinking"
HSR_ROOT = Path(get("baseline_dir")) / "coconut_hsr_qwen3-0.6B"
HSR_CHECKPOINT = Path(get("ckpt_dir")) / "coconut_hsr_qwen3-0.6B"


RUNS = (
    [
        Run(
            "Soft Thinking",
            "0.6B",
            benchmark,
            SOFT_ROOT / "qwen3-0.6B" / f"{benchmark}_softthink_fixed_k20.jsonl",
            "generated_text",
            20,
        )
        for benchmark in EXPECTED
    ]
    + [
        Run(
            "Soft Thinking",
            "1.7B",
            benchmark,
            SOFT_ROOT / "qwen3-1.7B" / f"{benchmark}_softthink_fixed_k10.jsonl",
            "generated_text",
            10,
        )
        for benchmark in EXPECTED
    ]
    + [
        Run(
            "Soft Thinking unended",
            model_size,
            benchmark,
            SOFT_ROOT / model_dir / f"{benchmark}_softthink_until_close_kmax256.jsonl",
            "generated_text",
            0,
        )
        for model_size, model_dir in (("0.6B", "qwen3-0.6B"), ("1.7B", "qwen3-1.7B"))
        for benchmark in EXPECTED
    ]
    + [
        Run(
            f"Coconut-style K={k}",
            "0.6B",
            benchmark,
            HSR_ROOT
            / (
                f"minerva_math500_zeroshot_latent{k}_gen4096_temp0.0.jsonl"
                if benchmark == "math500"
                else (
                    f"gsm8k_zeroshot_latent{k}_gen2048_temp0.0.jsonl"
                    if benchmark == "gsm8k"
                    else f"svamp_latent{k}_gen2048_temp0.0.jsonl"
                )
            ),
            "predicted",
            k,
        )
        for k in (10, 100)
        for benchmark in EXPECTED
    ]
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    questions = [str(row.get("question", "")) for row in rows]
    if len(questions) != len(set(questions)):
        raise ValueError(f"{path}: duplicate questions")
    return rows


def extract_assistant(text: str) -> str:
    delimiter = "<|im_start|>assistant\n"
    if delimiter in text:
        return text.split(delimiter)[-1].strip()
    marker = "<|im_start|>assistant"
    if marker in text:
        return text.split(marker)[-1].strip()
    return text.strip()


def score(run: Run) -> list[bool]:
    if run.benchmark == "gsm8k":
        results, _ = gsm8k_accuracy.score_file(
            run.path,
            pred_key=run.pred_key,
            gold_key="answer",
        )
        return [bool(result.correct) for result in results]
    if run.benchmark == "math500":
        results, _ = minerva_math500_score.score_file(
            run.path,
            pred_key=run.pred_key,
            gold_key="answer",
            question_key="question",
            compute_math_verify=True,
        )
    else:
        results, _ = svamp_score.core.score_file(
            run.path,
            pred_key=run.pred_key,
            gold_key="answer",
            question_key="question",
            extraction_mode="flexible-extract",
            compute_math_verify=True,
        )
    return [bool(result.math_verify) for result in results]


def total_steps(
    run: Run,
    row: dict[str, Any],
    tokenizer: Any,
) -> int:
    if run.family.startswith("Soft Thinking"):
        return int(row["total_sequential_steps"])
    assistant = extract_assistant(str(row[run.pred_key]))
    text_steps = len(tokenizer.encode(assistant, add_special_tokens=False))
    # The HSR decoder begins feedback only after the model emits <think>.
    realized_latent_steps = run.latent_steps if "<think>" in assistant else 0
    return realized_latent_steps + text_steps


def main() -> None:
    output_dir = Path(get("baseline_dir")) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(HSR_CHECKPOINT, trust_remote_code=True)
    records: list[dict[str, Any]] = []

    for run in RUNS:
        if not run.path.exists():
            continue
        rows = read_jsonl(run.path)
        correctness = score(run)
        if len(rows) != len(correctness):
            raise ValueError(
                f"{run.path}: {len(rows)} rows but {len(correctness)} scores"
            )
        steps = [total_steps(run, row, tokenizer) for row in rows]
        latent_phase_entries = (
            sum("<think>" in extract_assistant(str(row[run.pred_key])) for row in rows)
            if run.family.startswith("Coconut-style")
            else len(rows)
        )
        correct_steps = [
            length for length, correct in zip(steps, correctness) if correct
        ]
        record: dict[str, Any] = {
            "family": run.family,
            "model_size": run.model_size,
            "benchmark": run.benchmark,
            "latent_steps": run.latent_steps,
            "observed_n": len(rows),
            "expected_n": EXPECTED[run.benchmark],
            "complete": len(rows) == EXPECTED[run.benchmark],
            "latent_phase_entry_rate": latent_phase_entries / len(rows),
            "observed_accuracy": sum(correctness) / len(correctness),
            "median_total_steps_correct": (
                statistics.median(correct_steps) if correct_steps else None
            ),
            "source": str(run.path),
        }
        for budget in BUDGETS[run.benchmark]:
            record[f"accuracy_at_{budget}"] = sum(
                correct and length <= budget
                for correct, length in zip(correctness, steps)
            ) / len(rows)
        records.append(record)
        print(
            f"{run.label} {run.benchmark}: "
            f"{len(rows)}/{EXPECTED[run.benchmark]}, "
            f"acc={100 * record['observed_accuracy']:.1f}%, "
            f"median={record['median_total_steps_correct']}"
        )

    json_path = output_dir / "baseline_budget_results.json"
    json_path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "baseline_budget_results.csv"
    fields = sorted({key for row in records for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
