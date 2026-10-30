# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Score the cached mechanistic controls with the GLR benchmark evaluators.

This is a model-free analysis.  It reads the already cached generations and
uses the repository metric implementations and correctness fields:

- GSM8K: flexible numeric exact match from ``src/metric/gsm8k_accuracy.py``;
- MATH500: ``math_verify`` from ``src/metric/minerva_math500_score.py``;
- SVAMP: ``math_verify`` from ``src/metric/amc23_score.py``.

All methods are GLR inference variants, so the GLR-side evaluator is
used for every control.  Total steps are the cached post-prompt count,
including the latent positions.
"""

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from glr_paths import get  # noqa: E402

DEFAULT_ROOT = Path(get("analysis_dir")) / "mechanistic"
METHODS = (
    "learned",
    "zero_transition",
    "repeated_think",
    "random_head_norm_matched",
    "noisy_transition",
)
METHOD_LABELS = {
    "learned": "Learned transition",
    "zero_transition": "Zero transition",
    "repeated_think": "Repeated `<think>`",
    "random_head_norm_matched": "Norm-matched random head",
    "noisy_transition": "Noisy transition",
}
DATASETS = ("gsm8k", "math500", "svamp")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260727


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def metric_module(name: str):
    metric_dir = REPO_ROOT / "src" / "metric"
    for path in (str(REPO_ROOT), str(metric_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module(name)


def run_dir(root: Path, dataset: str, method: str) -> Path:
    base = root / "runs" / dataset / method / "K=10"
    if method == "noisy_transition":
        base = base / "sigma=1"
    return base / "seed=0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_records(dataset: str, records: list[dict[str, Any]]) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="glr_control_rescore_") as temp:
        prediction_path = Path(temp) / f"{dataset}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {
                            "generated_text": record["generated_text"],
                            "gold_answer": record["gold_answer"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if dataset == "gsm8k":
            metric = metric_module("gsm8k_accuracy")
            results, _ = metric.score_file(
                prediction_path,
                pred_key="generated_text",
                gold_key="gold_answer",
                extraction_mode="flexible-extract",
            )
            return np.asarray([bool(item.correct) for item in results])

        if dataset == "math500":
            metric = metric_module("minerva_math500_score")
            results, _ = metric.score_file(
                prediction_path,
                pred_key="generated_text",
                gold_key="gold_answer",
                compute_math_verify=True,
            )
            return np.asarray([bool(item.math_verify) for item in results])

        if dataset == "svamp":
            metric = metric_module("amc23_score")
            results, _ = metric.score_file(
                prediction_path,
                pred_key="generated_text",
                gold_key="gold_answer",
                extraction_mode="flexible-extract",
                compute_math_verify=True,
            )
            return np.asarray([bool(item.math_verify) for item in results])

        raise KeyError(dataset)


def bootstrap_accuracy_delta(
    learned: pd.DataFrame, control: pd.DataFrame, rng: np.random.Generator
) -> tuple[float, float, float]:
    keys = ["dataset", "example_id"]
    paired = learned[keys + ["metric_correct"]].merge(
        control[keys + ["metric_correct"]],
        on=keys,
        suffixes=("_learned", "_control"),
        validate="one_to_one",
    )
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    dataset_arrays = []
    for dataset in DATASETS:
        group = paired[paired.dataset == dataset]
        dataset_arrays.append(
            (
                group.metric_correct_learned.to_numpy(dtype=float),
                group.metric_correct_control.to_numpy(dtype=float),
            )
        )
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_differences = []
        for learned_correct, control_correct in dataset_arrays:
            indices = rng.integers(0, len(learned_correct), len(learned_correct))
            sampled_differences.append(
                control_correct[indices] - learned_correct[indices]
            )
        deltas[replicate] = 100 * np.concatenate(sampled_differences).mean()
    estimate = 100 * (
        paired.metric_correct_control.mean() - paired.metric_correct_learned.mean()
    )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(estimate), float(low), float(high)


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    summary = pd.read_parquet(root / "derived" / "per_example_summary.parquet")

    cached: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    control_ids: dict[str, set[str]] = {}
    for dataset in DATASETS:
        method_maps = {}
        for method in METHODS:
            records = read_jsonl(run_dir(root, dataset, method) / "metadata.jsonl")
            mapping = {str(record["example_id"]): record for record in records}
            if len(mapping) != len(records):
                raise ValueError(f"{dataset}/{method}: duplicate example IDs")
            method_maps[method] = mapping
        ids = set(method_maps["zero_transition"])
        for method in METHODS[1:]:
            ids &= set(method_maps[method])
        if len(ids) != 100:
            raise ValueError(
                f"{dataset}: expected 100 shared control IDs, found {len(ids)}"
            )
        cached.update(
            {(dataset, method): mapping for method, mapping in method_maps.items()}
        )
        control_ids[dataset] = ids

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        ids = sorted(control_ids[dataset])
        for method in METHODS:
            records = [cached[(dataset, method)][example_id] for example_id in ids]
            correctness = score_records(dataset, records)
            if len(correctness) != len(records):
                raise ValueError(f"{dataset}/{method}: scorer row-count mismatch")
            lookup = summary[
                (summary.dataset == dataset)
                & (summary.method == method)
                & (~summary.forced_continuation.astype(bool))
            ].copy()
            lookup["example_id"] = lookup.example_id.astype(str)
            lookup = lookup.set_index("example_id")
            for record, correct in zip(records, correctness):
                example_id = str(record["example_id"])
                if example_id not in lookup.index:
                    raise KeyError(
                        f"{dataset}/{method}/{example_id}: missing derived row"
                    )
                derived = lookup.loc[example_id]
                rows.append(
                    {
                        "dataset": dataset,
                        "example_id": example_id,
                        "method": method,
                        "metric_correct": bool(correct),
                        "total_steps": int(record["generated_token_count"]),
                        "immediate_close": bool(derived["immediate_close"]),
                        "correctness_metric": (
                            "flexible_numeric_exact_match"
                            if dataset == "gsm8k"
                            else "math_verify"
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    if len(frame) != 3 * 5 * 100:
        raise ValueError(f"Expected 1,500 rows, found {len(frame)}")

    aggregates: list[dict[str, Any]] = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=False):
        correct_steps = group.loc[group.metric_correct, "total_steps"]
        aggregates.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(group),
                "accuracy_percent": 100 * group.metric_correct.mean(),
                "median_total_steps_all": float(group.total_steps.median()),
                "median_total_steps_correct": float(correct_steps.median()),
                "immediate_close_percent": 100 * group.immediate_close.mean(),
            }
        )
    for method, group in frame.groupby("method", sort=False):
        correct_steps = group.loc[group.metric_correct, "total_steps"]
        aggregates.append(
            {
                "dataset": "pooled",
                "method": method,
                "n": len(group),
                "accuracy_percent": 100 * group.metric_correct.mean(),
                "median_total_steps_all": float(group.total_steps.median()),
                "median_total_steps_correct": float(correct_steps.median()),
                "immediate_close_percent": 100 * group.immediate_close.mean(),
            }
        )

    aggregate = pd.DataFrame(aggregates)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    learned = frame[frame.method == "learned"]
    deltas = []
    for method in METHODS[1:]:
        estimate, low, high = bootstrap_accuracy_delta(
            learned, frame[frame.method == method], rng
        )
        deltas.append(
            {
                "method": method,
                "accuracy_delta_control_minus_learned_pp": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    delta_frame = pd.DataFrame(deltas)

    output_dir = root / "derived"
    frame.to_csv(output_dir / "scored_control_examples.csv", index=False)
    aggregate.to_csv(output_dir / "scored_control_aggregates.csv", index=False)
    delta_frame.to_csv(
        output_dir / "scored_control_paired_deltas.csv", index=False
    )

    pooled = aggregate[aggregate.dataset == "pooled"].set_index("method")
    lines = [
        "| Method | Accuracy (%) | Median total steps, all examples | Immediate close (%) | Accuracy delta vs. learned (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    delta_by_method = delta_frame.set_index("method")
    for method in METHODS:
        row = pooled.loc[method]
        if method == "learned":
            delta = "—"
        else:
            item = delta_by_method.loc[method]
            delta = (
                f"{item.accuracy_delta_control_minus_learned_pp:+.1f} "
                f"[{item.ci95_low:+.1f}, {item.ci95_high:+.1f}]"
            )
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{row.accuracy_percent:.1f} | "
            f"{row.median_total_steps_all:.1f} | "
            f"{row.immediate_close_percent:.1f} | {delta} |"
        )
    lines.extend(
        [
            "",
            "*Note:* The fixed control subset contains the same 100 examples per "
            "dataset (300 paired examples total). Accuracy uses the GLR-side "
            "benchmark evaluator from `src/metric`: flexible numeric exact match on "
            "GSM8K and `math_verify` on MATH500/SVAMP. Total steps use the "
            "post-prompt accounting and include all ten latent positions. Brackets "
            "are 95% paired, dataset-stratified bootstrap confidence intervals "
            f"({BOOTSTRAP_REPLICATES:,} resamples).",
        ]
    )
    rendered = root / "rendered" / "scored_control_table.md"
    rendered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
