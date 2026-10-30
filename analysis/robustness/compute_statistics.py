# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""CPU-only statistical analysis and report rendering for saved generations."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROBUSTNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROBUSTNESS_DIR))

from robustness_common import (
    ANALYSIS_SEED,
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    GREEDY_CACHE_FILES,
    N_BOOTSTRAP,
    CORRECTNESS_FIELDS,
    REQUIRED_RECORD_FIELDS,
    SEEDS,
    _load_metric_module,
    load_dataset,
    majority_vote,
    read_jsonl,
)

METHODS = ("cot_sft", "glr_k10")
METHOD_LABELS = {"cot_sft": "CoT-SFT", "glr_k10": "GLR-10"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--analysis-seed", type=int, default=ANALYSIS_SEED)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def percentile_ci(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def load_shards(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    if not records:
        raise ValueError(f"No records in shards: {paths}")
    frame = pd.DataFrame.from_records(records)
    missing = sorted(set(REQUIRED_RECORD_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required record fields: {missing}")
    return frame


def require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing inference shards. Run analysis/robustness/run_robustness.sh first:\n"
            + "\n".join(missing)
        )


def apply_greedy_correctness(full: pd.DataFrame, prefix: pd.DataFrame) -> None:
    """Apply the correctness labels and the length-conditioned budget accuracy.

    A token prefix is not re-scored.  The complete
    cached generation with the benchmark/method-specific correctness field and
    credits it at budget B only when the complete post-prompt generation has
    token_count <= B.
    """
    full["metric_correct"] = full["correct"].astype(bool)
    prefix["metric_correct"] = prefix["correct"].astype(bool)
    full["full_correct"] = False
    prefix["full_correct"] = False
    prefix["completed_within_budget"] = False
    full["correctness_mode"] = ""
    prefix["correctness_mode"] = ""

    for dataset in DATASETS:
        dataset_rows = load_dataset(dataset)
        expected_ids = [str(row["id"]) for row in dataset_rows]
        for method in METHODS:
            cache_path = GREEDY_CACHE_FILES[method][dataset]
            cached = list(read_jsonl(cache_path))
            if len(cached) != len(expected_ids):
                raise ValueError(
                    f"{dataset}/{method}: expected {len(expected_ids)} cached "
                    f"rows, got {len(cached)}"
                )
            field = CORRECTNESS_FIELDS[dataset][method]
            if any(field not in row for row in cached):
                raise KeyError(
                    f"{dataset}/{method}: correctness field {field!r} "
                    f"is missing from {cache_path}"
                )
            labels = {
                example_id: bool(row[field])
                for example_id, row in zip(expected_ids, cached)
            }

            full_mask = (full["dataset"] == dataset) & (full["method"] == method)
            prefix_mask = (prefix["dataset"] == dataset) & (prefix["method"] == method)
            full_labels = full.loc[full_mask, "example_id"].astype(str).map(labels)
            if full_labels.isna().any():
                raise ValueError(f"{dataset}/{method}: unmatched full example IDs")
            prefix_labels = (
                prefix.loc[prefix_mask, "example_id"].astype(str).map(labels)
            )
            if prefix_labels.isna().any():
                raise ValueError(f"{dataset}/{method}: unmatched prefix example IDs")

            full.loc[full_mask, "full_correct"] = full_labels.to_numpy(dtype=bool)
            full.loc[full_mask, "correct"] = full_labels.to_numpy(dtype=bool)
            full.loc[full_mask, "correctness_mode"] = field

            full_steps = dict(
                zip(
                    full.loc[full_mask, "example_id"].astype(str),
                    full.loc[full_mask, "total_steps"].astype(int),
                )
            )
            completion_mask = (
                prefix.loc[prefix_mask, "example_id"]
                .astype(str)
                .map(full_steps)
                .le(DATASETS[dataset].constrained_budget)
            )
            budget_correct = prefix_labels.astype(bool) & completion_mask
            prefix.loc[prefix_mask, "full_correct"] = prefix_labels.to_numpy(
                dtype=bool
            )
            prefix.loc[prefix_mask, "completed_within_budget"] = (
                completion_mask.to_numpy(dtype=bool)
            )
            prefix.loc[prefix_mask, "correct"] = budget_correct.to_numpy(dtype=bool)
            prefix.loc[prefix_mask, "correctness_mode"] = field


def load_scored_sampled_shards(paths: list[Path]) -> pd.DataFrame:
    """Score sampled trajectories with the benchmark evaluators and the budget rule."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        records = list(read_jsonl(path))
        if not records:
            raise ValueError(f"No records in sampled shard: {path}")
        frame = pd.DataFrame.from_records(records)
        missing = sorted(set(REQUIRED_RECORD_FIELDS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing required record fields: {missing}")
        datasets = frame["dataset"].unique().tolist()
        methods = frame["method"].unique().tolist()
        if len(datasets) != 1 or len(methods) != 1:
            raise ValueError(f"{path}: expected exactly one dataset and method")
        dataset = str(datasets[0])
        method = str(methods[0])
        field = CORRECTNESS_FIELDS[dataset][method]

        base_correct = frame["correct"].astype(bool).to_numpy()
        predicted_answers = frame["predicted_answer"].tolist()
        valid_answers = frame["valid_answer"].astype(bool).to_numpy()

        if dataset == "math500":
            metric = _load_metric_module("minerva_math500_score")
            results, _ = metric.score_file(
                path,
                pred_key="generated_text",
                gold_key="gold_answer",
                compute_math_verify=True,
            )
            if len(results) != len(frame):
                raise ValueError(f"{path}: MATH500 scorer row-count mismatch")
            base_correct = np.asarray(
                [bool(result.math_verify) for result in results], dtype=bool
            )
            predicted_answers = [
                (
                    result.pred_norm
                    if result.pred_final_raw != "[invalidanswer]"
                    and result.pred_norm != "[invalidanswer]"
                    else None
                )
                for result in results
            ]
            valid_answers = np.asarray(
                [answer is not None for answer in predicted_answers], dtype=bool
            )
        elif dataset == "svamp" and method == "glr_k10":
            metric = _load_metric_module("amc23_score")
            results, _ = metric.score_file(
                path,
                pred_key="generated_text",
                gold_key="gold_answer",
                extraction_mode="flexible-extract",
                compute_math_verify=True,
            )
            if len(results) != len(frame):
                raise ValueError(f"{path}: SVAMP scorer row-count mismatch")
            base_correct = np.asarray(
                [bool(result.math_verify) for result in results], dtype=bool
            )
            predicted_answers = [
                (
                    result.pred_norm
                    if result.pred_final_raw not in ("[invalid]", "[invalidanswer]")
                    and result.pred_norm not in ("[invalid]", "[invalidanswer]")
                    else None
                )
                for result in results
            ]
            valid_answers = np.asarray(
                [answer is not None for answer in predicted_answers], dtype=bool
            )

        # A direct decode that reaches max_new_tokens without EOS is the prefix
        # of a generation whose complete length exceeds B.  It is therefore
        # unsuccessful under the correct & (token_count <= B) rule.
        completed = ~frame["hit_cap"].astype(bool).to_numpy()
        frame["metric_correct"] = base_correct
        frame["completed_within_budget"] = completed
        frame["correct"] = base_correct & completed
        frame["correctness_mode"] = field
        frame["predicted_answer"] = predicted_answers
        frame["valid_answer"] = valid_answers
        frames.append(frame)

    sampled = pd.concat(frames, ignore_index=True)
    missing = sorted(set(REQUIRED_RECORD_FIELDS) - set(sampled.columns))
    if missing:
        raise ValueError(f"Missing required sampled record fields: {missing}")
    return sampled


def consolidate(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = output_dir / "raw"
    full_paths = [
        raw / f"greedy_full_{dataset}_{method}.jsonl"
        for dataset in DATASETS
        for method in METHODS
    ]
    prefix_paths = [
        raw / f"greedy_prefix_{dataset}_{method}.jsonl"
        for dataset in DATASETS
        for method in METHODS
    ]
    sampled_paths = [
        raw / f"sampled_{dataset}_{method}_sample{sample_id}.jsonl"
        for dataset in DATASETS
        for method in METHODS
        for sample_id in range(3)
    ]
    require_paths(full_paths + prefix_paths + sampled_paths)
    full = load_shards(full_paths)
    prefix = load_shards(prefix_paths)
    apply_greedy_correctness(full, prefix)
    sampled = load_scored_sampled_shards(sampled_paths)

    key = [
        "dataset",
        "example_id",
        "method",
        "decoding",
        "sample_id",
        "total_step_budget",
    ]
    for label, frame in (
        ("greedy_full", full),
        ("greedy_constrained_prefixes", prefix),
        ("sampled_constrained", sampled),
    ):
        duplicates = frame.duplicated(key, keep=False)
        if duplicates.any():
            raise ValueError(f"{label}: duplicate composite keys")
        frame.sort_values(key, inplace=True, kind="stable")
        frame.reset_index(drop=True, inplace=True)
        frame.to_parquet(output_dir / f"{label}.parquet", index=False)
    return full, prefix, sampled


def align_methods(
    frame: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = frame[frame["dataset"] == dataset]
    cot = subset[subset["method"] == "cot_sft"].sort_values("example_id")
    glr = subset[subset["method"] == "glr_k10"].sort_values("example_id")
    if len(cot) != DATASETS[dataset].expected_count or len(glr) != len(cot):
        raise ValueError(
            f"{dataset}: expected {DATASETS[dataset].expected_count} rows/method, "
            f"got CoT={len(cot)}, GLR={len(glr)}"
        )
    if cot["example_id"].tolist() != glr["example_id"].tolist():
        raise ValueError(f"{dataset}: method example IDs do not align")
    if cot["gold_answer"].tolist() != glr["gold_answer"].tolist():
        raise ValueError(f"{dataset}: method gold answers do not align")
    return cot.reset_index(drop=True), glr.reset_index(drop=True)


def exact_mcnemar(cot: np.ndarray, glr: np.ndarray) -> dict[str, Any]:
    b = int(np.sum(cot & ~glr))
    c = int(np.sum(~cot & glr))
    discordant = b + c
    raw_p = (
        1.0
        if discordant == 0
        else float(
            binomtest(min(b, c), discordant, 0.5, alternative="two-sided").pvalue
        )
    )
    return {"b": b, "c": c, "discordant": discordant, "p_raw": raw_p}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * p_values[int(original_index)]
        running = max(running, candidate)
        adjusted[int(original_index)] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_accuracy(
    cot: np.ndarray, glr: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(cot)
    cot_boot = np.empty(n_boot)
    glr_boot = np.empty(n_boot)
    delta_boot = np.empty(n_boot)
    for replicate in range(n_boot):
        indices = rng.integers(0, n, n)
        cot_boot[replicate] = cot[indices].mean()
        glr_boot[replicate] = glr[indices].mean()
        delta_boot[replicate] = 100.0 * (glr_boot[replicate] - cot_boot[replicate])
    return {
        "n": n,
        "cot_sft": {
            "estimate_percent": 100.0 * float(cot.mean()),
            "ci95_percent": (np.array(percentile_ci(cot_boot)) * 100.0).tolist(),
        },
        "glr_k10": {
            "estimate_percent": 100.0 * float(glr.mean()),
            "ci95_percent": (np.array(percentile_ci(glr_boot)) * 100.0).tolist(),
        },
        "delta_glr_minus_cot_pp": {
            "estimate": 100.0 * float(glr.mean() - cot.mean()),
            "ci95": percentile_ci(delta_boot),
        },
    }


def _bootstrap_lengths(
    cot: np.ndarray, glr: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(cot)
    fields = {
        "cot_mean": np.empty(n_boot),
        "glr_mean": np.empty(n_boot),
        "cot_median": np.empty(n_boot),
        "glr_median": np.empty(n_boot),
        "mean_difference_glr_minus_cot": np.empty(n_boot),
        "median_difference_glr_minus_cot": np.empty(n_boot),
        "median_ratio_glr_over_cot": np.empty(n_boot),
        "median_reduction_percent": np.empty(n_boot),
    }
    for replicate in range(n_boot):
        indices = rng.integers(0, n, n)
        c = cot[indices]
        g = glr[indices]
        cmean, gmean = float(np.mean(c)), float(np.mean(g))
        cmed, gmed = float(np.median(c)), float(np.median(g))
        ratio = gmed / cmed if cmed else math.nan
        fields["cot_mean"][replicate] = cmean
        fields["glr_mean"][replicate] = gmean
        fields["cot_median"][replicate] = cmed
        fields["glr_median"][replicate] = gmed
        fields["mean_difference_glr_minus_cot"][replicate] = gmean - cmean
        fields["median_difference_glr_minus_cot"][replicate] = gmed - cmed
        fields["median_ratio_glr_over_cot"][replicate] = ratio
        fields["median_reduction_percent"][replicate] = 100.0 * (1.0 - ratio)
    cmed = float(np.median(cot))
    gmed = float(np.median(glr))
    return {
        "n": n,
        "cot_sft": {
            "mean": float(np.mean(cot)),
            "mean_ci95": percentile_ci(fields["cot_mean"]),
            "median": cmed,
            "median_ci95": percentile_ci(fields["cot_median"]),
            "iqr": [
                float(np.percentile(cot, 25)),
                float(np.percentile(cot, 75)),
            ],
        },
        "glr_k10": {
            "mean": float(np.mean(glr)),
            "mean_ci95": percentile_ci(fields["glr_mean"]),
            "median": gmed,
            "median_ci95": percentile_ci(fields["glr_median"]),
            "iqr": [
                float(np.percentile(glr, 25)),
                float(np.percentile(glr, 75)),
            ],
        },
        "paired": {
            "mean_difference_glr_minus_cot": float(np.mean(glr) - np.mean(cot)),
            "mean_difference_ci95": percentile_ci(
                fields["mean_difference_glr_minus_cot"]
            ),
            "median_difference_glr_minus_cot": gmed - cmed,
            "median_difference_ci95": percentile_ci(
                fields["median_difference_glr_minus_cot"]
            ),
            "median_ratio_glr_over_cot": gmed / cmed if cmed else math.nan,
            "median_ratio_ci95": percentile_ci(fields["median_ratio_glr_over_cot"]),
            "median_reduction_percent": (
                100.0 * (1.0 - gmed / cmed) if cmed else math.nan
            ),
            "median_reduction_ci95": percentile_ci(fields["median_reduction_percent"]),
        },
    }


def greedy_statistics(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    output: dict[str, Any] = {}
    accuracy_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    mcnemar_results: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        cot_prefix, glr_prefix = align_methods(prefix, dataset)
        cot_correct = cot_prefix["correct"].to_numpy(dtype=bool)
        glr_correct = glr_prefix["correct"].to_numpy(dtype=bool)
        accuracy = bootstrap_accuracy(
            cot_correct, glr_correct, n_boot, seed + 100 * dataset_index
        )
        mcnemar = exact_mcnemar(cot_correct, glr_correct)
        mcnemar_results.append(mcnemar)

        cot_full, glr_full = align_methods(full, dataset)
        cot_length = cot_full["total_steps"].to_numpy(dtype=float)
        glr_length = glr_full["total_steps"].to_numpy(dtype=float)
        all_lengths = _bootstrap_lengths(
            cot_length, glr_length, n_boot, seed + 1000 + dataset_index
        )
        joint_mask = cot_full["correct"].to_numpy(dtype=bool) & glr_full[
            "correct"
        ].to_numpy(dtype=bool)
        if not joint_mask.any():
            raise ValueError(f"{dataset}: no jointly correct examples")
        joint_lengths = _bootstrap_lengths(
            cot_length[joint_mask],
            glr_length[joint_mask],
            n_boot,
            seed + 2000 + dataset_index,
        )
        joint_lengths["unstable_fewer_than_30"] = bool(joint_mask.sum() < 30)
        correct_only: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for method, method_frame in (
            ("cot_sft", cot_full),
            ("glr_k10", glr_full),
        ):
            own_correct = method_frame["correct"].to_numpy(dtype=bool)
            correct_only[method] = {
                "n": int(own_correct.sum()),
                "median_total_steps": (
                    float(
                        np.median(
                            method_frame.loc[own_correct, "total_steps"].to_numpy(
                                dtype=float
                            )
                        )
                    )
                    if own_correct.any()
                    else None
                ),
            }
            diagnostics[method] = {
                "cap_hit_percent": 100.0 * float(method_frame["hit_cap"].mean()),
                "valid_answer_percent": 100.0
                * float(method_frame["valid_answer"].mean()),
                "answer_extraction_failure_percent": 100.0
                * float(1.0 - method_frame["valid_answer"].mean()),
                "repetition_percent": 100.0
                * float(method_frame["repetition_flag"].mean()),
            }

        output[dataset] = {
            "budget": DATASETS[dataset].constrained_budget,
            "full_cap": DATASETS[dataset].full_cap,
            "accuracy": accuracy,
            "mcnemar": mcnemar,
            "length_all_examples": all_lengths,
            "length_jointly_correct": joint_lengths,
            "length_method_correct_only": correct_only,
            "full_cap_diagnostics": diagnostics,
        }

    adjusted = holm_adjust([result["p_raw"] for result in mcnemar_results])
    for dataset, adjusted_p in zip(DATASETS, adjusted):
        output[dataset]["mcnemar"]["p_holm"] = adjusted_p
        accuracy = output[dataset]["accuracy"]
        mc = output[dataset]["mcnemar"]
        accuracy_rows.append(
            {
                "dataset": dataset,
                "budget": output[dataset]["budget"],
                "cot_sft_accuracy_percent": accuracy["cot_sft"]["estimate_percent"],
                "cot_sft_ci_low": accuracy["cot_sft"]["ci95_percent"][0],
                "cot_sft_ci_high": accuracy["cot_sft"]["ci95_percent"][1],
                "glr_k10_accuracy_percent": accuracy["glr_k10"]["estimate_percent"],
                "glr_k10_ci_low": accuracy["glr_k10"]["ci95_percent"][0],
                "glr_k10_ci_high": accuracy["glr_k10"]["ci95_percent"][1],
                "delta_pp": accuracy["delta_glr_minus_cot_pp"]["estimate"],
                "delta_ci_low": accuracy["delta_glr_minus_cot_pp"]["ci95"][0],
                "delta_ci_high": accuracy["delta_glr_minus_cot_pp"]["ci95"][1],
                "mcnemar_b": mc["b"],
                "mcnemar_c": mc["c"],
                "mcnemar_p_raw": mc["p_raw"],
                "mcnemar_p_holm": mc["p_holm"],
            }
        )
        for method in METHODS:
            all_stats = output[dataset]["length_all_examples"][method]
            length_rows.append(
                {
                    "dataset": dataset,
                    "full_cap": output[dataset]["full_cap"],
                    "method": method,
                    "all_mean": all_stats["mean"],
                    "all_mean_ci_low": all_stats["mean_ci95"][0],
                    "all_mean_ci_high": all_stats["mean_ci95"][1],
                    "all_median": all_stats["median"],
                    "all_median_ci_low": all_stats["median_ci95"][0],
                    "all_median_ci_high": all_stats["median_ci95"][1],
                    "all_iqr_low": all_stats["iqr"][0],
                    "all_iqr_high": all_stats["iqr"][1],
                    "correct_only_median": output[dataset][
                        "length_method_correct_only"
                    ][method]["median_total_steps"],
                    "joint_correct_n": output[dataset]["length_jointly_correct"]["n"],
                    "joint_correct_median": output[dataset]["length_jointly_correct"][
                        method
                    ]["median"],
                    "cap_hit_percent": output[dataset]["full_cap_diagnostics"][method][
                        "cap_hit_percent"
                    ],
                    "valid_answer_percent": output[dataset]["full_cap_diagnostics"][
                        method
                    ]["valid_answer_percent"],
                    "all_median_reduction_percent": output[dataset][
                        "length_all_examples"
                    ]["paired"]["median_reduction_percent"],
                    "all_median_reduction_ci_low": output[dataset][
                        "length_all_examples"
                    ]["paired"]["median_reduction_ci95"][0],
                    "all_median_reduction_ci_high": output[dataset][
                        "length_all_examples"
                    ]["paired"]["median_reduction_ci95"][1],
                    "joint_median_reduction_percent": output[dataset][
                        "length_jointly_correct"
                    ]["paired"]["median_reduction_percent"],
                    "joint_median_reduction_ci_low": output[dataset][
                        "length_jointly_correct"
                    ]["paired"]["median_reduction_ci95"][0],
                    "joint_median_reduction_ci_high": output[dataset][
                        "length_jointly_correct"
                    ]["paired"]["median_reduction_ci95"][1],
                }
            )
    return output, accuracy_rows, length_rows


def sampled_problem_arrays(
    sampled: pd.DataFrame, dataset: str, method: str
) -> tuple[list[str], dict[str, np.ndarray]]:
    subset = sampled[
        (sampled["dataset"] == dataset) & (sampled["method"] == method)
    ].copy()
    expected = DATASETS[dataset].expected_count * 3
    if len(subset) != expected:
        raise ValueError(
            f"{dataset}/{method}: expected {expected} sampled rows, got {len(subset)}"
        )
    ids = sorted(subset["example_id"].unique().tolist())
    fields: dict[str, list[float]] = {
        "sampled_pass1": [],
        "majority3": [],
        "pass3": [],
        "consensus_rate": [],
    }
    for example_id in ids:
        group = subset[subset["example_id"] == example_id].sort_values("sample_id")
        if group["sample_id"].tolist() != [0, 1, 2]:
            raise ValueError(
                f"{dataset}/{method}/{example_id}: sample IDs are not [0,1,2]"
            )
        if group["seed"].astype(int).tolist() != list(SEEDS):
            raise ValueError(
                f"{dataset}/{method}/{example_id}: unexpected replicate seeds"
            )
        correct = group["correct"].astype(bool).tolist()
        answers = [
            (value if isinstance(value, str) and not bool(hit_cap) else None)
            for value, hit_cap in zip(
                group["predicted_answer"].tolist(),
                group["hit_cap"].tolist(),
            )
        ]
        consensus, voted_correct = majority_vote(dataset, answers, correct)
        fields["sampled_pass1"].append(float(np.mean(correct)))
        fields["majority3"].append(float(voted_correct))
        fields["pass3"].append(float(any(correct)))
        fields["consensus_rate"].append(float(consensus))
    return ids, {key: np.asarray(value) for key, value in fields.items()}


def bootstrap_cluster_metrics(
    cot: dict[str, np.ndarray],
    glr: dict[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(cot["sampled_pass1"])
    distributions = {
        metric: {
            "cot_sft": np.empty(n_boot),
            "glr_k10": np.empty(n_boot),
            "delta_glr_minus_cot_pp": np.empty(n_boot),
        }
        for metric in cot
    }
    for replicate in range(n_boot):
        indices = rng.integers(0, n, n)
        for metric in cot:
            c = float(cot[metric][indices].mean())
            g = float(glr[metric][indices].mean())
            distributions[metric]["cot_sft"][replicate] = c
            distributions[metric]["glr_k10"][replicate] = g
            distributions[metric]["delta_glr_minus_cot_pp"][replicate] = 100.0 * (g - c)
    output: dict[str, Any] = {}
    for metric in cot:
        output[metric] = {
            "cot_sft": {
                "estimate_percent": 100.0 * float(cot[metric].mean()),
                "ci95_percent": (
                    np.array(percentile_ci(distributions[metric]["cot_sft"])) * 100.0
                ).tolist(),
            },
            "glr_k10": {
                "estimate_percent": 100.0 * float(glr[metric].mean()),
                "ci95_percent": (
                    np.array(percentile_ci(distributions[metric]["glr_k10"])) * 100.0
                ).tolist(),
            },
            "delta_glr_minus_cot_pp": {
                "estimate": 100.0 * float(glr[metric].mean() - cot[metric].mean()),
                "ci95": percentile_ci(distributions[metric]["delta_glr_minus_cot_pp"]),
            },
        }
    return output


def sampled_statistics(
    sampled: pd.DataFrame, n_boot: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        cot_ids, cot = sampled_problem_arrays(sampled, dataset, "cot_sft")
        glr_ids, glr = sampled_problem_arrays(sampled, dataset, "glr_k10")
        if cot_ids != glr_ids:
            raise ValueError(f"{dataset}: sampled method IDs do not align")
        metrics = bootstrap_cluster_metrics(
            cot, glr, n_boot, seed + 3000 + dataset_index
        )
        majority_mcnemar = exact_mcnemar(
            cot["majority3"].astype(bool), glr["majority3"].astype(bool)
        )
        diagnostics: dict[str, Any] = {}
        for method in METHODS:
            frame = sampled[
                (sampled["dataset"] == dataset) & (sampled["method"] == method)
            ]
            diagnostics[method] = {
                "cap_hit_percent": 100.0 * float(frame["hit_cap"].mean()),
                "repetition_percent": 100.0 * float(frame["repetition_flag"].mean()),
                "valid_answer_percent": 100.0 * float(frame["valid_answer"].mean()),
                "answer_extraction_failure_percent": 100.0
                * float(1.0 - frame["valid_answer"].mean()),
            }
        output[dataset] = {
            "budget": DATASETS[dataset].constrained_budget,
            "n_problems": len(cot_ids),
            "n_trajectories_per_method": len(cot_ids) * 3,
            "metrics": metrics,
            "majority3_mcnemar": majority_mcnemar,
            "diagnostics": diagnostics,
        }
        row: dict[str, Any] = {
            "dataset": dataset,
            "budget": DATASETS[dataset].constrained_budget,
            "n_problems": len(cot_ids),
            "majority_mcnemar_b": majority_mcnemar["b"],
            "majority_mcnemar_c": majority_mcnemar["c"],
            "majority_mcnemar_p_raw": majority_mcnemar["p_raw"],
        }
        for metric in metrics:
            for method in METHODS:
                row[f"{metric}_{method}_percent"] = metrics[metric][method][
                    "estimate_percent"
                ]
                row[f"{metric}_{method}_ci_low"] = metrics[metric][method][
                    "ci95_percent"
                ][0]
                row[f"{metric}_{method}_ci_high"] = metrics[metric][method][
                    "ci95_percent"
                ][1]
            row[f"{metric}_delta_pp"] = metrics[metric]["delta_glr_minus_cot_pp"][
                "estimate"
            ]
            row[f"{metric}_delta_ci_low"] = metrics[metric]["delta_glr_minus_cot_pp"][
                "ci95"
            ][0]
            row[f"{metric}_delta_ci_high"] = metrics[metric]["delta_glr_minus_cot_pp"][
                "ci95"
            ][1]
        for method in METHODS:
            for diagnostic, value in diagnostics[method].items():
                row[f"{diagnostic}_{method}"] = value
        rows.append(row)
    return output, rows


def fmt_ci(estimate: float, ci: list[float], decimals: int = 1) -> str:
    return f"{estimate:.{decimals}f} " f"[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def render_greedy_table(stats: dict[str, Any]) -> str:
    lines = [
        "**Paired uncertainty for greedy pass@1 at the representative constrained budgets.** "
        "A trajectory is credited only when the complete generation is correct "
        "under the benchmark-specific evaluator and its post-prompt length is at most the budget. "
        "Confidence intervals are obtained by 10,000 paired bootstrap resamples over "
        "evaluation problems. \\(p\\)-values use exact McNemar tests with Holm correction "
        "across the three benchmarks. All results use the fixed Qwen3-1.7B checkpoints "
        "and GLR \\(K=10\\).",
        "",
        "| Dataset | Budget | CoT-SFT acc. % [95% CI] | GLR-10 acc. % [95% CI] | "
        "\\(\\Delta\\) acc. pp [95% paired CI] | McNemar \\(p_{\\mathrm{Holm}}\\) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        item = stats[dataset]
        accuracy = item["accuracy"]
        delta = accuracy["delta_glr_minus_cot_pp"]
        delta_text = fmt_ci(delta["estimate"], delta["ci95"])
        if delta["ci95"][0] > 0 or delta["ci95"][1] < 0:
            delta_text = f"**{delta_text}**"
        lines.append(
            f"| {dataset.upper()} | {item['budget']} | "
            f"{fmt_ci(accuracy['cot_sft']['estimate_percent'], accuracy['cot_sft']['ci95_percent'])} | "
            f"{fmt_ci(accuracy['glr_k10']['estimate_percent'], accuracy['glr_k10']['ci95_percent'])} | "
            f"{delta_text} | {fmt_p(item['mcnemar']['p_holm'])} |"
        )
    return "\n".join(lines) + "\n"


def render_sampling_table(stats: dict[str, Any]) -> str:
    lines = [
        "**Robustness to temperature sampling.** Each problem is decoded three times "
        "with temperature \\(0.6\\) and top-\\(p=0.95\\). Confidence intervals use a "
        "problem-level cluster bootstrap that keeps the three trajectories for each "
        "resampled problem together. A trajectory is credited only "
        "when it completes within the stated budget and is correct under the same "
        "benchmark-specific evaluator. Majority@3 requires a strict two-of-three "
        "completed-answer consensus; Pass@3 is an oracle metric. Total inference cost is three "
        "trajectories per method and should not be interpreted as single-trajectory efficiency.",
        "",
        "| Dataset | Budget | Sampled pass@1: CoT / GLR | Majority@3: CoT / GLR | "
        "Pass@3: CoT / GLR |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        item = stats[dataset]
        metrics = item["metrics"]

        def pair(metric: str) -> str:
            return (
                f"{fmt_ci(metrics[metric]['cot_sft']['estimate_percent'], metrics[metric]['cot_sft']['ci95_percent'])}"
                " / "
                f"{fmt_ci(metrics[metric]['glr_k10']['estimate_percent'], metrics[metric]['glr_k10']['ci95_percent'])}"
            )

        lines.append(
            f"| {dataset.upper()} | {item['budget']} | {pair('sampled_pass1')} | "
            f"{pair('majority3')} | {pair('pass3')} |"
        )
    return "\n".join(lines) + "\n"


def render_length_table(stats: dict[str, Any]) -> str:
    lines = [
        "**Generation length:** tokenizer length of "
        "the complete final assistant turn after the prompt. Generated wrapper "
        "and EOS tokens are included; prompt tokens are excluded. GLR latent "
        "positions are already present in this sequence and are not added twice.",
        "",
        "| Dataset | Full cap | Method | Median steps, all [95% CI] | "
        "Mean steps, all [95% CI] | Correct-only median | Joint-correct median | "
        "Cap-hit % | Valid-answer % |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        item = stats[dataset]
        for method in METHODS:
            all_stats = item["length_all_examples"][method]
            lines.append(
                f"| {dataset.upper()} | {item['full_cap']} | {METHOD_LABELS[method]} | "
                f"{fmt_ci(all_stats['median'], all_stats['median_ci95'])} | "
                f"{fmt_ci(all_stats['mean'], all_stats['mean_ci95'])} | "
                f"{item['length_method_correct_only'][method]['median_total_steps']:.1f} | "
                f"{item['length_jointly_correct'][method]['median']:.1f} | "
                f"{item['full_cap_diagnostics'][method]['cap_hit_percent']:.1f} | "
                f"{item['full_cap_diagnostics'][method]['valid_answer_percent']:.1f} |"
            )
        all_pair = item["length_all_examples"]["paired"]
        joint = item["length_jointly_correct"]
        lines.append(
            f"| {dataset.upper()} summary | {item['full_cap']} | paired | "
            f"GLR median reduction: {fmt_ci(all_pair['median_reduction_percent'], all_pair['median_reduction_ci95'])}% | "
            f"Jointly correct \\(n={joint['n']}\\); reduction: "
            f"{fmt_ci(joint['paired']['median_reduction_percent'], joint['paired']['median_reduction_ci95'])}% | "
            "— | — | — | — |"
        )
    return "\n".join(lines) + "\n"


def latex_escape_dataset(dataset: str) -> str:
    return {"gsm8k": "GSM8K", "math500": "MATH500", "svamp": "SVAMP"}[dataset]


def render_main_latex(greedy: dict[str, Any], sampled: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & $B$ & Greedy $\Delta$ acc. & Sampled $\Delta$ pass@1 & "
        r"$\Delta$ Majority@3 & Median-step reduction \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        g_delta = greedy[dataset]["accuracy"]["delta_glr_minus_cot_pp"]
        s_delta = sampled[dataset]["metrics"]["sampled_pass1"]["delta_glr_minus_cot_pp"]
        m_delta = sampled[dataset]["metrics"]["majority3"]["delta_glr_minus_cot_pp"]
        reduction = greedy[dataset]["length_all_examples"]["paired"]
        lines.append(
            rf"{latex_escape_dataset(dataset)} & {greedy[dataset]['budget']} & "
            rf"${g_delta['estimate']:.1f}\,\substack{{[{g_delta['ci95'][0]:.1f},\,{g_delta['ci95'][1]:.1f}]}}$ & "
            rf"${s_delta['estimate']:.1f}\,\substack{{[{s_delta['ci95'][0]:.1f},\,{s_delta['ci95'][1]:.1f}]}}$ & "
            rf"${m_delta['estimate']:.1f}\,\substack{{[{m_delta['ci95'][0]:.1f},\,{m_delta['ci95'][1]:.1f}]}}$ & "
            rf"${reduction['median_reduction_percent']:.1f}\%\,"
            rf"\substack{{[{reduction['median_reduction_ci95'][0]:.1f},\,"
            rf"{reduction['median_reduction_ci95'][1]:.1f}]}}$ \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{All deltas are GLR-10 minus CoT-SFT in percentage points; "
            r"step reduction uses all-example full-cap greedy generations. Values "
            r"in brackets are paired 95\% bootstrap intervals. Intervals quantify "
            r"evaluation and inference-sampling uncertainty for fixed checkpoints, "
            r"not training-seed variation.}",
            r"\label{tab:glr_robustness}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_appendix_greedy_latex(stats: dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Method & Median all & Mean all & Correct-only median & "
        r"Joint-correct median & Cap-hit \% & Valid \% \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            all_stats = stats[dataset]["length_all_examples"][method]
            lines.append(
                rf"{latex_escape_dataset(dataset)} & {METHOD_LABELS[method]} & "
                rf"{all_stats['median']:.1f} [{all_stats['median_ci95'][0]:.1f}, {all_stats['median_ci95'][1]:.1f}] & "
                rf"{all_stats['mean']:.1f} [{all_stats['mean_ci95'][0]:.1f}, {all_stats['mean_ci95'][1]:.1f}] & "
                rf"{stats[dataset]['length_method_correct_only'][method]['median_total_steps']:.1f} & "
                rf"{stats[dataset]['length_jointly_correct'][method]['median']:.1f} & "
                rf"{stats[dataset]['full_cap_diagnostics'][method]['cap_hit_percent']:.1f} & "
                rf"{stats[dataset]['full_cap_diagnostics'][method]['valid_answer_percent']:.1f} \\"
            )
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Full-cap greedy generation length using the "
            r"post-prompt assistant-turn tokenization. Generated wrapper and EOS "
            r"tokens are included and prompt tokens are excluded. GLR latent "
            r"positions are already represented in this sequence and are counted "
            r"once. Confidence intervals use paired problem-level bootstrap resampling.}",
            r"\label{tab:glr_appendix_greedy_length}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_appendix_sampling_latex(stats: dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Sampled pass@1 (CoT/GLR) & Majority@3 (CoT/GLR) & "
        r"Pass@3 (CoT/GLR) & Consensus (CoT/GLR) & Repetition (CoT/GLR) \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        item = stats[dataset]

        def estimate_pair(metric: str) -> str:
            cot = item["metrics"][metric]["cot_sft"]
            glr = item["metrics"][metric]["glr_k10"]
            return (
                f"{cot['estimate_percent']:.1f} "
                f"[{cot['ci95_percent'][0]:.1f},{cot['ci95_percent'][1]:.1f}]/"
                f"{glr['estimate_percent']:.1f} "
                f"[{glr['ci95_percent'][0]:.1f},{glr['ci95_percent'][1]:.1f}]"
            )

        lines.append(
            rf"{latex_escape_dataset(dataset)} & {estimate_pair('sampled_pass1')} & "
            rf"{estimate_pair('majority3')} & {estimate_pair('pass3')} & "
            rf"{estimate_pair('consensus_rate')} & "
            rf"{item['diagnostics']['cot_sft']['repetition_percent']:.1f}/"
            rf"{item['diagnostics']['glr_k10']['repetition_percent']:.1f} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Temperature-sampled robustness at $T=0.6$, top-$p=0.95$ "
            r"with three trajectories per problem. Brackets give 95\% "
            r"problem-cluster bootstrap intervals. "
            r"Pass@3 is oracle-style; Majority@3 also consumes three trajectories.}",
            r"\label{tab:glr_appendix_sampling}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def result_phrase(dataset: str, metric: dict[str, Any]) -> str:
    delta = metric["delta_glr_minus_cot_pp"]
    estimate = delta["estimate"]
    ci = delta["ci95"]
    if ci[0] > 0:
        interpretation = "the improvement remains positive under paired resampling"
    elif ci[1] < 0:
        interpretation = "the decrease remains negative under paired resampling"
    else:
        direction = "positive" if estimate >= 0 else "negative"
        interpretation = (
            f"the estimate is directionally {direction}, but the interval includes "
            "zero, so we avoid claiming a reliable difference on this benchmark"
        )
    return (
        f"{dataset.upper()}: {estimate:.1f} pp "
        f"[{ci[0]:.1f}, {ci[1]:.1f}] ({interpretation})"
    )


def render_summary_paragraph(greedy: dict[str, Any], sampled: dict[str, Any]) -> str:
    deltas = ", ".join(
        f"{dataset.upper()} {fmt_ci(greedy[dataset]['accuracy']['delta_glr_minus_cot_pp']['estimate'], greedy[dataset]['accuracy']['delta_glr_minus_cot_pp']['ci95'])} pp"
        for dataset in DATASETS
    )
    return (
        "We evaluated the fixed Qwen3-1.7B CoT-SFT and GLR checkpoints on the "
        "complete GSM8K, MATH500, and SVAMP evaluation sets, using GLR K=10 "
        f"throughout. At constrained total-step budgets, paired greedy accuracy "
        f"differences (GLR minus CoT-SFT; 10,000 problem-level bootstrap resamples) "
        f"were {deltas}; exact paired McNemar tests with Holm correction are "
        "reported alongside these intervals. We additionally decoded three "
        "trajectories per problem at T=0.6 and top-p=0.95 and used a problem-cluster "
        "bootstrap for sampled pass@1, strict Majority@3, and oracle-style Pass@3. "
        "For greedy and sampled results alike, a trajectory is credited only if "
        "it completes within the stated budget and is correct under the "
        "benchmark-specific evaluator. "
        "Majority@3 and Pass@3 use three times as many trajectories and are not "
        "single-trajectory efficiency measures. Generation-length analyses include "
        "all examples at the full caps and a fixed jointly-correct subset. Steps "
        "are the tokenizer length of the complete post-prompt assistant turn, "
        "including generated wrapper/EOS tokens and the serialized GLR latent "
        "positions exactly once. These analyses quantify finite-test-set "
        "and decoding uncertainty for the fixed checkpoints.\n"
    )


def render_flagged_audit(full: pd.DataFrame, sampled: pd.DataFrame) -> str:
    lines = [
        "# Repetition diagnostic audit",
        "",
        "These flags are automatic and were not hand-edited. At most 20 examples "
        "are shown for each dataset/method/decoding condition.",
        "",
    ]
    combined = pd.concat([full, sampled], ignore_index=True)
    for (dataset, method, decoding), group in combined.groupby(
        ["dataset", "method", "decoding"], sort=True
    ):
        flagged = group[group["repetition_flag"]].head(20)
        lines.extend(
            [
                f"## {dataset} / {method} / {decoding}",
                "",
                f"Flagged rows shown: {len(flagged)}",
                "",
            ]
        )
        for _, row in flagged.iterrows():
            text = str(row["generated_text"]).replace("```", "'''")
            lines.extend(
                [
                    f"### Example {row['example_id']} (sample {row['sample_id']})",
                    "",
                    "```text",
                    text,
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def render_figures(
    output_dir: Path, greedy: dict[str, Any], sampled: dict[str, Any]
) -> None:
    labels = [latex_escape_dataset(dataset) for dataset in DATASETS]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for offset, source, key, label, color in (
        (-0.14, greedy, "accuracy", "Greedy pass@1", "#3366aa"),
        (0.14, sampled, "sampled_pass1", "Sampled pass@1", "#cc6633"),
    ):
        estimates, lower, upper = [], [], []
        for dataset in DATASETS:
            if source is greedy:
                delta = source[dataset][key]["delta_glr_minus_cot_pp"]
            else:
                delta = source[dataset]["metrics"][key]["delta_glr_minus_cot_pp"]
            estimates.append(delta["estimate"])
            lower.append(delta["estimate"] - delta["ci95"][0])
            upper.append(delta["ci95"][1] - delta["estimate"])
        axis.errorbar(
            positions + offset,
            estimates,
            yerr=np.array([lower, upper]),
            fmt="o",
            capsize=4,
            label=label,
            color=color,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("GLR-10 minus CoT-SFT (percentage points)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "figure_accuracy_robustness.pdf")
    figure.savefig(output_dir / "figure_accuracy_robustness.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    estimates, lower, upper = [], [], []
    for dataset in DATASETS:
        paired = greedy[dataset]["length_all_examples"]["paired"]
        estimates.append(paired["median_reduction_percent"])
        lower.append(
            paired["median_reduction_percent"] - paired["median_reduction_ci95"][0]
        )
        upper.append(
            paired["median_reduction_ci95"][1] - paired["median_reduction_percent"]
        )
    axis.errorbar(
        positions,
        estimates,
        yerr=np.array([lower, upper]),
        fmt="o",
        capsize=4,
        color="#2b8c62",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("GLR median-step reduction (%)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "figure_length_reduction.pdf")
    figure.savefig(output_dir / "figure_length_reduction.png", dpi=200)
    plt.close(figure)


def write_reports(
    output_dir: Path,
    greedy: dict[str, Any],
    sampled: dict[str, Any],
    full: pd.DataFrame,
) -> None:
    (output_dir / "table_greedy.md").write_text(
        render_greedy_table(greedy), encoding="utf-8"
    )
    (output_dir / "table_sampling.md").write_text(
        render_sampling_table(sampled), encoding="utf-8"
    )
    (output_dir / "table_length.md").write_text(
        render_length_table(greedy), encoding="utf-8"
    )
    (output_dir / "table_main.tex").write_text(
        render_main_latex(greedy, sampled), encoding="utf-8"
    )
    (output_dir / "table_appendix_greedy.tex").write_text(
        render_appendix_greedy_latex(greedy), encoding="utf-8"
    )
    (output_dir / "table_appendix_sampling.tex").write_text(
        render_appendix_sampling_latex(sampled), encoding="utf-8"
    )
    (output_dir / "summary_paragraph.md").write_text(
        render_summary_paragraph(greedy, sampled), encoding="utf-8"
    )
    readme = """# Qwen3-1.7B inference robustness results

This directory is generated entirely from saved per-example inference records.
`greedy_full.parquet`, `greedy_constrained_prefixes.parquet`, and
`sampled_constrained.parquet` are the analysis inputs. The CSV and JSON files
contain unrounded statistics; Markdown/LaTeX tables are presentation-rounded.

The sampled trajectories are decoded by `analysis/robustness/run_robustness.sh`;
the analysis scripts never load a model.

Pass@3 is oracle-style. Majority@3 and Pass@3 each consume three trajectories.
Intervals describe finite evaluation-set and decoding uncertainty for fixed
checkpoints.

Total generation steps: tokenize the final
assistant turn after the question with `add_special_tokens=False`, retaining
generated wrapper/EOS tokens and excluding the prompt. GLR latent positions
already occupy positions in that sequence and are not added a second time.

Greedy budgeted accuracy: the cached
full-generation correctness field is ANDed with `token_count <= budget`; a
truncated prefix is not rescored. Sampled trajectories use the same
benchmark/method-specific evaluator and completion-within-budget rule.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full, prefix, sampled = consolidate(output_dir)
    greedy, accuracy_rows, length_rows = greedy_statistics(
        full, prefix, args.bootstrap_replicates, args.analysis_seed
    )
    sampled_stats, sampled_rows = sampled_statistics(
        sampled, args.bootstrap_replicates, args.analysis_seed
    )
    statistics = {
        "metadata": {
            "analysis_seed": args.analysis_seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "model_size": "1.7B",
            "K": 10,
            "total_step_accounting": (
                "tokenizer length of final assistant turn after prompt; generated "
                "wrapper/EOS retained; GLR latent positions already included once"
            ),
            "budgeted_accuracy": (
                "benchmark-specific full-generation correctness "
                "AND complete post-prompt token count <= budget; sampled cap hits "
                "are incomplete and receive no credit"
            ),
            "correctness_fields": CORRECTNESS_FIELDS,
        },
        "greedy": greedy,
        "sampled": sampled_stats,
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(jsonable(statistics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(accuracy_rows).to_csv(
        output_dir / "greedy_accuracy_stats.csv", index=False
    )
    pd.DataFrame(length_rows).to_csv(
        output_dir / "greedy_length_stats.csv", index=False
    )
    pd.DataFrame(sampled_rows).to_csv(
        output_dir / "sampled_robustness_stats.csv", index=False
    )
    (output_dir / "flagged_repetition_examples.md").write_text(
        render_flagged_audit(full, sampled), encoding="utf-8"
    )
    write_reports(output_dir, greedy, sampled_stats, full)
    render_figures(output_dir, greedy, sampled_stats)
    print(output_dir)


if __name__ == "__main__":
    main()
