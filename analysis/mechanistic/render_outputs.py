# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Aggregate cached metrics and render all tables and figures."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from mechanistic_common import (
    BOOTSTRAP_SEED,
    DATASETS,
    DEFAULT_CACHE_ROOT,
    K_STAR,
    METHOD_LABELS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


METRICS: dict[str, tuple[str, Callable[[np.ndarray], float]]] = {
    "accuracy": ("correct", np.mean),
    "control_accuracy": ("control_correct", np.mean),
    "mean_actual_latent_steps": ("actual_latent_steps", np.mean),
    "valid_answer_rate": ("valid_answer", np.mean),
    "truncation_rate": ("truncated", np.mean),
    "median_total_steps_correct": (
        "total_steps_correct",
        np.median,
    ),
    "median_total_steps_control_correct": (
        "total_steps_control_correct",
        np.median,
    ),
    "mean_total_steps_correct": ("total_steps_correct", np.mean),
    "median_text_tokens_correct": ("text_tokens_correct", np.median),
    "immediate_close_rate": ("immediate_close", np.mean),
    "answer_within_16_rate": ("answer_within_16", np.mean),
    "final_progress": ("final_progress", np.mean),
    "resume_progress": ("resume_progress", np.mean),
    "final_cot_embedding_drift": ("final_cot_embedding_drift", np.mean),
    "final_hidden_drift": ("final_hidden_drift", np.mean),
    "future_hit_at_10": ("future_hit_at_10", np.mean),
    "future_recall_at_10": ("future_recall_at_10", np.mean),
    "future_hit_at_10_no_special": ("future_hit_at_10_no_special", np.mean),
    "future_recall_at_10_no_special": ("future_recall_at_10_no_special", np.mean),
    "nearest_vocab_cosine": ("nearest_vocab_cosine", np.mean),
    "frequency_random_future_hit_at_10": (
        "frequency_random_future_hit_at_10",
        np.mean,
    ),
}


def seed_average(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for category in (
        "immediate_close",
        "early_resume",
        "middle_resume",
        "late_resume",
        "unmatched_or_too_short",
    ):
        frame[f"resume_category_{category}"] = (
            frame.resume_category == category
        ).astype(float)
    identity = [
        "example_id",
        "dataset",
        "method",
        "K",
        "noise_sigma",
        "forced_continuation",
    ]
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in set(identity) | {"seed", "K"}
    ]
    result = frame.groupby(identity, dropna=False, as_index=False)[numeric].mean()
    for name in ("total_steps", "text_tokens"):
        result[f"{name}_correct"] = result[name].where(result["correct"] >= 0.5)
        result[f"{name}_control_correct"] = result[name].where(
            result["control_correct"] >= 0.5
        )
    return result


def estimate(values: np.ndarray, statistic: Callable[[np.ndarray], float]) -> float:
    values = values[np.isfinite(values)]
    return float(statistic(values)) if len(values) else math.nan


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    count: int,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    samples = np.empty(count, dtype=np.float64)
    for index in range(count):
        selected = values[rng.integers(0, len(values), len(values))]
        samples[index] = statistic(selected)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def aggregate(
    frame: pd.DataFrame,
    raw: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scopes = [(name, frame[frame.dataset == name]) for name in DATASETS]
    scopes.append(("pooled", frame))
    rows: list[dict[str, Any]] = []
    keys = ["method", "K", "noise_sigma", "forced_continuation"]
    for scope, scoped in scopes:
        for group_key, group in scoped.groupby(keys, dropna=False):
            row: dict[str, Any] = {
                "dataset": scope,
                **dict(zip(keys, group_key)),
                "n_examples": int(group.example_id.nunique()),
                "n_correct": float(group.correct.sum()),
            }
            raw_scope = raw if scope == "pooled" else raw[raw.dataset == scope]
            raw_group = raw_scope[
                (raw_scope.method == group_key[0])
                & (raw_scope.K == group_key[1])
                & (raw_scope.forced_continuation == group_key[3])
            ]
            if pd.isna(group_key[2]):
                raw_group = raw_group[raw_group.noise_sigma.isna()]
            else:
                raw_group = raw_group[np.isclose(raw_group.noise_sigma, group_key[2])]
            row["across_seed_accuracy_sd"] = float(
                raw_group.groupby("seed").correct.mean().std()
            )
            for metric, (column, function) in METRICS.items():
                values = group[column].to_numpy(dtype=float)
                point = estimate(values, function)
                low, high = bootstrap_ci(values, function, rng, bootstrap)
                row[metric] = point
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            drift_values = group.final_cot_embedding_drift.to_numpy(float)
            correct_mask = group.correct.to_numpy(float) >= 0.5
            for label, mask in (("correct", correct_mask), ("incorrect", ~correct_mask)):
                point = estimate(drift_values[mask], np.mean)
                low, high = bootstrap_ci(
                    drift_values[mask], np.mean, rng, bootstrap
                )
                row[f"cot_embedding_drift_{label}"] = point
                row[f"cot_embedding_drift_{label}_ci_low"] = low
                row[f"cot_embedding_drift_{label}_ci_high"] = high
            delta_draws = []
            if correct_mask.any() and (~correct_mask).any():
                for _ in range(bootstrap):
                    selected = rng.integers(0, len(group), len(group))
                    selected_correct = correct_mask[selected]
                    if selected_correct.any() and (~selected_correct).any():
                        sampled = drift_values[selected]
                        delta_draws.append(
                            sampled[~selected_correct].mean()
                            - sampled[selected_correct].mean()
                        )
            row["incorrect_minus_correct_drift"] = (
                row["cot_embedding_drift_incorrect"]
                - row["cot_embedding_drift_correct"]
            )
            if delta_draws:
                row["incorrect_minus_correct_drift_ci_low"], row[
                    "incorrect_minus_correct_drift_ci_high"
                ] = np.quantile(delta_draws, [0.025, 0.975])
            else:
                row["incorrect_minus_correct_drift_ci_low"] = math.nan
                row["incorrect_minus_correct_drift_ci_high"] = math.nan
            for category in (
                "immediate_close",
                "early_resume",
                "middle_resume",
                "late_resume",
                "unmatched_or_too_short",
            ):
                row[f"resume_category_{category}_rate"] = float(
                    group[f"resume_category_{category}"].mean()
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    ordinary = result[~result.forced_continuation.astype(bool)][
        ["dataset", "method", "K", "noise_sigma", "accuracy", "median_total_steps_correct"]
    ].rename(
        columns={
            "accuracy": "ordinary_accuracy",
            "median_total_steps_correct": "ordinary_median_total_steps_correct",
        }
    )
    result = result.merge(
        ordinary,
        on=["dataset", "method", "K", "noise_sigma"],
        how="left",
    )
    result["accuracy_difference_from_ordinary"] = (
        result.accuracy - result.ordinary_accuracy
    )
    result["median_steps_difference_from_ordinary"] = (
        result.median_total_steps_correct
        - result.ordinary_median_total_steps_correct
    )
    return result


def fit_logistic(
    frame: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
    bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    x = frame[continuous + categorical]
    y = frame.correct.astype(int).to_numpy()
    transform = ColumnTransformer(
        [
            ("continuous", make_pipeline(SimpleImputer(), StandardScaler()), continuous),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical,
            ),
        ]
    )

    def one_fit(indices: np.ndarray) -> np.ndarray:
        model = make_pipeline(
            transform,
            LogisticRegression(
                C=np.inf,
                solver="lbfgs",
                max_iter=1000,
                random_state=0,
            ),
        )
        model.fit(x.iloc[indices], y[indices])
        return model[-1].coef_[0, : len(continuous)]

    point = one_fit(np.arange(len(frame)))
    samples: list[np.ndarray] = []
    attempts = 0
    cluster_values = (
        frame.dataset.astype(str) + "::" + frame.example_id.astype(str)
    ).to_numpy()
    unique_clusters = np.unique(cluster_values)
    cluster_indices = {
        cluster: np.flatnonzero(cluster_values == cluster)
        for cluster in unique_clusters
    }
    while len(samples) < bootstrap and attempts < bootstrap * 5:
        attempts += 1
        selected_clusters = unique_clusters[
            rng.integers(0, len(unique_clusters), len(unique_clusters))
        ]
        indices = np.concatenate(
            [cluster_indices[cluster] for cluster in selected_clusters]
        )
        if len(np.unique(y[indices])) < 2:
            continue
        try:
            samples.append(one_fit(indices))
        except Exception:
            continue
    if len(samples) < max(1, bootstrap // 2):
        raise RuntimeError("too few successful regression bootstrap fits")
    draws = np.stack(samples)
    return point, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def regression_results(
    raw: pd.DataFrame, bootstrap: int, seed: int
) -> pd.DataFrame:
    learned = raw[
        (raw.method == "learned")
        & (~raw.forced_continuation.astype(bool))
        & raw.final_cot_embedding_drift.notna()
    ].copy()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    continuous = [
        "final_cot_embedding_drift",
        "final_hidden_drift",
        "final_progress",
        "actual_latent_steps",
    ]
    point, low, high = fit_logistic(
        learned, continuous, ["K", "dataset"], bootstrap, rng
    )
    for index, coefficient in enumerate(continuous):
        rows.append(
            {
                "formula": "correct ~ z(drift_E)+z(drift_H)+z(progress)+z(realized_K)+C(K_max)+C(dataset)",
                "dataset_scope": "pooled",
                "K_scope": "all",
                "coefficient": coefficient,
                "beta": point[index],
                "odds_ratio": math.exp(point[index]),
                "ci_low": low[index],
                "ci_high": high[index],
                "odds_ratio_ci_low": math.exp(low[index]),
                "odds_ratio_ci_high": math.exp(high[index]),
                "n_examples": len(learned),
            }
        )
    for k, group in learned.groupby("K"):
        if group.correct.nunique() < 2:
            continue
        point, low, high = fit_logistic(
            group,
            ["final_cot_embedding_drift", "actual_latent_steps"],
            ["dataset"],
            bootstrap,
            rng,
        )
        for index, coefficient in enumerate(
            ("final_cot_embedding_drift", "actual_latent_steps")
        ):
            rows.append(
                {
                    "formula": "correct ~ z(drift_E)+z(realized_K)+C(dataset)",
                    "dataset_scope": "pooled",
                    "K_scope": str(int(k)),
                    "coefficient": coefficient,
                    "beta": point[index],
                    "odds_ratio": math.exp(point[index]),
                    "ci_low": low[index],
                    "ci_high": high[index],
                    "odds_ratio_ci_low": math.exp(low[index]),
                    "odds_ratio_ci_high": math.exp(high[index]),
                    "n_examples": len(group),
                }
            )
    return pd.DataFrame(rows)


def trajectory_aggregates(
    trajectory: pd.DataFrame, bootstrap: int, seed: int
) -> pd.DataFrame:
    metrics = [
        "progress",
        "cot_embedding_cosine_drift",
        "hidden_cosine_drift",
        "nearest_vocab_cosine",
        "future_hit_at_10",
        "future_recall_at_10",
        "future_hit_at_10_no_special",
        "future_recall_at_10_no_special",
        "answer_hit_at_10_no_special",
        "answer_hit_at_10",
        "frequency_random_future_hit_at_10",
    ]
    per_example = (
        trajectory.groupby(
            [
                "example_id",
                "dataset",
                "method",
                "K",
                "noise_sigma",
                "forced_continuation",
                "correct",
                "normalized_step_bin",
            ],
            dropna=False,
            as_index=False,
        )[metrics]
        .mean()
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    scopes = [(name, per_example[per_example.dataset == name]) for name in DATASETS]
    scopes.append(("pooled", per_example))
    keys = [
        "method",
        "K",
        "noise_sigma",
        "forced_continuation",
        "correct",
        "normalized_step_bin",
    ]
    for scope, scoped in scopes:
        for group_key, group in scoped.groupby(keys, dropna=False):
            row = {
                "dataset": scope,
                **dict(zip(keys, group_key)),
                "n_examples": group.example_id.nunique(),
            }
            for metric in metrics:
                values = group[metric].to_numpy(float)
                row[metric] = estimate(values, np.mean)
                low, high = bootstrap_ci(values, np.mean, rng, bootstrap)
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(
    averaged: pd.DataFrame, bootstrap: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = [
        "control_correct",
        "valid_answer",
        "total_steps",
        "immediate_close",
        "final_progress",
        "resume_progress",
        "final_cot_embedding_drift",
        "future_hit_at_10",
    ]
    rows: list[dict[str, Any]] = []
    for dataset in ("pooled", *DATASETS):
        scoped = averaged if dataset == "pooled" else averaged[averaged.dataset == dataset]
        baseline = scoped[
            (scoped.method == "learned")
            & (scoped.K == K_STAR)
            & (~scoped.forced_continuation.astype(bool))
        ]
        baseline = baseline.set_index(["dataset", "example_id"])
        candidates = scoped[
            (scoped.K == K_STAR)
            & (~scoped.forced_continuation.astype(bool))
            & (scoped.method != "learned")
        ]
        for key, group in candidates.groupby(["method", "noise_sigma"], dropna=False):
            group = group.set_index(["dataset", "example_id"])
            common = baseline.index.intersection(group.index)
            if not len(common):
                continue
            for metric in metrics:
                delta = (
                    group.loc[common, metric].to_numpy(float)
                    - baseline.loc[common, metric].to_numpy(float)
                )
                delta = delta[np.isfinite(delta)]
                if not len(delta):
                    continue
                low, high = bootstrap_ci(delta, np.mean, rng, bootstrap)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": key[0],
                        "noise_sigma": key[1],
                        "reference_method": "learned",
                        "K": K_STAR,
                        "metric": metric,
                        "paired_difference_control_minus_learned": float(delta.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "n_pairs": len(delta),
                    }
                )
    return pd.DataFrame(rows)


def matched_control_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict learned K=10 and controls to identical per-dataset examples."""
    pieces: list[pd.DataFrame] = []
    for dataset in DATASETS:
        scoped = frame[
            (frame.dataset == dataset)
            & (frame.K == K_STAR)
            & (~frame.forced_continuation.astype(bool))
        ]
        methods = sorted(
            method for method in scoped.method.unique() if method != "learned"
        )
        if not methods:
            continue
        id_sets = [
            set(scoped[scoped.method == method].example_id.astype(str))
            for method in methods
        ]
        common = set.intersection(*id_sets)
        selected = scoped[
            scoped.example_id.astype(str).isin(common)
            & scoped.method.isin(["learned", *methods])
        ]
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0]


def pct(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{100 * value:.1f}%"


def num(value: float, digits: int = 3) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def controls_table(frame: pd.DataFrame, dataset: str) -> str:
    selected = frame[
        (frame.dataset == dataset)
        & (frame.K == K_STAR)
        & (~frame.forced_continuation.astype(bool))
    ].copy()
    selected = selected[
        (selected.method != "noisy_transition")
        | np.isclose(selected.noise_sigma.astype(float), 1.0, equal_nan=False)
    ]
    order = [
        "learned",
        "zero_transition",
        "repeated_think",
        "random_head_norm_matched",
        "noisy_transition",
        "shuffled_transition",
        "absolute_prediction",
    ]
    lines = [
        f"### {dataset.upper() if dataset != 'pooled' else 'Pooled'} controls (K*={K_STAR})",
        "",
        "| Method | Exact-equivalence acc. ↑ | Median total steps on correct ↓ | Immediate `</think>` ↓ | Resume progress ↑ | CoT-embedding drift ↓ | FutureHit@10 ↑ | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in order:
        rows = selected[selected.method == method]
        if rows.empty:
            continue
        row = rows.iloc[0]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                METHOD_LABELS[method],
                pct(row.control_accuracy),
                num(row.median_total_steps_control_correct, 0),
                pct(row.immediate_close_rate),
                num(row.resume_progress),
                num(row.final_cot_embedding_drift),
                pct(row.future_hit_at_10_no_special),
                int(row.n_examples),
            )
        )
    return "\n".join(lines)


def drift_table(summary: pd.DataFrame, raw: pd.DataFrame, dataset: str) -> str:
    pooled = raw if dataset == "pooled" else raw[raw.dataset == dataset]
    learned = pooled[
        (pooled.method == "learned") & (~pooled.forced_continuation.astype(bool))
    ]
    agg = summary[
        (summary.dataset == dataset)
        & (summary.method == "learned")
        & (~summary.forced_continuation.astype(bool))
    ]
    lines = [
        f"### {dataset.upper() if dataset != 'pooled' else 'Pooled'} drift and semantics",
        "",
        "| K (max) | Dataset coverage | N | Realized latent steps | Acc. ↑ | Final aligned CoT progress ↑ | Drift correct ↓ | Drift incorrect | Incorrect − correct | FutureHit@10 ↑ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in sorted(learned.K.unique()):
        row = agg[agg.K == k]
        if row.empty:
            continue
        row = row.iloc[0]
        coverage = ",".join(
            sorted(learned[learned.K == k].dataset.unique())
        )
        lines.append(
            f"| {int(k)} | {coverage} | {int(row.n_examples)} | "
            f"{num(row.mean_actual_latent_steps, 1)} | "
            f"{pct(row.accuracy)} | {num(row.final_progress)} | "
            f"{num(row.cot_embedding_drift_correct)} | "
            f"{num(row.cot_embedding_drift_incorrect)} | "
            f"{num(row.incorrect_minus_correct_drift)} | "
            f"{pct(row.future_hit_at_10_no_special)} |"
        )
    return "\n".join(lines)


def forced_table(frame: pd.DataFrame, dataset: str) -> str:
    selected = frame[
        (frame.dataset == dataset)
        & (frame.K == K_STAR)
        & (frame.forced_continuation.astype(bool))
    ].copy()
    selected = selected[
        (selected.method != "noisy_transition")
        | np.isclose(selected.noise_sigma.astype(float), 1.0, equal_nan=False)
    ]
    lines = [
        f"### {dataset.upper() if dataset != 'pooled' else 'Pooled'} forced continuation",
        "",
        "| Method | Acc. ↑ | Valid answer ↑ | Median total steps on correct ↓ | Δ acc. vs ordinary | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in (
        "learned",
        "zero_transition",
        "repeated_think",
        "random_head_norm_matched",
        "noisy_transition",
    ):
        rows = selected[selected.method == method]
        if rows.empty:
            continue
        row = rows.iloc[0]
        lines.append(
            f"| {METHOD_LABELS[method]} | {pct(row.accuracy)} | "
            f"{pct(row.valid_answer_rate)} | {num(row.median_total_steps_correct, 0)} | "
            f"{pct(row.accuracy_difference_from_ordinary)} | {int(row.n_examples)} |"
        )
    return "\n".join(lines)


def render_figure(
    trajectory: pd.DataFrame, output_dir: Path, bootstrap: int, seed: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    learned = trajectory[
        (trajectory.method == "learned")
        & (~trajectory.forced_continuation.astype(bool))
    ].copy()
    per_example = (
        learned.groupby(
            ["example_id", "dataset", "K", "correct", "normalized_step_bin"],
            as_index=False,
        )
        .agg(
            normalized_k=("normalized_k", "mean"),
            progress=("progress", "mean"),
            drift=("cot_embedding_cosine_drift", "mean"),
            future_hit=("future_hit_at_10_no_special", "mean"),
            random_hit=("frequency_random_future_hit_at_10", "mean"),
        )
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, per_example.K.nunique()))
    color_by_k = dict(zip(sorted(per_example.K.unique()), colors))
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        len(DATASETS), 3, figsize=(14.5, 3.45 * len(DATASETS)), squeeze=False
    )
    panels = [
        ("progress", "Aligned CoT progress"),
        ("drift", "CoT-embedding cosine drift"),
        ("future_hit", "FutureHit@10"),
    ]
    for row_index, dataset in enumerate(DATASETS):
        dataset_frame = per_example[per_example.dataset == dataset]
        for column_index, (metric, label) in enumerate(panels):
            ax = axes[row_index, column_index]
            for k in sorted(dataset_frame.K.unique()):
                for correct, linestyle in ((True, "-"), (False, "--")):
                    group = dataset_frame[
                        (dataset_frame.K == k) & (dataset_frame.correct == correct)
                    ]
                    xs, means, lows, highs = [], [], [], []
                    for _, values in group.groupby("normalized_step_bin"):
                        array = values[metric].to_numpy(float)
                        draws = np.empty(bootstrap)
                        for index in range(bootstrap):
                            draws[index] = np.mean(
                                array[rng.integers(0, len(array), len(array))]
                            )
                        xs.append(values.normalized_k.mean())
                        means.append(array.mean())
                        lows.append(np.quantile(draws, 0.025))
                        highs.append(np.quantile(draws, 0.975))
                    suffix = "correct" if correct else "incorrect"
                    ax.plot(
                        xs,
                        means,
                        linestyle,
                        color=color_by_k[k],
                        label=f"Kmax={int(k)}, {suffix}",
                    )
                    ax.fill_between(xs, lows, highs, color=color_by_k[k], alpha=0.10)
            if metric == "future_hit":
                baseline = (
                    dataset_frame.groupby("normalized_step_bin")
                    .agg(x=("normalized_k", "mean"), y=("random_hit", "mean"))
                )
                ax.plot(
                    baseline.x,
                    baseline.y,
                    ":",
                    color="black",
                    label="frequency baseline",
                )
            ax.set_xlabel("Normalized realized latent step")
            ax.set_ylabel(label)
            ax.set_title(dataset.upper())
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Progress, drift, and semantic projection of GLR latent trajectories")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_figure.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "trajectory_figure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_noise_figure(aggregate_frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise = aggregate_frame[
        (aggregate_frame.method == "noisy_transition")
        & (~aggregate_frame.forced_continuation.astype(bool))
        & (aggregate_frame.K == K_STAR)
    ].copy()
    if noise.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    panels = [
        ("accuracy", "Accuracy"),
        ("final_cot_embedding_drift", "CoT-embedding drift"),
        ("future_hit_at_10", "FutureHit@10"),
    ]
    for dataset in DATASETS:
        data = noise[noise.dataset == dataset].sort_values("noise_sigma")
        for ax, (metric, label) in zip(axes, panels):
            ax.plot(data.noise_sigma, data[metric], marker="o", label=dataset.upper())
            ax.fill_between(
                data.noise_sigma,
                data[f"{metric}_ci_low"],
                data[f"{metric}_ci_high"],
                alpha=0.12,
            )
            ax.set_xlabel(r"Noise severity $\sigma$")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"Norm-preserving transition-noise sweep at K={K_STAR}")
    fig.tight_layout()
    fig.savefig(output_dir / "noise_severity_figure.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "noise_severity_figure.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def select_qualitative_examples(summary: pd.DataFrame) -> pd.DataFrame:
    learned = summary[
        (summary.method == "learned")
        & (summary.K == K_STAR)
        & (~summary.forced_continuation.astype(bool))
        & (summary.correct >= 0.5)
        & (summary.original_correct >= 0.5)
    ].copy()
    selections: list[pd.Series] = []
    for dataset in DATASETS:
        candidates = learned[learned.dataset == dataset]
        if len(candidates) < 2:
            raise ValueError(
                f"{dataset}: need at least two GLR-correct K=10 examples"
            )
        low = candidates.nsmallest(1, "final_cot_embedding_drift").iloc[0].copy()
        high = candidates.nlargest(1, "final_cot_embedding_drift").iloc[0].copy()
        if low.example_id == high.example_id:
            high = candidates.nlargest(2, "final_cot_embedding_drift").iloc[-1].copy()
        low["selection_reason"] = "correct_at_cap_low_endpoint_drift"
        high["selection_reason"] = "correct_at_cap_high_endpoint_drift"
        selections.extend((low, high))
    return pd.DataFrame(selections).reset_index(drop=True)


def qualitative_tex(selections: pd.DataFrame, neighbors: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        r"Dataset/ID & GLR correct & $K$ & $k$ & aligned $j$ & top-5 projections & future-window overlap & drift \\",
        r"\midrule",
    ]
    for row in selections.itertuples():
        example_id = row.example_id
        candidates = neighbors[
            (neighbors.example_id == example_id)
            & (neighbors.dataset == row.dataset)
            & (neighbors.method == "learned")
            & (neighbors.K == K_STAR)
            & (neighbors.projection_variant == "no_special")
        ]
        final_k = int(candidates.k.max())
        shown_steps = sorted({1, max(1, math.ceil(final_k / 2)), final_k})
        for shown_k in shown_steps:
            shown = (
                candidates[candidates.k == shown_k]
                .sort_values("rank")
                .head(5)
            )
            tokens = ", ".join(
                str(value)
                .replace("\\", r"\textbackslash{}")
                .replace("&", r"\&")
                .replace("_", r"\_")
                for value in shown.token_string
            )
            overlap = ", ".join(
                str(value)
                .replace("\\", r"\textbackslash{}")
                .replace("&", r"\&")
                .replace("_", r"\_")
                for value in shown[
                    shown.in_matched_future_window
                ].token_string
            )
            lines.append(
                f"{row.dataset}/{example_id} & {bool(row.correct)} & {K_STAR} & "
                f"{shown_k} & {int(shown.aligned_j.iloc[0])} & {tokens} & "
                f"{overlap} & "
                f"{shown.cot_embedding_cosine_drift.iloc[0]:.3f} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def mechanistic_tex(aggregate_frame: pd.DataFrame) -> str:
    selected = aggregate_frame[
        (aggregate_frame.method == "learned")
        & (aggregate_frame.K == K_STAR)
        & (~aggregate_frame.forced_continuation.astype(bool))
        & (aggregate_frame.dataset.isin([*DATASETS, "pooled"]))
    ].copy()
    order = {name: index for index, name in enumerate([*DATASETS, "pooled"])}
    selected["_order"] = selected.dataset.map(order)
    selected = selected.sort_values("_order")
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Dataset & $N$ & Acc. $\uparrow$ & Realized $K$ & Progress $\uparrow$ & Drift (correct) $\downarrow$ & Drift (incorrect) & $\Delta$ drift & FutureHit@10 $\uparrow$ \\",
        r"\midrule",
    ]
    for row in selected.itertuples():
        label = "All" if row.dataset == "pooled" else row.dataset.upper()
        lines.append(
            f"{label} & {int(row.n_examples)} & {100*row.accuracy:.1f} & "
            f"{row.mean_actual_latent_steps:.1f} & "
            f"{row.final_progress:.3f} & {row.cot_embedding_drift_correct:.3f} & "
            f"{row.cot_embedding_drift_incorrect:.3f} & "
            f"{row.incorrect_minus_correct_drift:.3f} & "
            f"{100*row.future_hit_at_10_no_special:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def k_endpoint_tex(aggregate_frame: pd.DataFrame) -> str:
    selected = aggregate_frame[
        (aggregate_frame.method == "learned")
        & (~aggregate_frame.forced_continuation.astype(bool))
        & (aggregate_frame.dataset.isin(("gsm8k", "math500")))
    ].sort_values(["dataset", "K"])
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & $K_{\max}$ & $N$ & Realized $K$ & Acc. $\uparrow$ & Progress $\uparrow$ & CoT-emb. drift $\downarrow$ & FutureHit@10 $\uparrow$ \\",
        r"\midrule",
    ]
    for row in selected.itertuples():
        lines.append(
            f"{row.dataset.upper()} & {int(row.K)} & {int(row.n_examples)} & "
            f"{row.mean_actual_latent_steps:.1f} & "
            f"{100*row.accuracy:.1f} & {row.final_progress:.3f} & "
            f"{row.final_cot_embedding_drift:.3f} & "
            f"{100*row.future_hit_at_10_no_special:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def ablation_tex(aggregate_frame: pd.DataFrame) -> str:
    selected = aggregate_frame[
        (aggregate_frame.dataset == "pooled")
        & (aggregate_frame.K == K_STAR)
        & (~aggregate_frame.forced_continuation.astype(bool))
    ]
    selected = selected[
        (selected.method != "noisy_transition")
        | np.isclose(selected.noise_sigma.astype(float), 1.0, equal_nan=False)
    ]
    order = [
        "learned",
        "zero_transition",
        "repeated_think",
        "random_head_norm_matched",
        "noisy_transition",
        "shuffled_transition",
        "absolute_prediction",
    ]
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & $N$ & Exact acc. $\uparrow$ & Total steps $\downarrow$ & Resume progress $\uparrow$ & Immediate close $\downarrow$ & CoT-emb. drift $\downarrow$ & FutureHit@10 $\uparrow$ \\",
        r"\midrule",
    ]
    for method in order:
        rows = selected[selected.method == method]
        if rows.empty:
            continue
        row = rows.iloc[0]
        label = METHOD_LABELS[method].replace("_", r"\_").replace("%", r"\%")
        lines.append(
            f"{label} & {int(row.n_examples)} & {100*row.control_accuracy:.1f} & "
            f"{row.median_total_steps_control_correct:.0f} & "
            f"{row.resume_progress:.3f} & {100*row.immediate_close_rate:.1f} & "
            f"{row.final_cot_embedding_drift:.3f} & "
            f"{100*row.future_hit_at_10_no_special:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or args.cache_root / "derived"
    output_dir = args.output_dir or args.cache_root / "rendered"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(results_dir / "per_example_summary.parquet")
    trajectory = pd.read_parquet(results_dir / "trajectory_steps.parquet")
    neighbors = pd.read_parquet(results_dir / "nearest_neighbors.parquet")
    averaged = seed_average(raw)
    aggregate_frame = aggregate(
        averaged, raw, args.bootstrap_resamples, args.bootstrap_seed
    )
    aggregate_frame.to_csv(results_dir / "aggregate_results.csv", index=False)
    control_averaged = matched_control_subset(averaged)
    control_raw = matched_control_subset(raw)
    control_aggregate = (
        aggregate(
            control_averaged,
            control_raw,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
        if not control_averaged.empty
        else pd.DataFrame()
    )
    control_aggregate.to_csv(
        results_dir / "control_subset_aggregate_results.csv", index=False
    )
    trajectory_aggregate = trajectory_aggregates(
        trajectory, args.bootstrap_resamples, args.bootstrap_seed
    )
    trajectory_aggregate.to_csv(
        results_dir / "trajectory_aggregate_results.csv", index=False
    )
    paired = paired_comparisons(
        averaged, args.bootstrap_resamples, args.bootstrap_seed
    )
    paired.to_csv(results_dir / "paired_comparisons.csv", index=False)
    regression = regression_results(
        raw,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    regression.to_csv(results_dir / "regression_results.csv", index=False)

    sections = []
    for dataset in ("pooled", *DATASETS):
        if not control_aggregate.empty:
            sections.append(controls_table(control_aggregate, dataset))
        sections.append(drift_table(aggregate_frame, averaged, dataset))
        if aggregate_frame.forced_continuation.astype(bool).any():
            sections.append(forced_table(aggregate_frame, dataset))
    (output_dir / "tables.md").write_text(
        "\n\n".join(sections) + "\n", encoding="utf-8"
    )
    (output_dir / "ablation_table.tex").write_text(
        (
            ablation_tex(control_aggregate)
            if not control_aggregate.empty
            else "% Controls were disabled for this run.\n"
        ),
        encoding="utf-8",
    )
    (output_dir / "mechanistic_table.tex").write_text(
        mechanistic_tex(aggregate_frame), encoding="utf-8"
    )
    (output_dir / "k_endpoint_table.tex").write_text(
        k_endpoint_tex(aggregate_frame), encoding="utf-8"
    )
    aggregate_frame[
        (aggregate_frame.method == "learned")
        & (aggregate_frame.K == K_STAR)
        & (~aggregate_frame.forced_continuation.astype(bool))
        & aggregate_frame.dataset.isin([*DATASETS, "pooled"])
    ][
        [
            "dataset",
            "n_examples",
            "accuracy",
            "mean_actual_latent_steps",
            "final_progress",
            "cot_embedding_drift_correct",
            "cot_embedding_drift_incorrect",
            "incorrect_minus_correct_drift",
            "future_hit_at_10_no_special",
        ]
    ].to_csv(output_dir / "mechanistic_table.csv", index=False)
    aggregate_frame[
        (aggregate_frame.method == "learned")
        & (~aggregate_frame.forced_continuation.astype(bool))
        & aggregate_frame.dataset.isin(("gsm8k", "math500"))
    ][
        [
            "dataset",
            "K",
            "n_examples",
            "mean_actual_latent_steps",
            "accuracy",
            "final_progress",
            "final_cot_embedding_drift",
            "future_hit_at_10_no_special",
        ]
    ].to_csv(output_dir / "k_endpoint_table.csv", index=False)
    if not control_aggregate.empty:
        control_aggregate[
            (control_aggregate.dataset == "pooled")
            & (control_aggregate.K == K_STAR)
            & (~control_aggregate.forced_continuation.astype(bool))
        ].to_csv(output_dir / "ablation_table.csv", index=False)
    selections = select_qualitative_examples(raw)
    selection_columns = [
        "dataset",
        "example_id",
        "K",
        "correct",
        "original_correct",
        "correctness_key",
        "selection_reason",
        "final_cot_embedding_drift",
        "final_progress",
        "future_hit_at_10_no_special",
    ]
    selections[selection_columns].to_csv(
        output_dir / "qualitative_example_selection.csv", index=False
    )
    (output_dir / "nearest_neighbor_examples.tex").write_text(
        qualitative_tex(selections, neighbors), encoding="utf-8"
    )
    render_figure(
        trajectory, output_dir, args.bootstrap_resamples, args.bootstrap_seed
    )
    render_noise_figure(aggregate_frame, output_dir)

    learned = aggregate_frame[
        (aggregate_frame.dataset == "pooled")
        & (aggregate_frame.method == "learned")
        & (aggregate_frame.K == K_STAR)
        & (~aggregate_frame.forced_continuation.astype(bool))
    ].iloc[0]
    regression_e = regression[
        (regression.K_scope == str(K_STAR))
        & (regression.coefficient == "final_cot_embedding_drift")
    ].iloc[0]
    forced_learned = aggregate_frame[
        (aggregate_frame.dataset == "pooled")
        & (aggregate_frame.method == "learned")
        & (aggregate_frame.K == K_STAR)
        & (aggregate_frame.forced_continuation.astype(bool))
    ]
    forced_sentence = (
        f" Under 16-token forced continuation, learned GLR retains "
        f"{100*forced_learned.iloc[0].accuracy:.1f}% accuracy."
        if not forced_learned.empty
        else ""
    )
    absolute_available = (aggregate_frame.method == "absolute_prediction").any()
    absolute_sentence = (
        " The matched absolute-embedding baseline is included in the causal table."
        if absolute_available
        else ""
    )
    controls_available = (
        aggregate_frame.method != "learned"
    ).any()
    controls_sentence = (
        " On a fixed random 100-example subset per dataset, the position, "
        "formatting, random-head, and noisy-transition controls are reported "
        "in the accompanying causal table."
        if controls_available
        else ""
    )
    paragraph = (
        "We agree that the original K=0 ablation shows that latent rollout causally "
        "changes decoding, but does not by itself establish the geometry or semantic "
        "content of the latent trajectory. We therefore evaluated all test examples "
        "from GSM8K, MATH500, and SVAMP. We teacher-forced each CoT-SFT "
        "trace through the GLR checkpoint and monotonically aligned GLR latent "
        "states to this explicit CoT trajectory in the checkpoint's contextual "
        f"state space. At the paper setting K*={K_STAR}, learned GLR reaches aligned progress "
        f"{learned.final_progress:.3f} with {learned.mean_actual_latent_steps:.1f} "
        "realized latent steps on average, CoT-embedding drift "
        f"{learned.final_cot_embedding_drift:.3f}, and FutureHit@10 "
        f"{100*learned.future_hit_at_10_no_special:.1f}%. The dataset- and realized-step-controlled accuracy "
        f"odds ratio is {regression_e.odds_ratio:.3f} per standard-deviation increase "
        "in CoT-embedding drift at fixed K=10."
        + controls_sentence
        + forced_sentence
        + absolute_sentence
        + " These results support the narrower "
        "claim that learned GLR transitions carry task-relevant state toward later "
        "portions of the model's discrete reasoning trajectory; they do not prove "
        "that each latent state corresponds to a uniquely verbalizable reasoning step."
    )
    (output_dir / "summary_paragraph.txt").write_text(
        paragraph + "\n", encoding="utf-8"
    )
    print(f"Rendered outputs written to {output_dir}")


if __name__ == "__main__":
    main()
