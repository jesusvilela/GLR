# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Validate inference integrity, statistics, tables, and optional LaTeX compile."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from robustness_common import (
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    CORRECTNESS_FIELDS,
    REQUIRED_RECORD_FIELDS,
    SEEDS,
)

REQUIRED_OUTPUTS = (
    "run_manifest.json",
    "greedy_full.parquet",
    "greedy_constrained_prefixes.parquet",
    "sampled_constrained.parquet",
    "flagged_repetition_examples.md",
    "statistics.json",
    "greedy_accuracy_stats.csv",
    "greedy_length_stats.csv",
    "sampled_robustness_stats.csv",
    "table_greedy.md",
    "table_sampling.md",
    "table_length.md",
    "table_main.tex",
    "table_appendix_greedy.tex",
    "table_appendix_sampling.tex",
    "summary_paragraph.md",
    "figure_accuracy_robustness.pdf",
    "figure_accuracy_robustness.png",
    "figure_length_reduction.pdf",
    "figure_length_reduction.png",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compile-latex-if-available", action="store_true")
    return parser.parse_args()


def validate_schema(frame: pd.DataFrame, label: str) -> None:
    missing = sorted(set(REQUIRED_RECORD_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing fields {missing}")
    if not (frame["total_steps"] == frame["text_steps"]).all():
        raise ValueError(f"{label}: total-step accounting failure")
    token_lengths = frame["generated_token_ids"].map(len)
    if not (token_lengths == frame["total_steps"]).all():
        raise ValueError(f"{label}: post-prompt step/token ID mismatch")
    if not (frame["total_steps"] <= frame["total_step_budget"]).all():
        raise ValueError(f"{label}: record exceeds total-step budget")
    if not (frame.loc[frame["method"] == "cot_sft", "latent_steps"] == 0).all():
        raise ValueError(f"{label}: CoT-SFT has nonzero latent steps")
    if not (frame.loc[frame["method"] == "glr_k10", "latent_steps"] == 10).all():
        raise ValueError(f"{label}: GLR does not have exactly ten latent steps")


def validate_method_alignment(frame: pd.DataFrame, label: str) -> None:
    for dataset, spec in DATASETS.items():
        subset = frame[frame["dataset"] == dataset]
        method_ids: dict[str, set[str]] = {}
        for method in ("cot_sft", "glr_k10"):
            rows = subset[subset["method"] == method]
            method_ids[method] = set(rows["example_id"].astype(str))
            if len(method_ids[method]) != spec.expected_count:
                raise ValueError(
                    f"{label}/{dataset}/{method}: expected {spec.expected_count} "
                    f"unique examples, got {len(method_ids[method])}"
                )
        if method_ids["cot_sft"] != method_ids["glr_k10"]:
            raise ValueError(f"{label}/{dataset}: method IDs do not align")
        for example_id, group in subset.groupby("example_id"):
            if group["gold_answer"].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{label}/{dataset}/{example_id}: gold answers do not match"
                )
            if group["prompt_hash"].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{label}/{dataset}/{example_id}: prompt hashes do not match"
                )


def validate_cross_record_sets(
    full: pd.DataFrame, prefix: pd.DataFrame, sampled: pd.DataFrame
) -> None:
    for dataset, spec in DATASETS.items():
        expected_budgets = {
            "full": spec.full_cap,
            "prefix": spec.constrained_budget,
            "sampled": spec.constrained_budget,
        }
        subsets = {
            "full": full[full["dataset"] == dataset],
            "prefix": prefix[prefix["dataset"] == dataset],
            "sampled": sampled[sampled["dataset"] == dataset],
        }
        reference: dict[str, tuple[str, str]] = {}
        for _, row in subsets["full"].iterrows():
            key = str(row["example_id"])
            value = (str(row["prompt_hash"]), str(row["gold_answer"]))
            if key in reference and reference[key] != value:
                raise ValueError(f"{dataset}/{key}: inconsistent full records")
            reference[key] = value
        for label, subset in subsets.items():
            if not (subset["total_step_budget"] == expected_budgets[label]).all():
                raise ValueError(f"{dataset}/{label}: wrong total-step budget")
            for _, row in subset.iterrows():
                key = str(row["example_id"])
                value = (str(row["prompt_hash"]), str(row["gold_answer"]))
                if reference.get(key) != value:
                    raise ValueError(
                        f"{dataset}/{key}: prompt/gold differs across record sets"
                    )


def validate_prefix_reports(output_dir: Path) -> None:
    for dataset in DATASETS:
        for method in ("cot_sft", "glr_k10"):
            path = (
                output_dir
                / "validation"
                / f"prefix_equivalence_{dataset}_{method}.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report["passed"]:
                raise ValueError(f"Prefix validation did not pass: {path}")
            mode = report.get("validation_mode", "direct_prefix_equivalence")
            if mode == "cache_alignment_and_token_prefix_derivation":
                if report["n_checked"] != DATASETS[dataset].expected_count:
                    raise ValueError(
                        f"Cache alignment did not cover the full test set: {path}"
                    )
                if report.get("direct_generation_checks") != 0:
                    raise ValueError(
                        f"Cache-only report has unexpected direct checks: {path}"
                    )
            else:
                if report["n_checked"] < 20:
                    raise ValueError(f"Too few direct prefix checks: {path}")
                if not all(check["matches"] for check in report["checks"]):
                    raise ValueError(f"Prefix mismatch recorded in {path}")


def validate_sampled(frame: pd.DataFrame) -> None:
    validate_method_alignment(frame, "sampled_constrained")
    for dataset, spec in DATASETS.items():
        for method in ("cot_sft", "glr_k10"):
            subset = frame[(frame["dataset"] == dataset) & (frame["method"] == method)]
            if len(subset) != spec.expected_count * 3:
                raise ValueError(
                    f"{dataset}/{method}: expected {spec.expected_count * 3} "
                    f"sampled rows, got {len(subset)}"
                )
            counts = subset.groupby("example_id").size()
            if not (counts == 3).all():
                raise ValueError(f"{dataset}/{method}: not exactly 3 samples/problem")
            for _, group in subset.groupby("example_id"):
                ordered = group.sort_values("sample_id")
                if ordered["sample_id"].tolist() != [0, 1, 2]:
                    raise ValueError(f"{dataset}/{method}: invalid sample IDs")
                if ordered["seed"].astype(int).tolist() != list(SEEDS):
                    raise ValueError(f"{dataset}/{method}: invalid seed list")
            if not (subset["decoding"] == "sampled_t0.6_p0.95").all():
                raise ValueError(f"{dataset}/{method}: sampled decoding label mismatch")
            if (subset["correct"].astype(bool) & subset["hit_cap"].astype(bool)).any():
                raise ValueError(
                    f"{dataset}/{method}: cap-reaching sample credited as "
                    "correct under the paper budget definition"
                )


def validate_correctness(
    full: pd.DataFrame, prefix: pd.DataFrame, sampled: pd.DataFrame
) -> None:
    for frame_name, frame in (
        ("greedy_full", full),
        ("greedy_constrained_prefixes", prefix),
        ("sampled_constrained", sampled),
    ):
        required = {"metric_correct", "correctness_mode"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{frame_name}: missing scoring fields {missing}")
        for dataset in DATASETS:
            for method in ("cot_sft", "glr_k10"):
                subset = frame[
                    (frame["dataset"] == dataset) & (frame["method"] == method)
                ]
                expected_mode = CORRECTNESS_FIELDS[dataset][method]
                if not (subset["correctness_mode"] == expected_mode).all():
                    raise ValueError(
                        f"{frame_name}/{dataset}/{method}: expected "
                        f"correctness mode {expected_mode!r}"
                    )

    for dataset in DATASETS:
        for method in ("cot_sft", "glr_k10"):
            subset = prefix[
                (prefix["dataset"] == dataset) & (prefix["method"] == method)
            ]
            recomputed = subset["full_correct"].astype(bool) & subset[
                "completed_within_budget"
            ].astype(bool)
            if not (recomputed == subset["correct"].astype(bool)).all():
                raise ValueError(
                    f"{dataset}/{method}: greedy budget accuracy does not equal "
                    "full correctness AND completion within budget"
                )


def validate_no_placeholders(output_dir: Path) -> None:
    report_files = [
        path
        for path in output_dir.iterdir()
        if path.suffix in (".md", ".tex", ".csv", ".json")
    ]
    forbidden = ("[INSERT", "INSERT DATA", "TODO", "PLACEHOLDER")
    for path in report_files:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                raise ValueError(f"{path}: unresolved marker {marker!r}")


def compile_latex(output_dir: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("pdflatex unavailable; structural LaTeX checks only")
        return
    for name in (
        "table_main.tex",
        "table_appendix_greedy.tex",
        "table_appendix_sampling.tex",
    ):
        table = (output_dir / name).read_text(encoding="utf-8")
        document = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\begin{document}\n"
            f"{table}\n"
            "\\end{document}\n"
        )
        with tempfile.TemporaryDirectory(prefix="glr_latex_") as temp:
            source = Path(temp) / "table.tex"
            source.write_text(document, encoding="utf-8")
            subprocess.run(
                [pdflatex, "-halt-on-error", "-interaction=batchmode", source.name],
                cwd=temp,
                check=True,
                capture_output=True,
                text=True,
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required outputs: {missing}")

    full = pd.read_parquet(output_dir / "greedy_full.parquet")
    prefix = pd.read_parquet(output_dir / "greedy_constrained_prefixes.parquet")
    sampled = pd.read_parquet(output_dir / "sampled_constrained.parquet")
    for label, frame in (
        ("greedy_full", full),
        ("greedy_constrained_prefixes", prefix),
        ("sampled_constrained", sampled),
    ):
        validate_schema(frame, label)
    validate_method_alignment(full, "greedy_full")
    validate_method_alignment(prefix, "greedy_constrained_prefixes")
    validate_sampled(sampled)
    validate_correctness(full, prefix, sampled)
    validate_cross_record_sets(full, prefix, sampled)
    validate_prefix_reports(output_dir)
    validate_no_placeholders(output_dir)

    statistics = json.loads(
        (output_dir / "statistics.json").read_text(encoding="utf-8")
    )
    if statistics["metadata"]["bootstrap_replicates"] != 10_000:
        raise ValueError("statistics.json does not use 10,000 bootstrap replicates")
    if args.compile_latex_if_available:
        compile_latex(output_dir)
    print(f"All validation checks passed: {output_dir}")


if __name__ == "__main__":
    main()
