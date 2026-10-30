# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Model-free validation of matched-stack caches and generated artifacts."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_efficiency import load_cache, validate_records
from efficiency_common import (
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    METHODS,
    atomic_write_json,
    canonical_hash,
    load_dataset,
    load_static_manifest,
)

REQUIRED_OUTPUTS = (
    "summaries/per_example.csv",
    "summaries/per_example.parquet",
    "summaries/aggregate_metrics.json",
    "summaries/aggregate_metrics.csv",
    "summaries/bootstrap_intervals.json",
    "summaries/environment.json",
    "summaries/cache_completeness.json",
    "tables/efficiency_table.md",
    "tables/efficiency_table.tex",
    "figures/efficiency_latency.pdf",
    "figures/efficiency_latency.png",
    "figures/efficiency_latency_data.csv",
    "logs/head_call_validation.json",
    "logs/timing_stability.json",
    "logs/run.log",
    "traces/call_traces.jsonl",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=("cache", "outputs"), required=True)
    parser.add_argument("--compile-latex-if-available", action="store_true")
    return parser.parse_args()


def recompute_flops(record: dict[str, Any], architecture: dict[str, int]) -> int:
    d = int(architecture["hidden_size"])
    f = int(architecture["intermediate_size"])
    layers = int(architecture["num_hidden_layers"])
    d_kv = int(architecture["num_key_value_heads"]) * int(architecture["head_dim"])
    vocabulary = int(architecture["vocab_size"])
    backbone = 0
    for call in record["backbone_call_trace"]:
        q, s = int(call["query_length"]), int(call["context_length"])
        backbone += layers * (
            q * (4 * d * d + 4 * d * d_kv + 6 * d * f) + 4 * d * q * s
        )
    vocab = int(record["vocabulary_head_positions"]) * 2 * d * vocabulary
    transition = int(record["transition_head_positions"]) * 2 * d * d
    return int(backbone + vocab + transition)


def cache_validation(output_dir: Path) -> dict[str, Any]:
    static = load_static_manifest(output_dir)
    frame, completeness = load_cache(output_dir)
    if not completeness["complete"]:
        raise ValueError(f"Incomplete cache: {completeness}")
    head_validation = validate_records(frame, static)
    if "0.6B" in json.dumps(static):
        raise ValueError("A 0.6B path appears in the static manifest")
    expected_architecture = {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
    }
    if static["architecture"] != expected_architecture:
        raise ValueError(f"Architecture mismatch: {static['architecture']}")
    records = frame.to_dict(orient="records")
    source_rows = {
        benchmark: {
            str(row["id"]): (str(row["q"]), str(row["a"]))
            for row in load_dataset(benchmark)
        }
        for benchmark in DATASETS
    }
    for record in records:
        expected_prompt, expected_gold = source_rows[record["benchmark"]][
            str(record["example_id"])
        ]
        if (
            record["prompt"] != expected_prompt
            or record["gold_answer"] != expected_gold
        ):
            raise ValueError(
                f"Prompt/gold mismatch: {record['benchmark']}/"
                f"{record['example_id']}/{record['method']}"
            )
        if recompute_flops(record, static["architecture"]) != int(
            record["approximate_flops"]
        ):
            raise ValueError(
                f"FLOP mismatch: {record['benchmark']}/"
                f"{record['example_id']}/{record['method']}"
            )
    fairness = static["semantic_payload"]["decoder"]
    expected_fairness = {
        "greedy": True,
        "sampling": False,
        "dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "use_kv_cache": True,
        "batch_size": 1,
        "torch_compile": False,
        "cuda_graphs": False,
        "quantization": False,
        "speculative_decoding": False,
    }
    for key, expected in expected_fairness.items():
        if fairness.get(key) != expected:
            raise ValueError(f"Fairness setting mismatch: {key}")
    if fairness.get("eos_token_ids") != [151645, 151643]:
        raise ValueError("Unexpected stopping-token IDs")
    if fairness.get("prompt_construction") != (
        "raw q field, without an added chat template"
    ):
        raise ValueError("Unexpected prompt construction")

    environment_paths = sorted(
        (output_dir / "manifests").glob("environment_shard*.json")
    )
    if len(environment_paths) != 5:
        raise ValueError(
            f"Expected five environment manifests, got {len(environment_paths)}"
        )
    environments = [
        json.loads(path.read_text(encoding="utf-8")) for path in environment_paths
    ]
    if any(env.get("gpu_shared_with_other_process") for env in environments):
        raise ValueError("A shard reports a known concurrent GPU process")
    gpu_profiles = sorted(
        {
            (
                str(env.get("gpu_name")),
                int(env.get("gpu_total_memory_bytes", 0)),
            )
            for env in environments
        }
    )
    timing_by_manifest = {
        str(path.relative_to(output_dir)): env["timing_hash"]
        for path, env in zip(environment_paths, environments)
    }
    for record in records:
        if (
            timing_by_manifest.get(record["environment_manifest"])
            != record["timing_hash"]
        ):
            raise ValueError(
                f"Environment/timing mismatch: {record['benchmark']}/"
                f"{record['example_id']}/{record['method']}"
            )
    pair_environment_counts = frame.groupby(
        ["benchmark", "example_id"], sort=False
    ).agg(
        environment_manifests=("environment_manifest", "nunique"),
        timing_hashes=("timing_hash", "nunique"),
    )
    if (
        pair_environment_counts["environment_manifests"].ne(1).any()
        or pair_environment_counts["timing_hashes"].ne(1).any()
    ):
        raise ValueError(
            "At least one CoT-SFT/GLR example pair was timed on different "
            "hardware or software environments"
        )

    # A model-free second-pass cache audit: every expected record has the
    # experiment semantic identity, so a resume would execute zero examples.
    reusable = sum(
        record.get("status") == "complete"
        and record.get("semantic_hash") == static["semantic_hash"]
        for record in records
    )
    expected_rows = 2 * sum(spec.expected_count for spec in DATASETS.values())
    if reusable != expected_rows:
        raise ValueError("Second-pass cache audit found non-reusable records")

    inspection = {}
    for benchmark in DATASETS:
        inspection[benchmark] = {}
        for method in METHODS:
            subset = (
                frame[(frame["benchmark"] == benchmark) & (frame["method"] == method)]
                .sort_values("example_id")
                .head(10)
            )
            inspection[benchmark][method] = [
                {
                    "example_id": row["example_id"],
                    "prompt_hash": row["prompt_hash"],
                    "stop_reason": row["stop_reason"],
                    "correct": bool(row["correct"]),
                    "latent_steps": int(row["latent_step_count"]),
                    "text_tokens": int(row["generated_text_token_count"]),
                    "total_steps": int(row["total_post_prompt_step_count"]),
                    "generation_latency_ms": float(row["generation_latency_ms"]),
                    "generated_text_preview": str(row["generated_text"])[:300],
                }
                for _, row in subset.iterrows()
            ]
    atomic_write_json(
        output_dir / "logs" / "manual_inspection_10_per_condition.json",
        inspection,
    )
    return {
        "passed": True,
        "records": len(frame),
        "conditions": completeness["conditions"],
        "head_invariants": head_validation,
        "flop_records_recomputed": len(frame),
        "fairness_settings": expected_fairness,
        "gpu_profiles": [
            {
                "gpu_name": name,
                "gpu_total_memory_bytes": total_memory,
            }
            for name, total_memory in gpu_profiles
        ],
        "paired_environment_invariant": True,
        "environment_manifests": len(environments),
        "resume_dry_run_examples_to_execute": expected_rows - reusable,
        "manual_inspection_records": 10 * len(DATASETS) * len(METHODS),
        "analysis_entry_points_load_models": False,
    }


def compare_reference(output_dir: Path) -> dict[str, Any]:
    static = load_static_manifest(output_dir)
    aggregate = json.loads(
        (output_dir / "summaries" / "aggregate_metrics.json").read_text(
            encoding="utf-8"
        )
    )["aggregate"]
    output = {}
    for benchmark in DATASETS:
        output[benchmark] = {}
        for method in METHODS:
            measured = aggregate[benchmark][method]
            reference = static["reference_summary"][benchmark][method]
            accuracy_difference = (
                measured["accuracy_percent"] - reference["accuracy_percent"]
            )
            mean_step_difference = (
                measured["mean_total_steps"] - reference["mean_total_steps"]
            )
            reference_steps = float(reference["mean_total_steps"])
            relative_step_difference_percent = (
                100 * mean_step_difference / reference_steps
                if reference_steps
                else None
            )
            substantial = abs(accuracy_difference) > 2.0 or (
                relative_step_difference_percent is not None
                and abs(relative_step_difference_percent) > 5.0
            )
            output[benchmark][method] = {
                "matched_accuracy_percent": measured["accuracy_percent"],
                "reference_accuracy_percent": reference["accuracy_percent"],
                "accuracy_difference_pp": accuracy_difference,
                "matched_mean_total_steps": measured["mean_total_steps"],
                "reference_mean_total_steps": reference["mean_total_steps"],
                "mean_step_difference": mean_step_difference,
                "relative_mean_step_difference_percent": (
                    relative_step_difference_percent
                ),
                "substantial_difference_flag": substantial,
                "flag_thresholds": {
                    "absolute_accuracy_difference_pp": 2.0,
                    "absolute_relative_mean_step_difference_percent": 5.0,
                },
                "note": (
                    "Reference uses the scored greedy decode truncated/rescored at "
                    "the same cap. GLR matched-stack output omits latent placeholder "
                    "token IDs while counting ten latent steps explicitly."
                ),
            }
    atomic_write_json(output_dir / "logs" / "reference_comparison.json", output)
    return output


def compile_latex(output_dir: Path) -> None:
    executable = shutil.which("pdflatex")
    if executable is None:
        return
    for relative in (
        "tables/efficiency_table.tex",
    ):
        table = (output_dir / relative).read_text(encoding="utf-8")
        document = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage[margin=0.5in]{geometry}\n"
            "\\begin{document}\n"
            f"{table}\n"
            "\\end{document}\n"
        )
        with tempfile.TemporaryDirectory(prefix="efficiency_latex_") as temp:
            source = Path(temp) / "table.tex"
            source.write_text(document, encoding="utf-8")
            subprocess.run(
                [
                    executable,
                    "-halt-on-error",
                    "-interaction=batchmode",
                    source.name,
                ],
                cwd=temp,
                check=True,
                capture_output=True,
                text=True,
            )


def output_validation(output_dir: Path, compile_latex_flag: bool) -> dict[str, Any]:
    missing = [
        relative
        for relative in REQUIRED_OUTPUTS
        if not (output_dir / relative).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing outputs: {missing}")
    per_example = pd.read_csv(output_dir / "summaries" / "per_example.csv")
    expected_rows = 2 * sum(spec.expected_count for spec in DATASETS.values())
    if len(per_example) != expected_rows:
        raise ValueError("per_example.csv row count mismatch")
    plotting = pd.read_csv(output_dir / "figures" / "efficiency_latency_data.csv")
    if len(plotting) != expected_rows:
        raise ValueError("Plotting-data row count mismatch")
    for relative in (
        "tables/efficiency_table.md",
        "tables/efficiency_table.tex",
    ):
        text = (output_dir / relative).read_text(encoding="utf-8")
        if any(marker in text for marker in ("TODO", "PLACEHOLDER", "[INSERT")):
            raise ValueError(f"Placeholder in {relative}")
    if compile_latex_flag:
        compile_latex(output_dir)
    reference = compare_reference(output_dir)
    return {
        "passed": True,
        "required_outputs": len(REQUIRED_OUTPUTS),
        "per_example_rows": len(per_example),
        "plotting_rows": len(plotting),
        "reference_comparison_hash": canonical_hash(reference),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "logs" / f"validation_{args.stage}.json"
    try:
        if args.stage == "cache":
            report = cache_validation(output_dir)
        else:
            report = output_validation(output_dir, args.compile_latex_if_available)
        atomic_write_json(report_path, report)
        print(report_path)
    except Exception as error:
        atomic_write_json(
            report_path,
            {
                "passed": False,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
