# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Generate all summaries, confidence intervals, tables, and figures from cache."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from efficiency_common import (
    ANALYSIS_SEED,
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    K_BY_METHOD,
    METHODS,
    N_BOOTSTRAP,
    SCHEMA_VERSION,
    atomic_write_json,
    ensure_output_tree,
    load_dataset,
    load_static_manifest,
    record_path,
)

METHOD_LABELS = {"cot_sft": "CoT-SFT", "glr_k10": "GLR-10"}
COLORS = {"cot_sft": "#4C78A8", "glr_k10": "#E45756"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--analysis-seed", type=int, default=ANALYSIS_SEED)
    return parser.parse_args()


def load_cache(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    completeness: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "conditions": {}}
    for benchmark, spec in DATASETS.items():
        source = load_dataset(benchmark)
        for method in METHODS:
            key = f"{benchmark}/{method}"
            counts = {
                "expected": spec.expected_count,
                "completed": 0,
                "failed": 0,
                "missing": 0,
                "skipped_from_cache": 0,
            }
            for index, example in enumerate(source):
                path = record_path(
                    output_dir, benchmark, index, str(example["id"]), method
                )
                if not path.exists():
                    counts["missing"] += 1
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    counts["failed"] += 1
                    continue
                if record.get("status") != "complete":
                    counts["failed"] += 1
                    continue
                counts["completed"] += 1
                rows.append(record)
            completeness["conditions"][key] = counts
    completeness["complete"] = all(
        values["completed"] == values["expected"]
        and values["failed"] == 0
        and values["missing"] == 0
        for values in completeness["conditions"].values()
    )
    frame = pd.DataFrame(rows)
    return frame, completeness


def validate_records(frame: pd.DataFrame, static: dict[str, Any]) -> dict[str, Any]:
    expected_rows = 2 * sum(spec.expected_count for spec in DATASETS.values())
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} complete rows, got {len(frame)}")
    if set(frame["schema_version"]) != {SCHEMA_VERSION}:
        raise ValueError("Record schema mismatch")
    if set(frame["semantic_hash"]) != {static["semantic_hash"]}:
        raise ValueError("Record/static semantic hash mismatch")
    if frame.duplicated(["benchmark", "example_id", "method"]).any():
        raise ValueError("Duplicate benchmark/example/method records")
    problems = []
    for record in frame.to_dict(orient="records"):
        benchmark, method = record["benchmark"], record["method"]
        if int(record["K"]) != K_BY_METHOD[method]:
            problems.append(f"{benchmark}/{record['example_id']}/{method}: wrong K")
        if int(record["latent_step_count"]) != K_BY_METHOD[method]:
            problems.append(
                f"{benchmark}/{record['example_id']}/{method}: latent count"
            )
        total = int(record["generated_text_token_count"]) + int(
            record["latent_step_count"]
        )
        if total != int(record["total_post_prompt_step_count"]):
            problems.append(f"{benchmark}/{record['example_id']}/{method}: total steps")
        if total > DATASETS[benchmark].max_total_steps:
            problems.append(f"{benchmark}/{record['example_id']}/{method}: over budget")
        if int(record["backbone_call_count"]) != len(record["backbone_call_trace"]):
            problems.append(
                f"{benchmark}/{record['example_id']}/{method}: backbone trace"
            )
        if int(record["backbone_call_count"]) != total:
            problems.append(
                f"{benchmark}/{record['example_id']}/{method}: "
                "one-backbone-call-per-post-prompt-step invariant"
            )
        if int(record["vocabulary_head_call_count"]) != int(
            record["generated_text_token_count"]
        ):
            problems.append(
                f"{benchmark}/{record['example_id']}/{method}: "
                "one-vocabulary-call-per-text-token invariant"
            )
        expected_context = int(record["prompt_token_count"])
        for call_index, call in enumerate(record["backbone_call_trace"]):
            expected_query = int(record["prompt_token_count"]) if call_index == 0 else 1
            if (
                int(call["query_length"]) != expected_query
                or int(call["context_length"]) != expected_context
            ):
                problems.append(
                    f"{benchmark}/{record['example_id']}/{method}: "
                    f"query/context trace at call {call_index}"
                )
                break
            expected_context += 1
        transition = [
            item for item in record["head_call_trace"] if item["head"] == "transition"
        ]
        forbidden_vocab = [
            item
            for item in record["head_call_trace"]
            if item["head"] == "vocabulary"
            and item["phase"].startswith("latent_update")
        ]
        invalid_vocab_phase = [
            item
            for item in record["head_call_trace"]
            if item["head"] == "vocabulary"
            and item["phase"] not in ("text_decode", "post_latent_text_decode")
        ]
        if (
            len(transition) != K_BY_METHOD[method]
            or forbidden_vocab
            or invalid_vocab_phase
        ):
            problems.append(
                f"{benchmark}/{record['example_id']}/{method}: head invariant"
            )
        if float(record["generation_latency_ms"]) <= 0:
            problems.append(
                f"{benchmark}/{record['example_id']}/{method}: nonpositive latency"
            )
        if int(record["approximate_flops"]) <= 0:
            problems.append(
                f"{benchmark}/{record['example_id']}/{method}: nonpositive FLOPs"
            )
    if problems:
        raise ValueError("; ".join(problems[:20]))
    return {
        "passed": True,
        "records_checked": len(frame),
        "glr_transition_calls_expected": 10,
        "glr_vocabulary_calls_during_latent_updates": 0,
        "all_total_step_budgets_valid": True,
    }


def wilson_interval(correct: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [100 * (center - half), 100 * (center + half)]


def aggregate_metrics(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    output: dict[str, Any] = {}
    rows = []
    for benchmark in DATASETS:
        output[benchmark] = {}
        for method in METHODS:
            group = frame[
                (frame["benchmark"] == benchmark) & (frame["method"] == method)
            ]
            latency = group["generation_latency_ms"].to_numpy(float)
            steps = group["total_post_prompt_step_count"].to_numpy(float)
            text = group["generated_text_token_count"].to_numpy(float)
            flops = group["approximate_flops"].to_numpy(float) / 1e12
            ttft = group["time_to_first_text_token_ms"].to_numpy(float)
            correct = int(group["correct"].astype(bool).sum())
            metrics = {
                "n": len(group),
                "accuracy_percent": 100 * correct / len(group),
                "accuracy_wilson_ci95_percent": wilson_interval(correct, len(group)),
                "mean_total_steps": float(np.mean(steps)),
                "median_total_steps": float(np.median(steps)),
                "mean_text_tokens": float(np.mean(text)),
                "median_text_tokens": float(np.median(text)),
                "mean_latency_ms": float(np.mean(latency)),
                "median_latency_ms": float(np.median(latency)),
                "p90_latency_ms": float(np.percentile(latency, 90)),
                "p95_latency_ms": float(np.percentile(latency, 95)),
                "examples_per_second": float(len(group) / (np.sum(latency) / 1000)),
                "mean_tflops_per_example": float(np.mean(flops)),
                "median_tflops_per_example": float(np.median(flops)),
                "max_step_truncation_percent": float(
                    100 * group["max_step_truncation"].astype(bool).mean()
                ),
                "mean_ttft_ms": float(np.mean(ttft)),
                "median_ttft_ms": float(np.median(ttft)),
                "mean_vocabulary_head_calls": float(
                    group["vocabulary_head_call_count"].mean()
                ),
                "mean_transition_head_calls": float(
                    group["transition_head_call_count"].mean()
                ),
                "mean_backbone_calls": float(group["backbone_call_count"].mean()),
            }
            output[benchmark][method] = metrics
            rows.append({"benchmark": benchmark, "method": method, **metrics})
    return output, pd.DataFrame(rows)


def paired_arrays(
    frame: pd.DataFrame, benchmark: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    pivot = frame[frame["benchmark"] == benchmark].pivot(
        index="example_id", columns="method", values=field
    )
    if pivot.isna().any().any():
        raise ValueError(f"{benchmark}/{field}: incomplete pairing")
    return (
        pivot["cot_sft"].to_numpy(float),
        pivot["glr_k10"].to_numpy(float),
    )


def point_metrics(frame: pd.DataFrame, benchmark: str) -> dict[str, float]:
    cot_acc, glr_acc = paired_arrays(frame, benchmark, "correct")
    cot_latency, glr_latency = paired_arrays(frame, benchmark, "generation_latency_ms")
    cot_flops, glr_flops = paired_arrays(frame, benchmark, "approximate_flops")
    cot_steps, glr_steps = paired_arrays(
        frame, benchmark, "total_post_prompt_step_count"
    )
    cot_vocab, glr_vocab = paired_arrays(frame, benchmark, "vocabulary_head_call_count")
    return {
        "accuracy_difference_glr_minus_cot_pp": float(
            100 * (np.mean(glr_acc) - np.mean(cot_acc))
        ),
        "median_latency_speedup_cot_over_glr": float(
            np.median(cot_latency) / np.median(glr_latency)
        ),
        "mean_latency_speedup_cot_over_glr": float(
            np.mean(cot_latency) / np.mean(glr_latency)
        ),
        "mean_flop_reduction_percent": float(
            100 * (1 - np.mean(glr_flops) / np.mean(cot_flops))
        ),
        "median_total_step_difference_glr_minus_cot": float(
            np.median(glr_steps) - np.median(cot_steps)
        ),
        "mean_vocabulary_head_call_reduction_percent": float(
            100 * (1 - np.mean(glr_vocab) / np.mean(cot_vocab))
        ),
    }


def bootstrap(frame: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    output = {}
    for benchmark_index, benchmark in enumerate(DATASETS):
        fields = {
            name: paired_arrays(frame, benchmark, source)
            for name, source in {
                "accuracy": "correct",
                "latency": "generation_latency_ms",
                "flops": "approximate_flops",
                "steps": "total_post_prompt_step_count",
                "vocabulary_calls": "vocabulary_head_call_count",
            }.items()
        }
        n = len(fields["accuracy"][0])
        rng = np.random.default_rng(seed + benchmark_index)
        distributions = {
            name: np.empty(n_boot) for name in point_metrics(frame, benchmark)
        }
        for replicate in range(n_boot):
            indices = rng.integers(0, n, n)
            ca, ga = (value[indices] for value in fields["accuracy"])
            cl, gl = (value[indices] for value in fields["latency"])
            cf, gf = (value[indices] for value in fields["flops"])
            cs, gs = (value[indices] for value in fields["steps"])
            cv, gv = (value[indices] for value in fields["vocabulary_calls"])
            distributions["accuracy_difference_glr_minus_cot_pp"][replicate] = 100 * (
                np.mean(ga) - np.mean(ca)
            )
            distributions["median_latency_speedup_cot_over_glr"][replicate] = np.median(
                cl
            ) / np.median(gl)
            distributions["mean_latency_speedup_cot_over_glr"][replicate] = np.mean(
                cl
            ) / np.mean(gl)
            distributions["mean_flop_reduction_percent"][replicate] = 100 * (
                1 - np.mean(gf) / np.mean(cf)
            )
            distributions["median_total_step_difference_glr_minus_cot"][replicate] = (
                np.median(gs) - np.median(cs)
            )
            distributions["mean_vocabulary_head_call_reduction_percent"][replicate] = (
                100 * (1 - np.mean(gv) / np.mean(cv))
            )
        points = point_metrics(frame, benchmark)
        output[benchmark] = {
            name: {
                "estimate": points[name],
                "ci95": [
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                ],
            }
            for name, values in distributions.items()
        }
    return output


def correct_only_summary(frame: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for benchmark in DATASETS:
        output[benchmark] = {}
        for method in METHODS:
            group = frame[
                (frame["benchmark"] == benchmark)
                & (frame["method"] == method)
                & frame["correct"].astype(bool)
            ]
            output[benchmark][method] = {
                "n_correct": len(group),
                "median_latency_ms": (
                    float(group["generation_latency_ms"].median())
                    if len(group)
                    else None
                ),
                "median_total_steps": (
                    float(group["total_post_prompt_step_count"].median())
                    if len(group)
                    else None
                ),
            }
    return output


def efficiency_markdown(aggregate: dict[str, Any], intervals: dict[str, Any]) -> str:
    lines = [
        "| Benchmark | Method | Accuracy (%) | Median total steps | Median latency (s) | P95 latency (s) | Approx. mean TFLOPs/example | Speedup vs. CoT |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    display = {"gsm8k": "GSM8K", "svamp": "SVAMP", "math500": "MATH500"}
    for benchmark in DATASETS:
        for method in METHODS:
            metrics = aggregate[benchmark][method]
            if method == "cot_sft":
                speed = "1.00×"
            else:
                value = intervals[benchmark]["median_latency_speedup_cot_over_glr"]
                speed = (
                    f"{value['estimate']:.2f}× "
                    f"[{value['ci95'][0]:.2f}, {value['ci95'][1]:.2f}]"
                )
            lines.append(
                f"| {display[benchmark]} | {METHOD_LABELS[method]} | "
                f"{metrics['accuracy_percent']:.1f} | "
                f"{metrics['median_total_steps']:.0f} | "
                f"{metrics['median_latency_ms']/1000:.3f} | "
                f"{metrics['p95_latency_ms']/1000:.3f} | "
                f"{metrics['mean_tflops_per_example']:.2f} | {speed} |"
            )
    lines.extend(
        [
            "",
            "*Note:* Both methods use the same HuggingFace stack, BF16, FlashAttention-2, KV caching, greedy decoding, batch size 1, and one GPU per request. Each example's CoT-SFT/GLR pair is measured on the same GPU profile; if multiple profiles are used, absolute latencies pool the fixed profile mixture and paired confidence intervals resample intact method pairs. Latency includes prefill and all latent/text decoding through the stopping event, but excludes model loading, tokenization, detokenization, and answer scoring. GLR total steps count 10 latent steps plus generated text tokens. FLOPs are approximate dense-matmul FLOPs computed from actual instrumented backbone/head call traces. All-example metrics are reported.",
            "",
        ]
    )
    return "\n".join(lines)


def latex_rows(aggregate: dict[str, Any], intervals: dict[str, Any]) -> str:
    display = {"gsm8k": "GSM8K", "svamp": "SVAMP", "math500": "MATH500"}
    rows = []
    for benchmark in DATASETS:
        for method in METHODS:
            metrics = aggregate[benchmark][method]
            if method == "cot_sft":
                speed = r"$1.00\times$"
            else:
                value = intervals[benchmark]["median_latency_speedup_cot_over_glr"]
                speed = (
                    rf"${value['estimate']:.2f}\times$ "
                    rf"$[{value['ci95'][0]:.2f},{value['ci95'][1]:.2f}]$"
                )
            rows.append(
                f"{display[benchmark]} & {METHOD_LABELS[method]} & "
                f"{metrics['accuracy_percent']:.1f} & "
                f"{metrics['median_total_steps']:.0f} & "
                f"{metrics['median_latency_ms']/1000:.3f} & "
                f"{metrics['p95_latency_ms']/1000:.3f} & "
                f"{metrics['mean_tflops_per_example']:.2f} & {speed} \\\\"
            )
    return "\n".join(rows)


def latex_table(
    aggregate: dict[str, Any], intervals: dict[str, Any], with_caption: bool
) -> str:
    caption = (
        "Matched-stack inference efficiency on Qwen3-1.7B. CoT-SFT and "
        "GLR-10 use the same HuggingFace BF16 implementation, KV caching, "
        "greedy decoding, and batch size 1. Latency includes prefill and all "
        "subsequent model steps. Approximate FLOPs count dense matrix "
        "multiplications from instrumented call traces."
    )
    prefix = "\\begin{table*}[t]\n\\centering\n" + (
        f"\\caption{{{caption}}}\n\\label{{tab:matched-efficiency}}\n" if with_caption else ""
    )
    note = (
        "\\\\[-1mm]\\multicolumn{8}{p{0.97\\linewidth}}{\\footnotesize "
        "Timing excludes loading, tokenization, detokenization, and scoring; "
        "each paired CoT-SFT/GLR example uses the same GPU profile. When "
        "multiple profiles are used, absolute latencies pool the fixed profile "
        "mixture and confidence intervals retain pairing. GLR steps include "
        "ten latent updates. FLOPs are approximate dense-matmul counts from "
        "actual backbone/head traces. All examples are used.}"
    )
    return (
        prefix + "\\begin{tabular}{llrrrrrl}\n\\toprule\n"
        "Benchmark & Method & Acc. (\\%) & Med. steps & Med. lat. (s) & "
        "P95 lat. (s) & Mean TFLOPs & Speedup \\\\\n\\midrule\n"
        + latex_rows(aggregate, intervals)
        + "\n\\midrule\n"
        + note
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def generate_figure(
    frame: pd.DataFrame,
    intervals: dict[str, Any],
    output_dir: Path,
) -> None:
    plot_rows = frame[
        ["benchmark", "example_id", "method", "generation_latency_ms"]
    ].copy()
    plot_rows["generation_latency_s"] = plot_rows["generation_latency_ms"] / 1000
    plot_rows.to_csv(
        output_dir / "figures" / "efficiency_latency_data.csv", index=False
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    display = {"gsm8k": "GSM8K", "svamp": "SVAMP", "math500": "MATH500"}
    for axis, benchmark in zip(axes, DATASETS):
        for method in METHODS:
            values = np.sort(
                plot_rows[
                    (plot_rows["benchmark"] == benchmark)
                    & (plot_rows["method"] == method)
                ]["generation_latency_s"].to_numpy(float)
            )
            y = np.arange(1, len(values) + 1) / len(values)
            axis.step(
                values,
                y,
                where="post",
                label=METHOD_LABELS[method],
                color=COLORS[method],
                linewidth=1.8,
            )
        speed = intervals[benchmark]["median_latency_speedup_cot_over_glr"]
        axis.text(
            0.04,
            0.96,
            f"median speedup {speed['estimate']:.2f}×\n"
            f"95% CI [{speed['ci95'][0]:.2f}, {speed['ci95'][1]:.2f}]",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        axis.set_title(display[benchmark])
        axis.set_xscale("log")
        axis.set_xlabel("Generation latency (s, log scale)")
        axis.grid(True, alpha=0.2)
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(
        output_dir / "figures" / "efficiency_latency.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "figures" / "efficiency_latency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_output_tree(output_dir)
    static = load_static_manifest(output_dir)
    frame, completeness = load_cache(output_dir)
    atomic_write_json(
        output_dir / "summaries" / "cache_completeness.json", completeness
    )
    if not completeness["complete"]:
        atomic_write_json(
            output_dir / "logs" / "analysis_failure.json",
            {
                "status": "incomplete",
                "message": "Primary cache is incomplete; no claims table generated.",
                "cache_completeness": completeness,
            },
        )
        raise RuntimeError("Efficiency cache incomplete; see cache_completeness.json")

    head_validation = validate_records(frame, static)
    atomic_write_json(
        output_dir / "logs" / "head_call_validation.json", head_validation
    )
    selected_columns = [
        "benchmark",
        "example_id",
        "dataset_index",
        "method",
        "correct",
        "latent_step_count",
        "generated_text_token_count",
        "total_post_prompt_step_count",
        "generation_latency_ms",
        "time_to_first_text_token_ms",
        "approximate_flops",
        "vocabulary_head_call_count",
        "transition_head_call_count",
        "backbone_call_count",
        "stop_reason",
        "max_step_truncation",
        "phase",
        "run_id",
        "timing_hash",
    ]
    portable_frame = frame[selected_columns].copy()
    portable_frame.to_parquet(
        output_dir / "summaries" / "per_example.parquet", index=False
    )
    portable_frame.to_csv(output_dir / "summaries" / "per_example.csv", index=False)
    aggregate, aggregate_frame = aggregate_metrics(frame)
    intervals = bootstrap(frame, args.bootstrap_replicates, args.analysis_seed)
    relative = {benchmark: point_metrics(frame, benchmark) for benchmark in DATASETS}
    payload = {
        "metadata": {
            "analysis_seed": args.analysis_seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "all_examples_primary": True,
        },
        "aggregate": aggregate,
        "relative_glr_vs_cot": relative,
        "correct_only_secondary": correct_only_summary(frame),
    }
    atomic_write_json(output_dir / "summaries" / "aggregate_metrics.json", payload)
    aggregate_frame.to_csv(
        output_dir / "summaries" / "aggregate_metrics.csv", index=False
    )
    atomic_write_json(output_dir / "summaries" / "bootstrap_intervals.json", intervals)
    environments = []
    for path in sorted((output_dir / "manifests").glob("environment_shard*.json")):
        environments.append(json.loads(path.read_text(encoding="utf-8")))
    atomic_write_json(
        output_dir / "summaries" / "environment.json",
        {"shards": environments},
    )

    markdown = efficiency_markdown(aggregate, intervals)
    (output_dir / "tables" / "efficiency_table.md").write_text(
        markdown, encoding="utf-8"
    )
    (output_dir / "tables" / "efficiency_table.tex").write_text(
        latex_table(aggregate, intervals, with_caption=True), encoding="utf-8"
    )
    generate_figure(frame, intervals, output_dir)
    trace_rows = []
    for record in frame.to_dict(orient="records"):
        trace_rows.append(
            {
                "benchmark": record["benchmark"],
                "example_id": record["example_id"],
                "method": record["method"],
                "backbone_call_count": record["backbone_call_count"],
                "vocabulary_head_call_count": record["vocabulary_head_call_count"],
                "transition_head_call_count": record["transition_head_call_count"],
                "query_context_trace": [
                    [item["query_length"], item["context_length"]]
                    for item in record["backbone_call_trace"]
                ],
            }
        )
    trace_path = output_dir / "traces" / "call_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    combined_log = output_dir / "logs" / "run.log"
    with combined_log.open("w", encoding="utf-8") as handle:
        handle.write("Matched-stack efficiency run log assembled from shard logs.\n")
        for path in sorted((output_dir / "logs").glob("run_shard*.log")):
            handle.write(f"\n===== {path.name} =====\n")
            handle.write(path.read_text(encoding="utf-8"))
        handle.write("\n===== analysis =====\n")
        handle.write(
            f"cache_complete=true records={len(frame)} "
            f"bootstrap_replicates={args.bootstrap_replicates} "
            f"analysis_seed={args.analysis_seed}\n"
        )

    benchmark_names = ", ".join(DATASETS)
    result_readme = """# Qwen3-1.7B matched-stack efficiency results

This directory contains resumable, per-example matched-HuggingFace latency and
approximate dense-matmul FLOP measurements for CoT-SFT and GLR-10 on the
complete selected benchmarks: {benchmarks}.

The primary timing is synchronized wall-clock time from prefill through the
stopping event. It excludes model loading, tokenization, detokenization,
answer parsing/scoring, cache writes, and analysis. GLR total steps equal ten
latent steps plus generated text tokens. FLOPs omit lower-order operations and
memory movement and are computed from the instrumented backbone/head traces.

From the repository root:

```bash
# Run/resume all five GPU shards, then validate, analyze, and render.
bash analysis/latency/run_latency.sh

# Regenerate all summaries, tables, figures, and output validation without
# importing or loading either model checkpoint.
STAGE=analysis bash analysis/latency/run_latency.sh
```

The launcher runs five independent, balanced shards and never enables
`--force`; valid records are reused automatically.
""".format(benchmarks=benchmark_names)
    (output_dir / "README.md").write_text(result_readme, encoding="utf-8")
    print(output_dir / "tables" / "efficiency_table.md")


if __name__ == "__main__":
    main()
