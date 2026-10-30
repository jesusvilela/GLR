# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Accuracy vs. generation budget and generation-length distributions for one model on one benchmark.

Reads the scored generations of CoT-SFT and of every GLR-K run of the model, and writes to figure_dir:
  <benchmark>_qwen3-<model>.{pdf,png}              accuracy vs. budget | generated steps for correct answers
  <benchmark>_qwen3-<model>_all_answers.{pdf,png}  generated steps for all answers
  <benchmark>_qwen3-<model>_budget_table.csv       accuracy (%) per budget and median generated steps

Usage: python plots/plot_benchmark.py --model 1.7B --benchmark gsm8k_zeroshot
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

from glr_paths import get
from plots.plot_common import (
    COMBINED_FIGSIZE,
    DIST_FIGSIZE,
    PLOT_TEXT_PARAMS,
    PLOT_TEXT_PARAMS_local,
    SINGLE_FIGSIZE,
    Variant,
    _box_scatter,
    _save,
    budgeted_accuracy,
    log_y_axis,
    parse_variant,
    plt,
    summarize,
)

BUDGETS_2048 = [64, 128, 256, 512, 1024, 2048]
BUDGETS_4096 = [64, 128, 256, 512, 1024, 2048, 4096]
YTICKS_2048 = [50, 100, 200, 500, 1000, 2048]
YTICKS_4096 = [50, 100, 200, 500, 1000, 2000, 4096]

# Per-benchmark settings. Correctness: GSM8K uses the exact-match `correct` field; the other benchmarks use
# Math-Verify, except SVAMP CoT-SFT, which uses exact match.
BENCHMARKS = {
    "gsm8k_zeroshot": dict(
        title="GSM8K", stem="gsm8k", budgets=BUDGETS_2048, cap=2048, yticks=YTICKS_2048, ylim=(40, 2048),
        glr_correct="correct", sft_correct="correct", box_style="main",
    ),
    "minerva_math500_zeroshot": dict(
        title="MATH500", stem="math500", budgets=[128, 256, 512, 1024, 2048, 4096], cap=4096, yticks=YTICKS_4096,
        ylim=(40, 4096), glr_correct="math_verify_correct", sft_correct="math_verify_correct", box_style="main",
    ),
    "svamp": dict(
        title="SVAMP", stem="svamp", budgets=BUDGETS_2048, cap=2048, yticks=YTICKS_2048, ylim=(40, 2048 * 1.08),
        glr_correct="math_verify_correct", sft_correct="exact_match_correct", box_style="bold",
    ),
    "multiarith": dict(
        title="MultiArith", stem="multiarith", budgets=BUDGETS_2048, cap=2048, yticks=YTICKS_2048,
        ylim=(40, 2048 * 1.08), glr_correct="math_verify_correct", sft_correct="math_verify_correct", box_style="bold",
    ),
    "amc23": dict(
        title="AMC23", stem="amc23", budgets=BUDGETS_4096, cap=4096, yticks=YTICKS_4096, ylim=(40, 4096 * 1.08),
        glr_correct="math_verify_correct", sft_correct="math_verify_correct", box_style="bold",
    ),
    "olympiadbench_oe_to_maths_en_comp_train": dict(
        title="OlympiadBench", stem="olympiadbench", budgets=BUDGETS_4096, cap=4096, yticks=YTICKS_4096,
        ylim=(40, 4096 * 1.08), glr_correct="math_verify_correct", sft_correct="math_verify_correct", box_style="bold",
    ),
}
MODELS = {"0.6B": ("qwen3_0.6b", PLOT_TEXT_PARAMS), "1.7B": ("qwen3_1.7b", PLOT_TEXT_PARAMS_local)}


def latent_k(p: Path) -> int:
    return int(re.search(r"_latent(\d+)\.jsonl$", p.name).group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    args = parser.parse_args()
    cfg = BENCHMARKS[args.benchmark]
    model_key, text_params = MODELS[args.model]
    model_name = f"Qwen3-{args.model}"

    scored_dir = Path(get("scored_dir"))
    fig_dir = Path(get("figure_dir"))
    tokenizer = AutoTokenizer.from_pretrained(get(model_key), use_fast=True)

    # CoT-SFT baseline and every GLR-K run of this model (sorted by K).
    variants = [Variant("CoT-SFT", scored_dir / f"cot_sft_qwen3-{args.model}" / f"{args.benchmark}.jsonl",
                        "decoded_output", cfg["sft_correct"])]
    glr_files = sorted((scored_dir / f"glr_qwen3-{args.model}").glob(f"{args.benchmark}_latent*.jsonl"), key=latent_k)
    variants += [Variant(f"GLR-{latent_k(p)}", p, "predicted", cfg["glr_correct"]) for p in glr_files]

    df = pd.concat([parse_variant(v, tokenizer) for v in variants], ignore_index=True)
    order = [v.label for v in variants]
    palette = dict(zip(order, sns.color_palette("pastel", n_colors=len(order))))
    palette["CoT-SFT"] = (0.60, 0.60, 0.60)  # medium gray baseline

    # Tables: budgeted accuracy (%) and generation-length summary.
    acc_df = pd.concat(
        [budgeted_accuracy(df[df["variant"] == v], cfg["budgets"]).assign(variant=v) for v in order], ignore_index=True
    )
    acc_table = (
        acc_df.assign(acc_pct=lambda d: (100 * d["acc"]).round(1))
        .pivot(index="variant", columns="budget", values="acc_pct")
        .reindex(order)
    )
    summary = summarize(df).set_index("variant").reindex(order)
    table = acc_table.join(summary[["n_total", "n_correct", "acc", "tokens_median", "tokens_median_correct"]])
    table["acc"] = (100 * table["acc"]).round(1)

    stem = f"{cfg['stem']}_qwen3-{args.model}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(fig_dir / f"{stem}_budget_table.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(f"\n{cfg['title']} — {model_name}: accuracy (%) per generation budget, median generated steps")
        print(table)

    df["token_count_capped"] = df["token_count"].clip(upper=cfg["cap"])

    # Figure: accuracy vs. budget (left) and generated steps for correct answers (right).
    with mpl.rc_context(text_params):
        fig = plt.figure(figsize=COMBINED_FIGSIZE)
        gs = fig.add_gridspec(
            1, 4, width_ratios=[SINGLE_FIGSIZE[0] / 2, SINGLE_FIGSIZE[0] / 2, DIST_FIGSIZE[0] / 2, DIST_FIGSIZE[0] / 2]
        )
        ax_acc = fig.add_subplot(gs[0, 0:2])
        ax_dist = fig.add_subplot(gs[0, 2:4])

        for variant in order:
            acc_variant = acc_df[acc_df["variant"] == variant].sort_values("budget")
            is_baseline = variant == "CoT-SFT"
            ax_acc.plot(
                acc_variant["budget"],
                acc_variant["acc"],
                label=variant,
                color=palette[variant],
                linestyle=(0, (4, 2)) if is_baseline else "-",
                marker="o",
                linewidth=2.5 if is_baseline else 2.2,
                markersize=5.2,
                zorder=2 if is_baseline else 3,
            )
        ax_acc.set_xscale("log", base=2)
        ax_acc.set_xticks(cfg["budgets"])
        ax_acc.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax_acc.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x*100:.0f}"))
        ax_acc.set_xlabel("Generation length budget (tokens)")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.set_title(f"Accuracy vs. generation length budget ({cfg['title']})")
        ax_acc.legend(title="", loc="upper left", frameon=True, framealpha=0.95, handlelength=2.5, borderpad=0.35, labelspacing=0.32)
        sns.despine(ax=ax_acc, top=True, right=True)

        _box_scatter(ax_dist, df[df["correct"]], order, palette, style=cfg["box_style"])
        log_y_axis(ax_dist, cfg["yticks"], cfg["ylim"])
        ax_dist.set_ylabel("Generated tokens")
        ax_dist.set_xticks(np.arange(len(order)))
        ax_dist.set_xticklabels(order)
        ax_dist.set_title(f"Generated tokens for correct answers ({cfg['title']})")

        ax_acc.text(
            -0.22, 0.5, model_name, transform=ax_acc.transAxes, rotation=90, ha="center", va="center",
            fontsize=mpl.rcParams["axes.labelsize"] + 4, fontweight="bold",
        )
        fig.tight_layout(w_pad=1.0)
        print("Saved", *_save(fig, fig_dir, stem))

    # Figure: generated steps for all answers (generations longer than the cap are shown at the cap).
    with mpl.rc_context(text_params):
        fig, ax = plt.subplots(figsize=DIST_FIGSIZE)
        _box_scatter(ax, df, order, palette, style=cfg["box_style"])
        log_y_axis(ax, cfg["yticks"], cfg["ylim"])
        ax.set_ylabel("Generated tokens")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order)
        ax.set_title(f"Generated tokens for all answers ({cfg['title']}, {model_name})")
        fig.tight_layout()
        print("Saved", *_save(fig, fig_dir, f"{stem}_all_answers"))


if __name__ == "__main__":
    main()
