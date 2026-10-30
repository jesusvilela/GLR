# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""K=0 ablation: generated steps of the GLR Qwen3-1.7B model with K = 0, 5, and 10 latent steps.

Reads scored_dir/glr_qwen3-1.7B_k0_ablation (K=0) and scored_dir/glr_qwen3-1.7B (K=5, 10), and writes to
figure_dir, per benchmark (GSM8K, MATH500):
  qwen3-1.7B_glr0_5_10_<benchmark>_boxplot_{correct,all}.{pdf,png}
  qwen3-1.7B_glr0_5_10_<benchmark>_table.csv

Usage: python plots/plot_k0_ablation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

from glr_paths import get
from plots.plot_common import DIST_FIGSIZE, Variant, _box_scatter, _lighten, _save, log_y_axis, parse_variant, plt

GLR_KS = [0, 5, 10]
BENCHMARKS = {
    "gsm8k_zeroshot": dict(title="GSM8K", stem="gsm8k", cap=2048, correct="correct",
                           yticks=[50, 100, 200, 500, 1000, 2048], ylim=(45, 2200)),
    "minerva_math500_zeroshot": dict(title="MATH500", stem="math500", cap=4096, correct="math_verify_correct",
                                     yticks=[50, 100, 200, 500, 1000, 2000, 4096], ylim=(45, 4500)),
}

_pastel = sns.color_palette("pastel", n_colors=6)
GLR_COLOR = {
    0: _lighten(_pastel[1], 0.55),
    5: _pastel[1],
    10: _pastel[2],
}


def main() -> None:
    scored_dir = Path(get("scored_dir"))
    fig_dir = Path(get("figure_dir"))
    tokenizer = AutoTokenizer.from_pretrained(get("qwen3_1.7b"), use_fast=True)

    for benchmark, cfg in BENCHMARKS.items():
        paths = {0: scored_dir / "glr_qwen3-1.7B_k0_ablation" / f"{benchmark}_latent0.jsonl"}
        paths.update({k: scored_dir / "glr_qwen3-1.7B" / f"{benchmark}_latent{k}.jsonl" for k in GLR_KS if k})
        df = pd.concat(
            [parse_variant(Variant(f"GLR-{k}", paths[k], "predicted", cfg["correct"]), tokenizer) for k in GLR_KS],
            ignore_index=True,
        )
        df["token_count_capped"] = df["token_count"].clip(upper=cfg["cap"])
        order = [f"GLR-{k}" for k in GLR_KS]
        palette = {f"GLR-{k}": GLR_COLOR[k] for k in GLR_KS}

        summary_all = df.groupby("variant", sort=False).agg(
            n_total=("token_count", "size"),
            n_correct=("correct", "sum"),
            acc=("correct", "mean"),
            tokens_median_capped=("token_count_capped", "median"),
        )
        summary_correct = df[df["correct"]].groupby("variant", sort=False).agg(
            tokens_median_correct_capped=("token_count_capped", "median"),
        )
        table = summary_all.join(summary_correct)
        table["acc"] = (100 * table["acc"]).round(1)
        stem = f"qwen3-1.7B_glr0_5_10_{cfg['stem']}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(fig_dir / f"{stem}_table.csv")
        print(f"\n{cfg['title']} — Qwen3-1.7B GLR, K = 0 / 5 / 10 (steps capped at {cfg['cap']})")
        print(table)

        for correct_only in [True, False]:
            plot_df = df[df["correct"]] if correct_only else df
            answers = "correct answers" if correct_only else "all answers"
            fig, ax = plt.subplots(figsize=DIST_FIGSIZE)
            _box_scatter(ax, plot_df, order, palette, style="bold", scatter_alpha=0.35)
            ax.set_title(f"{cfg['title']}: Generated tokens for {answers}")
            ax.set_xlabel("")
            ax.set_ylabel("Generated tokens")
            log_y_axis(ax, cfg["yticks"], cfg["ylim"])
            fig.tight_layout()
            print("Saved", *_save(fig, fig_dir, f"{stem}_boxplot_{'correct' if correct_only else 'all'}"))


if __name__ == "__main__":
    main()
