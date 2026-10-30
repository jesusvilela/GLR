# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Shared plotting style and helpers for the paper figures."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PLOT_TEXT_PARAMS = {
    "axes.titlesize": 16.0,
    "axes.labelsize": 18.0,
    "legend.fontsize": 14.0,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
}

# Qwen3-1.7B panels have more variants, so the legend and x tick labels are smaller.
PLOT_TEXT_PARAMS_local = {
    "axes.titlesize": 16.0,
    "axes.labelsize": 18.0,
    "legend.fontsize": 12.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 14.0,
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
mpl.rcParams.update(
    {
        # paper-ready typography: serif fonts, embedded TrueType text, and
        # figures sized close to their final paper width so labels are not shrunk.
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        **PLOT_TEXT_PARAMS,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.9,
        "grid.linewidth": 0.55,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.8,
    }
)

SINGLE_FIGSIZE = (6.6, 3.6)
DIST_FIGSIZE = (7.2, 3.4)
COMBINED_FIGSIZE = (
    0.85 * 0.93 * (SINGLE_FIGSIZE[0] + DIST_FIGSIZE[0]),
    max(SINGLE_FIGSIZE[1], DIST_FIGSIZE[1]),
)

DELIM = "<|im_end|>\n<|im_start|>assistant\n"
EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path} line {line_no}: {e}") from e


def extract_final_assistant_turn(text: str | None) -> str:
    """Extract the last assistant turn from a chat-formatted transcript.

    Split on '<|im_end|>\n<|im_start|>assistant\n' and take the last segment.
    """
    if not text:
        return ""

    if DELIM in text:
        out = text.split(DELIM)[-1]
    elif "<|im_start|>assistant\n" in text:
        out = text.split("<|im_start|>assistant\n")[-1]
    else:
        out = text

    # Strip known wrapper tokens
    out = out.replace("<|im_start|>assistant", "")
    return out.strip()


def count_tokens(tokenizer, text: str) -> int:
    # Count tokens in the extracted assistant turn (no extra special tokens).
    # For GLR, the K latent positions are part of the decoded text and are counted as steps.
    return len(tokenizer.encode(text, add_special_tokens=False))


def get_bool(d: dict[str, Any], *keys: str) -> bool | None:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
    return None


@dataclass(frozen=True)
class Variant:
    label: str
    path: Path
    text_field: str
    correct_field: str


def parse_variant(v: Variant, tokenizer) -> pd.DataFrame:
    """One row per example: correctness and number of generated steps after the prompt."""
    rows: list[dict[str, Any]] = []
    for rec in read_jsonl(v.path):
        raw_text = rec.get(v.text_field)
        assistant_text = extract_final_assistant_turn(
            raw_text if isinstance(raw_text, str) else None
        )
        tok = count_tokens(tokenizer, assistant_text)
        correct = get_bool(rec, v.correct_field)
        if correct is None:
            raise KeyError(f"Missing {v.correct_field} for {v.label} in {v.path}")
        # A generation cut off at the decoding limit did not finish, so it is not counted as correct.
        finished = isinstance(raw_text, str) and raw_text.endswith(EOS_TOKENS)
        rows.append(
            {"variant": v.label, "correct": bool(correct) and finished, "token_count": int(tok)}
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No rows parsed for variant {v.label} from {v.path}")
    return df


def budgeted_accuracy(df: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    """Accuracy when only answers finished within `budget` generated steps count as correct."""
    out = []
    for b in budgets:
        out.append(
            {
                "budget": b,
                "acc": float(((df["correct"]) & (df["token_count"] <= b)).mean()),
            }
        )
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary_all = (
        df.groupby("variant", sort=False)
        .agg(
            n_total=("token_count", "size"),
            n_correct=("correct", "sum"),
            acc=("correct", "mean"),
            tokens_mean=("token_count", "mean"),
            tokens_median=("token_count", "median"),
            tokens_p10=("token_count", lambda s: float(s.quantile(0.10))),
            tokens_p90=("token_count", lambda s: float(s.quantile(0.90))),
        )
        .reset_index()
    )
    summary_correct = (
        df[df["correct"]]
        .groupby("variant", sort=False)
        .agg(
            tokens_mean_correct=("token_count", "mean"),
            tokens_median_correct=("token_count", "median"),
            tokens_p10_correct=("token_count", lambda s: float(s.quantile(0.10))),
            tokens_p90_correct=("token_count", lambda s: float(s.quantile(0.90))),
        )
        .reset_index()
    )
    return summary_all.merge(summary_correct, on="variant", how="left")


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="0.90", linewidth=0.7)
    ax.grid(False, axis="x")
    sns.despine(ax=ax, top=True, right=True)


def _rasterize_points(ax: plt.Axes) -> None:
    for coll in ax.collections:
        coll.set_rasterized(True)


# Box-plot styles: "main" for the GSM8K/MATH500 figures, "bold" (thicker median, larger points)
# for the other benchmarks and the K=0 ablation.
BOX_STYLES = {
    "main": {
        "median_color": "0.20",
        "median_linewidth": 1.2,
        "point_size": 1.4,
        "scatter_alpha": 0.22,
    },
    "bold": {
        "median_color": "0.10",
        "median_linewidth": 2.0,
        "point_size": 2.5,
        "scatter_alpha": 0.40,
    },
}


def _box_scatter(
    ax: plt.Axes,
    d: pd.DataFrame,
    order: list[str],
    palette_map: dict[str, tuple[float, float, float]],
    style: str = "main",
    x: str = "variant",
    scatter_alpha: float | None = None,
) -> None:
    s = BOX_STYLES[style]
    token_col = (
        "token_count_capped" if "token_count_capped" in d.columns else "token_count"
    )
    sns.boxplot(
        data=d,
        x=x,
        y=token_col,
        order=order,
        hue=x,
        palette=palette_map,
        dodge=False,
        width=0.48,
        showcaps=True,
        showfliers=False,
        boxprops={"edgecolor": "0.25", "linewidth": 0.9},
        medianprops={"color": s["median_color"], "linewidth": s["median_linewidth"]},
        whiskerprops={"color": "0.35", "linewidth": 0.9},
        capprops={"color": "0.35", "linewidth": 0.9},
        ax=ax,
        zorder=2,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    sns.stripplot(
        data=d,
        x=x,
        y=token_col,
        order=order,
        hue=x,
        palette=palette_map,
        dodge=False,
        jitter=0.26,
        size=s["point_size"],
        alpha=s["scatter_alpha"] if scatter_alpha is None else scatter_alpha,
        linewidth=0,
        ax=ax,
        zorder=1,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    _rasterize_points(ax)
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=0)
    _style_axes(ax)


def _save(fig: plt.Figure, fig_dir: Path, stem: str) -> tuple[Path, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"{stem}.pdf"
    out_png = fig_dir / f"{stem}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_pdf, out_png


def _lighten(color: tuple, factor: float = 0.50) -> tuple:
    """Blend a colour toward white by `factor` (0 = unchanged, 1 = white)."""
    r, g, b = color[:3]
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def log_y_axis(ax: plt.Axes, yticks: list[int], ylim: tuple[float, float]) -> None:
    ax.set_yscale("log")
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_ylim(*ylim)
