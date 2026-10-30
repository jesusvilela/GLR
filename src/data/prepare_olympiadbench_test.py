# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Prepare OlympiadBench (English configs) into this repo's JSONL decoding format.

Dataset: https://huggingface.co/datasets/Hothan/OlympiadBench

Important:
    As of 2026-05-03, the OlympiadBench configs on HF expose only a `train` split
    via `datasets.load_dataset(...)` (no `test`).

This script therefore dumps the `train` split.

By default it dumps *all* English configs ("partitions") into separate JSONL
files in eval_dir.

Output format (the same for all eval sets):
  {"id": ..., "q": ..., "a": ..., "target": ..., "problem": ..., "solution": ...}
"""

import argparse
import json
import re
from pathlib import Path

from glr_paths import get

REVISION = "91184b52131e7fc9455fef848035173aea8cc01a"


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _build_q(problem: str) -> str:
    problem = problem.strip("\n")
    return (
        "<|im_start|>user\n"
        f"Question: {problem}\n"
        "Answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _pick_answer_text(row: dict) -> str:
    """Extract a single gold answer string.

    OlympiadBench commonly stores `final_answer` as a list (sometimes None).
    """
    final_answer = row.get("final_answer")
    if isinstance(final_answer, list):
        if len(final_answer) == 0:
            return ""
        if len(final_answer) == 1:
            return _as_text(final_answer[0]).strip()
        return json.dumps(final_answer, ensure_ascii=False)
    if final_answer in (None, ""):
        return ""
    return _as_text(final_answer).strip()


def _pick_solution_text(row: dict) -> str:
    sol = row.get("solution")
    if isinstance(sol, list):
        parts = [str(s).strip() for s in sol if s not in (None, "")]
        return "\n\n".join([p for p in parts if p])
    return _as_text(sol).strip()


def _build_problem_text(row: dict) -> str:
    question = _as_text(row.get("question")).strip()
    context = row.get("context")
    context_text = _as_text(context).strip()
    if context_text:
        if question:
            return f"{context_text}\n\n{question}"
        return context_text
    return question


def _iter_english_configs() -> list[str]:
    from datasets import get_dataset_config_names

    configs = get_dataset_config_names("Hothan/OlympiadBench")
    return [c for c in configs if "_en_" in c]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dump OlympiadBench (English configs) to repo JSONL format"
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dump only this single HF config (default: dump all English configs)",
    )
    p.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated list of HF config names to dump (overrides --config)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to dump (default: train)",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    from datasets import load_dataset

    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    elif args.config:
        configs = [args.config]
    else:
        configs = _iter_english_configs()

    if not configs:
        raise SystemExit("No configs selected (unexpected).")

    # Only the English subsets are used.
    non_en = [c for c in configs if "_en_" not in c]
    if non_en:
        raise SystemExit(f"Non-English configs requested but disallowed: {non_en}")

    out_dir = Path(get("eval_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for cfg in configs:
        ds = load_dataset("Hothan/OlympiadBench", cfg, revision=REVISION)
        if args.split not in ds:
            raise SystemExit(
                f"OlympiadBench config '{cfg}' does not contain split '{args.split}'. "
                f"Available splits: {list(ds.keys())}."
            )

        out_name = f"olympiadbench_{_slug(cfg)}_{_slug(args.split)}.jsonl"
        out_path = out_dir / out_name

        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(ds[args.split]):
                # `row` is dict-like.
                row_id = row.get("id", idx)
                problem = _build_problem_text(row).strip()
                if not problem:
                    continue

                ans_str = _pick_answer_text(row)
                solution_text = _pick_solution_text(row)

                rec = {
                    "id": f"{cfg}:{row_id}",
                    "q": _build_q(problem),
                    "a": ans_str,
                    "target": ans_str,
                    "problem": problem,
                    "solution": solution_text,
                }
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
                n += 1

        total_written += n
        print(f"wrote {n} examples -> {out_path}")

    print(f"done: wrote {total_written} total examples across {len(configs)} configs")


if __name__ == "__main__":
    main()
