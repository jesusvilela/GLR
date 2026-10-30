# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Prepare AMC 2023 (test split) into this repo's JSONL decoding format.

Dataset: https://huggingface.co/datasets/math-ai/amc23

Output format (the same for all eval sets):
  {"id": ..., "q": ..., "a": ..., "target": ..., "problem": ..., "solution": ...}

Run:
  bash scripts/data/prepare_eval_sets.sh
"""

import json
from pathlib import Path

from glr_paths import get

REVISION = "80815d37005feb82cd7f8fbc6901d5d3eff43057"


def _build_q(problem: str) -> str:
    problem = problem.strip("\n")
    return (
        "<|im_start|>user\n"
        f"Question: {problem}\n"
        "Answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("math-ai/amc23", revision=REVISION)
    if "test" not in ds:
        raise SystemExit(f"amc23: expected 'test' split, found: {list(ds.keys())}")

    out_path = Path(get("eval_dir")) / "amc23.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds["test"]):
            rid = row.get("id", idx)
            problem = str(row.get("question", "")).strip()
            if not problem:
                continue

            ans_str = str(row.get("answer", "")).strip()
            url = str(row.get("url", "")).strip()

            rec = {
                "id": rid,
                "q": _build_q(problem),
                "a": ans_str,
                "target": ans_str,
                "problem": problem,
                "solution": url,
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1

    print(f"wrote {n} examples -> {out_path}")


if __name__ == "__main__":
    main()
