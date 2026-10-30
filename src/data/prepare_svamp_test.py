# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Prepare SVAMP (test split) into this repo's JSONL decoding format.

Dataset: https://huggingface.co/datasets/ChilleD/SVAMP

Output format (the same for all eval sets):
  {"id": ..., "q": ..., "a": ..., "target": ..., "problem": ..., "solution": ...}

Run:
  bash scripts/data/prepare_eval_sets.sh
"""

import json
from pathlib import Path

from glr_paths import get

REVISION = "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"


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

    ds = load_dataset("ChilleD/SVAMP", revision=REVISION)
    if "test" not in ds:
        raise SystemExit(f"SVAMP: expected 'test' split, found: {list(ds.keys())}")

    out_path = Path(get("eval_dir")) / "svamp.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds["test"]):
            rid = row.get("ID", idx)

            problem = row.get("question_concat")
            if problem is None or str(problem).strip() == "":
                body = str(row.get("Body", "")).strip()
                question = str(row.get("Question", "")).strip()
                problem = (body + "\n" + question).strip()
            else:
                problem = str(problem).strip()

            ans_str = str(row.get("Answer", "")).strip()
            equation = str(row.get("Equation", "")).strip()

            rec = {
                "id": rid,
                "q": _build_q(problem),
                "a": ans_str,
                "target": ans_str,
                "problem": problem,
                "solution": equation,
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1

    print(f"wrote {n} examples -> {out_path}")


if __name__ == "__main__":
    main()
