# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Prepare MATH500 (test split) into this repo's JSONL decoding format.

The prompt is the zero-shot Minerva prompt of lm-evaluation-harness
(`lm_eval_tasks/minerva_math/minerva_math500_zeroshot.yaml`) in the Qwen3 chat format. The target is
the last boxed answer of the reference solution, normalized as in lm-evaluation-harness. Output format:
  {"id": ..., "q": ..., "a": ..., "target": ..., "problem": ..., "solution": ...}

Run:
  bash scripts/data/prepare_eval_sets.sh
"""

import json
from pathlib import Path

from glr_paths import get
from src.metric.minerva_math500_score import _last_boxed_only_string, _remove_boxed, normalize_final_answer

REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def _build_q(problem: str) -> str:
    return (
        "<|im_start|>user\n"
        f"Problem:\n{problem}\n\nSolution:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=REVISION)

    out_path = Path(get("eval_dir")) / "minerva_math500_zeroshot.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            target = normalize_final_answer(_remove_boxed(_last_boxed_only_string(row["solution"])))
            rec = {
                "id": idx,
                "q": _build_q(row["problem"]),
                "a": target,
                "target": target,
                "problem": row["problem"],
                "solution": "",
            }
            f.write(json.dumps(rec))
            f.write("\n")

    print(f"wrote {len(ds)} examples -> {out_path}")


if __name__ == "__main__":
    main()
