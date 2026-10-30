# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Prepare GSM8K (main config, test split) into this repo's JSONL decoding format.

The prompt is the zero-shot lm-evaluation-harness prompt (`lm_eval_tasks/gsm8k/gsm8k_zeroshot.yaml`)
in the Qwen3 chat format. Output format:
  {"id": ..., "q": ..., "a": ..., "target": ..., "problem": ..., "solution": ...}

Run:
  bash scripts/data/prepare_eval_sets.sh
"""

import json
from pathlib import Path

from glr_paths import get

REVISION = "740312add88f781978c0658806c59bc2815b9866"


def _build_q(question: str) -> str:
    return (
        "<|im_start|>user\n"
        f"Question: {question}\n"
        "Answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test", revision=REVISION)

    out_path = Path(get("eval_dir")) / "gsm8k_zeroshot.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            rec = {
                "id": idx,
                "q": _build_q(row["question"]),
                "a": row["answer"],
                "target": row["answer"],
                "problem": row["question"],
                "solution": "",
            }
            f.write(json.dumps(rec))
            f.write("\n")

    print(f"wrote {len(ds)} examples -> {out_path}")


if __name__ == "__main__":
    main()
