# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Score SVAMP custom decodes (numeric exact_match + math_verify).

Designed for the decoding JSONL format used in this repo:
  {"question": <prompt>, "answer": <gold answer>, "predicted": <model output>}

Also supports scoring lm-eval-harness `samples_*.jsonl` logs via `--pred-format
lm_eval_samples` (reads predictions from `resps`) and optional joining against a
gold file by id.

Example:
  python src/metric/svamp_score.py \
    --pred-file /path/to/svamp_latent10_gen8192_temp0.0.json

Dependencies:
  - exact_match: no extra deps
  - math_verify: `pip install sympy antlr4-python3-runtime==4.11 math-verify`
"""

import argparse
from pathlib import Path

import amc23_score as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-file", type=Path, required=True)
    parser.add_argument("--pred-key", type=str, default="predicted")
    parser.add_argument("--gold-key", type=str, default="answer")
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument(
        "--gold-solution-file",
        type=Path,
        default=None,
        help=(
            "Optional gold JSONL/JSON. If provided, evaluation runs over this file and "
            "predictions are joined by id (gold `id` to pred `doc_id`, with a positional fallback)."
        ),
    )
    parser.add_argument(
        "--math-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute math_verify using `math_verify` (requires extra deps).",
    )
    parser.add_argument(
        "--extraction-mode",
        type=str,
        default="flexible-extract",
        choices=("flexible-extract", "strict-match"),
        help="Answer extraction mode for exact_match.",
    )
    parser.add_argument(
        "--pred-format",
        type=str,
        default="simple",
        choices=("simple", "lm_eval_samples"),
        help=(
            "Prediction file format. 'simple' expects a per-line dict with a prediction field (default: predicted). "
            "'lm_eval_samples' expects lm-eval-harness log_samples JSONL with nested `resps`."
        ),
    )
    parser.add_argument(
        "--print-mismatches",
        type=int,
        default=0,
        help="Print the first N exact_match mismatches.",
    )
    parser.add_argument(
        "--save-file",
        type=Path,
        default=None,
        help="Optional path to save an annotated copy of the prediction file with per-record 'correct' fields.",
    )
    args = parser.parse_args()

    results, metrics = core.score_file(
        args.pred_file,
        pred_key=args.pred_key,
        gold_key=args.gold_key,
        question_key=args.question_key,
        extraction_mode=args.extraction_mode,
        compute_math_verify=args.math_verify,
        gold_solution_file=args.gold_solution_file,
        pred_format=args.pred_format,
    )

    exact_correct = sum(r.exact_match for r in results)
    mv_correct = sum(r.math_verify for r in results)

    print(f"file: {args.pred_file}")
    print(f"total: {int(metrics['n'])}")
    print(f"exact_match_correct: {exact_correct}")
    print(f"math_verify_correct: {mv_correct}")
    print(f"exact_match: {metrics['exact_match']:.6f}")
    print(f"math_verify: {metrics['math_verify']:.6f}")

    if args.print_mismatches:
        shown = 0
        for i, r in enumerate(results):
            if r.exact_match:
                continue
            print("-" * 80)
            print(f"idx: {i}")
            print(f"math_verify: {r.math_verify}")
            print(f"gold_raw: {r.gold_raw!r}")
            print(f"pred_final_raw: {r.pred_final_raw!r}")
            print(f"gold_norm: {r.gold_norm!r}")
            print(f"pred_norm: {r.pred_norm!r}")
            shown += 1
            if shown >= args.print_mismatches:
                break

    if args.save_file is not None:
        core._save_annotated_file(
            args.pred_file,
            args.save_file,
            pred_key=args.pred_key,
            gold_key=args.gold_key,
            question_key=args.question_key,
            extraction_mode=args.extraction_mode,
            compute_math_verify=args.math_verify,
            gold_solution_file=args.gold_solution_file,
            pred_format=args.pred_format,
        )


if __name__ == "__main__":
    main()
