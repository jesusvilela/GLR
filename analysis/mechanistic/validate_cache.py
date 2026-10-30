# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Fail-fast validation for mechanistic caches and derived outputs."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors import safe_open
from transformers import AutoTokenizer

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from mechanistic_common import (
    DATASETS,
    DEFAULT_CACHE_ROOT,
    GLR_CHECKPOINT,
    K_STAR,
    expected_ids,
    extract_assistant_turn,
    read_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--skip-derived", action="store_true")
    parser.add_argument("--skip-existing-identity", action="store_true")
    return parser.parse_args()


def metadata(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metadata.jsonl"
    if not path.exists() or not (run_dir / "_COMPLETE").exists():
        raise FileNotFoundError(f"incomplete run: {run_dir}")
    return list(read_jsonl(path))


def tensor_for(run_dir: Path, row: dict[str, Any], suffix: str):
    for meta_path in run_dir.glob("shard-*.metadata.jsonl"):
        rows = [json.loads(line) for line in meta_path.read_text().splitlines()]
        if any(item["tensor_prefix"] == row["tensor_prefix"] for item in rows):
            tensor_path = meta_path.with_name(
                meta_path.name.replace(".metadata.jsonl", ".safetensors")
            )
            with safe_open(tensor_path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(f"{row['tensor_prefix']}_{suffix}")
    raise KeyError((row["example_id"], suffix))


def assert_same_ids(run_dir: Path, expected: list[str]) -> list[dict[str, Any]]:
    rows = metadata(run_dir)
    found = [str(row["example_id"]) for row in rows]
    if found != expected:
        raise AssertionError(f"{run_dir}: example IDs/order differ from test set")
    return rows


def assert_ordered_subset_ids(
    run_dir: Path, expected: list[str]
) -> list[dict[str, Any]]:
    rows = metadata(run_dir)
    found = [str(row["example_id"]) for row in rows]
    expected_set = set(expected)
    if len(found) != len(set(found)) or not set(found).issubset(expected_set):
        raise AssertionError(f"{run_dir}: invalid or duplicate subset IDs")
    positions = [expected.index(value) for value in found]
    if positions != sorted(positions):
        raise AssertionError(f"{run_dir}: subset order differs from test set")
    return rows


def validate_identity(cache_root: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(GLR_CHECKPOINT)
    audit: dict[str, Any] = {}
    for dataset, spec in DATASETS.items():
        run_dir = cache_root / "runs" / dataset / "learned" / f"K={K_STAR}" / "seed=0"
        cache_rows = metadata(run_dir)
        supplied = list(read_jsonl(ANALYSIS_DIR.parents[1] / spec.glr_k10_decode))
        if len(cache_rows) != len(supplied):
            raise AssertionError(f"{dataset}: supplied/cache K=10 counts differ")
        counts = {
            "n": len(cache_rows),
            "exact_K_latent_states": 0,
            "latent_prefix_exact": 0,
            "whole_short_prefix_exact": 0,
            "resume_prefix_exact": 0,
            "resume_available": 0,
            "thought_truncated_no_resume": 0,
            "valid_realized_latent_steps": 0,
        }
        for index, (cached, old) in enumerate(zip(cache_rows, supplied)):
            text = extract_assistant_turn(str(old["predicted"]))
            old_ids = tokenizer.encode(text, add_special_tokens=False)[: spec.cap]
            new_ids = [
                int(value)
                for value in tensor_for(run_dir, cached, "generated_ids").tolist()
            ]
            if old_ids != new_ids:
                raise AssertionError(
                    f"{dataset} row {index}: cached supplied decode IDs differ"
                )
            replay_ids = [
                int(value)
                for value in tensor_for(run_dir, cached, "replay_prefix_ids").tolist()
            ]
            replay_text = tokenizer.decode(replay_ids, skip_special_tokens=False)
            replay_matches = text.startswith(replay_text)
            close_marker = "</think>"
            replay_close = replay_text.find(close_marker)
            supplied_close = text.find(close_marker)
            latent_matches = (
                replay_close >= 0
                and supplied_close >= 0
                and replay_text[: replay_close + len(close_marker)]
                == text[: supplied_close + len(close_marker)]
            )
            resume_ids = [
                int(value)
                for value in tensor_for(run_dir, cached, "resume_ids").tolist()
            ]
            resume_h = tensor_for(run_dir, cached, "resume_h")
            resume_at_value = cached.get("resume_generated_index")
            if resume_at_value is None:
                # A supplied decode can contain an opening <think> region but
                # no textual resume point before its cap. The replay then
                # intentionally stores no resume tokens or hidden states.
                if not bool(cached.get("thought_truncated")):
                    raise AssertionError(
                        f"{dataset} row {index}: missing resume point without "
                        "a truncated thought"
                    )
                if resume_ids or int(resume_h.numel()) != 0:
                    raise AssertionError(
                        f"{dataset} row {index}: no-resume replay has nonempty "
                        "resume tensors"
                    )
                resume_matches = False
                counts["thought_truncated_no_resume"] += 1
            else:
                resume_at = int(resume_at_value)
                resume_hidden_count = int(resume_h.shape[0])
                # The final sampled resume token need not itself be consumed
                # by another backbone step. An immediate EOS therefore has
                # one resume token and zero resume hidden states; ordinary
                # short continuations have one hidden state per stored token.
                allowed_hidden_counts = {
                    len(resume_ids),
                    max(0, len(resume_ids) - 1),
                }
                if not resume_ids or resume_hidden_count not in allowed_hidden_counts:
                    raise AssertionError(
                        f"{dataset} row {index}: available resume has "
                        "inconsistent token/hidden tensors"
                    )
                through_resume = tokenizer.decode(
                    replay_ids[: resume_at + len(resume_ids)],
                    skip_special_tokens=False,
                )
                resume_matches = text.startswith(through_resume)
                counts["resume_available"] += 1
            if bool(cached.get("replay_prefix_matches_supplied")) != replay_matches:
                raise AssertionError(
                    f"{dataset} row {index}: replay-prefix audit flag differs"
                )
            if (
                bool(cached.get("latent_prefix_matches_supplied")) != latent_matches
                or bool(cached.get("replay_resume_matches_supplied")) != resume_matches
                or cached.get("replay_equivalence_basis")
                != "diagnostic_exact_decoded_text_prefix_not_required"
            ):
                raise AssertionError(
                    f"{dataset} row {index}: replay-equivalence audit failed"
                )
            actual_latent_steps = int(cached.get("actual_latent_steps", -1))
            if not 1 <= actual_latent_steps <= K_STAR:
                raise AssertionError(
                    f"{dataset} row {index}: realized latent steps "
                    f"{actual_latent_steps} outside [1,{K_STAR}]"
                )
            counts["valid_realized_latent_steps"] += 1
            if actual_latent_steps == K_STAR:
                counts["exact_K_latent_states"] += 1
            counts["latent_prefix_exact"] += int(latent_matches)
            counts["whole_short_prefix_exact"] += int(replay_matches)
            counts["resume_prefix_exact"] += int(resume_matches)
        if counts["valid_realized_latent_steps"] != counts["n"]:
            raise AssertionError(
                f"{dataset}: not every learned replay has a valid realized "
                f"latent-step count in [1,{K_STAR}]"
            )
        audit[dataset] = {
            **counts,
            "early_terminal_count": (counts["n"] - counts["exact_K_latent_states"]),
            "exact_K_latent_state_rate": (
                counts["exact_K_latent_states"] / counts["n"]
            ),
            "latent_prefix_exact_rate": counts["latent_prefix_exact"] / counts["n"],
            "whole_short_prefix_exact_rate": counts["whole_short_prefix_exact"]
            / counts["n"],
            "resume_prefix_exact_rate": counts["resume_prefix_exact"] / counts["n"],
            "resume_prefix_exact_rate_when_available": (
                counts["resume_prefix_exact"] / counts["resume_available"]
                if counts["resume_available"]
                else None
            ),
        }
    return {
        "status": "passed",
        "interpretation": (
            "Replay textual equivalence is diagnostic, not an inclusion "
            "criterion. Accuracy/length labels come from the line-aligned "
            "scored decodes; trajectories come from checkpoint "
            "replays with the predeclared K=10 maximum latent budget on the "
            "current hardware. Realized latent steps are validated in [1,K] "
            "and early terminal events are reported explicitly."
        ),
        "datasets": audit,
    }


def main() -> None:
    args = parse_args()
    replay_audit: dict[str, Any] | None = None
    for dataset in DATASETS:
        ids = expected_ids(dataset)
        reference_dir = args.cache_root / "references" / dataset / "cot_sft_glr_encoded"
        assert_same_ids(reference_dir, ids)
        learned_dir = (
            args.cache_root / "runs" / dataset / "learned" / f"K={K_STAR}" / "seed=0"
        )
        learned_rows = assert_same_ids(learned_dir, ids)
        learned_by_id = {str(row["example_id"]): row for row in learned_rows}
        control_root = args.cache_root / "runs" / dataset
        for complete in control_root.rglob("_COMPLETE"):
            run_dir = complete.parent
            if "/learned/" in run_dir.as_posix():
                rows = assert_same_ids(run_dir, ids)
            else:
                rows = assert_ordered_subset_ids(run_dir, ids)
            if f"K={K_STAR}" in str(run_dir) and "forced_continuation" not in str(
                run_dir
            ):
                for row in rows:
                    if str(row["example_id"]) not in learned_by_id:
                        raise AssertionError(
                            f"{run_dir}: control ID missing from learned K=10"
                        )
                    actual_steps = int(row.get("actual_latent_steps", -1))
                    if not 1 <= actual_steps <= int(row["K"]):
                        raise AssertionError(
                            f"{run_dir}: realized latent steps outside [1,K]"
                        )
                    if row["method"] in {
                        "matched_norm_random_direction",
                        "random_head_norm_matched",
                        "noisy_transition",
                    }:
                        left = np.asarray(row["learned_transition_norms_float32"])
                        right = np.asarray(row["applied_transition_norms_float32"])
                        if not np.allclose(left, right, rtol=1e-4, atol=1e-4):
                            raise AssertionError(f"{run_dir}: norm matching failed")
                    if row["method"] == "zero_transition":
                        z = tensor_for(run_dir, row, "z").float().numpy()
                        if not np.allclose(z, z[:1], rtol=0.0, atol=2e-3):
                            raise AssertionError(f"{run_dir}: zero trajectory moved")
    if not args.skip_existing_identity:
        replay_audit = validate_identity(args.cache_root)
        write_json(args.cache_root / "replay_equivalence_audit.json", replay_audit)
    if not args.skip_derived:
        derived = args.cache_root / "derived"
        trajectory = pd.read_parquet(derived / "trajectory_steps.parquet")
        summary = pd.read_parquet(derived / "per_example_summary.parquet")
        required = {
            "trajectory_steps.parquet",
            "per_example_summary.parquet",
            "nearest_neighbors.parquet",
            "aggregate_results.csv",
            "regression_results.csv",
            "trajectory_aggregate_results.csv",
            "paired_comparisons.csv",
            "control_subset_aggregate_results.csv",
        }
        missing = [name for name in required if not (derived / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing derived files: {missing}")
        if not (
            (trajectory.aligned_j.diff().fillna(0) >= 0) | (trajectory.k == 1)
        ).all():
            raise AssertionError("alignment is not monotonic")
        if not trajectory.progress.between(0, 1).all():
            raise AssertionError("progress outside (0,1]")
        if summary.final_cot_embedding_drift.isna().any():
            raise AssertionError("missing endpoint drift")
        rendered_required = {
            "mechanistic_table.tex",
            "mechanistic_table.csv",
            "k_endpoint_table.tex",
            "k_endpoint_table.csv",
            "ablation_table.tex",
            "trajectory_figure.pdf",
            "nearest_neighbor_examples.tex",
            "qualitative_example_selection.csv",
            "summary_paragraph.txt",
        }
        rendered_dir = args.cache_root / "rendered"
        missing_rendered = [
            name for name in rendered_required if not (rendered_dir / name).exists()
        ]
        if missing_rendered:
            raise FileNotFoundError(f"missing rendered files: {missing_rendered}")
        selection = pd.read_csv(
            args.cache_root / "rendered" / "qualitative_example_selection.csv"
        )
        if len(selection) != 2 * len(DATASETS):
            raise AssertionError(
                "expected exactly two qualitative examples per dataset"
            )
        if not selection.correct.astype(bool).all():
            raise AssertionError(
                "a qualitative example is not GLR-correct at the token cap"
            )
        if not selection.original_correct.astype(bool).all():
            raise AssertionError(
                "a qualitative example lacks the supplied correctness marker"
            )
    write_json(
        args.cache_root / "validation_report.json",
        {
            "status": "passed",
            "cache_validation": True,
            "existing_identity_validation": not args.skip_existing_identity,
            "derived_output_validation": not args.skip_derived,
            "datasets": list(DATASETS),
            "K_maximum_latent_budget": K_STAR,
            "replay_audit_status": (
                replay_audit.get("status") if replay_audit is not None else "skipped"
            ),
        },
    )
    print("All requested cache and derived-output validations passed.")


if __name__ == "__main__":
    main()
