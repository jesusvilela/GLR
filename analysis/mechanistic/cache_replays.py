# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Cache trajectories at low cost from already-produced greedy decodes.

This worker performs no answer decoding.  It (1) teacher-forces the scored
CoT-SFT trace through the GLR checkpoint to obtain an explicit CoT reference,
or (2) runs only the first K latent transitions and attaches accuracy/length
from the line-aligned scored GLR decode.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
for path in (REPO_ROOT, ANALYSIS_DIR):
    sys.path.insert(0, str(path))

from cache_trajectories import (  # noqa: E402
    CaptureController,
    _special_span,
    _storage_dtype,
    completed_ids,
    flush_shard,
    stack,
    verify_partial_cache,
)
from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config  # noqa: E402
from glr_qwen3.modeling_qwen3_latent import (  # noqa: E402
    CustomQwen3ForCausalLM,
)
from mechanistic_common import (  # noqa: E402
    DATASETS,
    DEFAULT_CACHE_ROOT,
    GLR_CHECKPOINT,
    K_STAR,
    K_VALUES,
    cache_run_dir,
    decode_path,
    extract_assistant_turn,
    generation_finished,
    load_dataset,
    marked_control_correct,
    marked_correct,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)


class StopAfterLatentTrajectory(StoppingCriteria):
    """Stop after K latent states plus a short explicit resume prefix."""

    def __init__(self, controller: CaptureController, resume_tokens: int = 16):
        self.controller = controller
        self.resume_tokens = resume_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        del input_ids, scores, kwargs
        return (
            self.controller.resume_generated_index is not None
            and len(self.controller.resume_hidden) >= self.resume_tokens
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--task", required=True, choices=("reference", "learned"))
    parser.add_argument("--k", type=int, default=K_STAR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--tensor-storage-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.task == "reference" and args.k != K_STAR:
        parser.error("--k is unused for reference; leave it at the default")
    if args.task == "learned" and args.k not in K_VALUES:
        parser.error(f"learned replay K must be one of {K_VALUES}")
    if args.dataset == "svamp" and args.task == "learned" and args.k != K_STAR:
        parser.error("SVAMP has a supplied decode only for K=10")
    return args


def load_model(args: argparse.Namespace):
    latent_budget = 0 if args.task == "reference" else args.k
    config = CustomQwen3Config.from_pretrained(
        GLR_CHECKPOINT,
        cache_dir=args.cache_dir,
        latent_reason_token_budget=latent_budget,
    )
    tokenizer = AutoTokenizer.from_pretrained(GLR_CHECKPOINT, cache_dir=args.cache_dir)
    model = CustomQwen3ForCausalLM.from_pretrained(
        GLR_CHECKPOINT,
        config=config,
        cache_dir=args.cache_dir,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval()
    model.debug_directional = False
    return tokenizer, model


def validate_alignment(
    dataset: str,
    decode_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    source: Path,
) -> None:
    if len(decode_rows) != len(gold_rows):
        raise ValueError(
            f"{source}: {len(decode_rows)} rows != {len(gold_rows)} test rows"
        )
    for index, (record, target) in enumerate(zip(decode_rows, gold_rows)):
        source_prompt = record.get("q", record.get("question"))
        if source_prompt != target["q"]:
            raise ValueError(f"{source}:{index + 1}: prompt/order mismatch")


def run_dir_for(args: argparse.Namespace) -> Path:
    if args.task == "reference":
        return cache_run_dir(args.cache_root, args.dataset, "reference", 0)
    return cache_run_dir(args.cache_root, args.dataset, "learned", args.k, seed=0)


def make_metadata_base(
    args: argparse.Namespace,
    index: int,
    gold: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        "example_id": str(gold.get("id", index)),
        "dataset": args.dataset,
        "seed": 0,
        "noise_sigma": None,
        "forced_continuation": False,
        "tensor_prefix": prefix,
        "prompt_hash": sha256_text(str(gold["q"])),
        "gold_answer": str(gold["a"]),
        "checkpoint": str(GLR_CHECKPOINT),
    }


def cache_reference(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: CustomQwen3ForCausalLM,
    gold_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    run_dir: Path,
    done: set[str],
) -> None:
    spec = DATASETS[args.dataset]
    storage_dtype = _storage_dtype(args.tensor_storage_dtype)
    shard_index = len(list(run_dir.glob("shard-*.safetensors")))
    records: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    start_ids = [int(value) for value in model.config.think_tokens_start]
    end_ids = [int(value) for value in model.config.think_tokens_end]

    for index, (gold, decoded) in enumerate(zip(gold_rows, decode_rows)):
        example_id = str(gold.get("id", index))
        if example_id in done:
            continue
        assistant = extract_assistant_turn(str(decoded.get("decoded_output", "")))
        assistant_ids = tokenizer.encode(assistant, add_special_tokens=False)[
            : spec.cap
        ]
        reasoning_start, reasoning_end, truncated = _special_span(
            assistant_ids, start_ids, end_ids
        )
        if reasoning_end <= reasoning_start:
            raise ValueError(
                f"{args.dataset}/{example_id}: empty CoT-SFT reasoning span"
            )
        prompt_ids = tokenizer.encode(str(gold["q"]), add_special_tokens=False)
        forward_ids = prompt_ids + assistant_ids[:reasoning_end]
        input_ids = torch.tensor([forward_ids], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            output = model.model(input_ids=input_ids, use_cache=False)
        start = len(prompt_ids) + reasoning_start
        end = len(prompt_ids) + reasoning_end
        h_discrete = (
            output.last_hidden_state[0, start:end]
            .detach()
            .to("cpu", dtype=storage_dtype)
        )
        reasoning_ids = assistant_ids[reasoning_start:reasoning_end]
        if h_discrete.shape[0] != len(reasoning_ids):
            raise RuntimeError(
                f"{args.dataset}/{example_id}: reference state/token mismatch"
            )

        prefix = f"e{shard_index:05d}_{len(records):04d}"
        metadata = make_metadata_base(args, index, gold, prefix)
        sft_correct, correct_key = marked_correct(args.dataset, decoded, "sft")
        metadata.update(
            {
                "method": "reference",
                "K": 0,
                "reference_kind": "cot_sft_teacher_forced_through_glr",
                "reference_decode": str(decode_path(args.dataset, "sft")),
                "reference_correct": sft_correct,
                "correctness_key": correct_key,
                "reference_truncated_at_cap": truncated,
                "reasoning_start": reasoning_start,
                "reasoning_end": reasoning_end,
                "generated_token_count": len(assistant_ids),
            }
        )
        tensors[f"{prefix}_prompt_ids"] = torch.tensor(prompt_ids, dtype=torch.int64)
        tensors[f"{prefix}_generated_ids"] = torch.tensor(
            assistant_ids, dtype=torch.int64
        )
        tensors[f"{prefix}_reasoning_ids"] = torch.tensor(
            reasoning_ids, dtype=torch.int64
        )
        tensors[f"{prefix}_h_discrete"] = h_discrete.contiguous()
        records.append(metadata)
        if len(records) == args.shard_size:
            flush_shard(run_dir, shard_index, records, tensors)
            shard_index += 1
            records, tensors = [], {}
    if records:
        flush_shard(run_dir, shard_index, records, tensors)


def cache_learned(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: CustomQwen3ForCausalLM,
    gold_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    run_dir: Path,
    done: set[str],
) -> None:
    spec = DATASETS[args.dataset]
    storage_dtype = _storage_dtype(args.tensor_storage_dtype)
    width = int(model.config.hidden_size)
    shard_index = len(list(run_dir.glob("shard-*.safetensors")))
    records: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}

    for index, (gold, decoded) in enumerate(zip(gold_rows, decode_rows)):
        example_id = str(gold.get("id", index))
        if example_id in done:
            continue
        inputs = tokenizer(str(gold["q"]), return_tensors="pt").to(model.device)
        controller = CaptureController(
            model=model,
            method="learned",
            k=args.k,
            seed=0,
            example_id=example_id,
            donor_index=None,
            noise_sigma=None,
            storage_dtype=storage_dtype,
        )
        controller.prompt_len = int(inputs["input_ids"].shape[-1])
        controller.install()
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.k + 32,
                    do_sample=False,
                    num_beams=1,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    logits_to_keep=1,
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList(
                        [StopAfterLatentTrajectory(controller)]
                    ),
                )
        finally:
            controller.close()
        if (
            not controller.z
            or len(controller.z) > args.k
            or len(controller.h) != len(controller.z)
        ):
            raise RuntimeError(
                f"{args.dataset}/{example_id}: captured z={len(controller.z)}, "
                f"h={len(controller.h)}, budget K={args.k}"
            )

        assistant = extract_assistant_turn(str(decoded.get("predicted", "")))
        assistant_ids_full = tokenizer.encode(assistant, add_special_tokens=False)
        assistant_ids = assistant_ids_full[: spec.cap]
        replay_generated_ids = [
            int(value) for value in output[0, inputs["input_ids"].shape[-1] :].tolist()
        ]
        resume_at = controller.resume_generated_index
        replay_resume_ids = (
            replay_generated_ids[resume_at : resume_at + 16]
            if resume_at is not None
            else []
        )
        # The supplied JSONL contains decoded text, not original token IDs.
        # Tokenization is not unique: for example, five generated newline
        # tokens can be re-encoded as one multi-newline token even though the
        # decoded strings are identical. Validate exact decoded-text prefixes
        # and preserve the actual replay IDs separately.
        replay_text = tokenizer.decode(replay_generated_ids, skip_special_tokens=False)
        replay_matches = assistant.startswith(replay_text)
        replay_through_resume_text = (
            tokenizer.decode(
                replay_generated_ids[: resume_at + len(replay_resume_ids)],
                skip_special_tokens=False,
            )
            if resume_at is not None
            else ""
        )
        resume_matches = (
            assistant.startswith(replay_through_resume_text)
            if resume_at is not None
            else False
        )
        common_prefix_char_count = next(
            (
                position
                for position, (left, right) in enumerate(zip(replay_text, assistant))
                if left != right
            ),
            min(len(replay_text), len(assistant)),
        )
        close_marker = "</think>"
        replay_close = replay_text.find(close_marker)
        supplied_close = assistant.find(close_marker)
        latent_prefix_matches = (
            replay_close >= 0
            and supplied_close >= 0
            and replay_text[: replay_close + len(close_marker)]
            == assistant[: supplied_close + len(close_marker)]
        )
        original_correct, correct_key = marked_correct(args.dataset, decoded, "glr")
        control_correct, control_correct_key = marked_control_correct(
            args.dataset, decoded
        )
        within_cap = generation_finished(str(decoded.get("predicted", ""))) and len(assistant_ids_full) <= spec.cap
        correct_at_cap = original_correct and within_cap
        control_correct_at_cap = control_correct and within_cap
        parsed: str | None = None
        valid = len(assistant_ids_full) <= spec.cap
        reasoning_start, reasoning_end, thought_truncated = _special_span(
            assistant_ids,
            [int(value) for value in model.config.think_tokens_start],
            [int(value) for value in model.config.think_tokens_end],
        )

        prefix = f"e{shard_index:05d}_{len(records):04d}"
        metadata = make_metadata_base(args, index, gold, prefix)
        metadata.update(
            {
                "method": "learned",
                "K": args.k,
                "trajectory_only_replay": True,
                "decode_source": str(decode_path(args.dataset, "glr", args.k)),
                "correctness_key": correct_key,
                "original_correct": original_correct,
                "correct": correct_at_cap,
                "correct_at_cap": correct_at_cap,
                "control_correct": control_correct_at_cap,
                "control_correctness_key": control_correct_key,
                "parsed_answer": parsed,
                "valid_answer": valid,
                "generated_text": assistant,
                "generated_token_count": min(len(assistant_ids_full), spec.cap),
                "uncapped_generated_token_count": len(assistant_ids_full),
                "generation_cap": spec.cap,
                "truncated": len(assistant_ids_full) > spec.cap,
                "thought_truncated": thought_truncated,
                "stopped_on_eos": False,
                "reasoning_start": reasoning_start,
                "reasoning_end": reasoning_end,
                "resume_generated_index": resume_at,
                "replay_resume_matches_supplied": resume_matches,
                "replay_prefix_matches_supplied": replay_matches,
                "latent_prefix_matches_supplied": latent_prefix_matches,
                "replay_common_prefix_char_count": common_prefix_char_count,
                "replay_equivalence_basis": (
                    "diagnostic_exact_decoded_text_prefix_not_required"
                ),
                "replay_prefix_token_count": len(replay_generated_ids),
                "accuracy_and_length_source": (
                    "line_aligned_scored_greedy_decode"
                ),
                "position_ids": controller.position_ids,
                "kv_cache_lengths": controller.cache_lengths,
                "kv_cache_length_before_resume": (
                    controller.cache_lengths[-1] + 1
                    if controller.cache_lengths
                    else None
                ),
                "learned_transition_norms_float32": (
                    controller.learned_transition_norms
                ),
                "applied_transition_norms_float32": (
                    controller.applied_transition_norms
                ),
                "actual_latent_steps": len(controller.z),
            }
        )
        prompt_ids = [int(value) for value in inputs["input_ids"][0].tolist()]
        tensors[f"{prefix}_prompt_ids"] = torch.tensor(prompt_ids, dtype=torch.int64)
        tensors[f"{prefix}_generated_ids"] = torch.tensor(
            assistant_ids, dtype=torch.int64
        )
        tensors[f"{prefix}_replay_prefix_ids"] = torch.tensor(
            replay_generated_ids, dtype=torch.int64
        )
        tensors[f"{prefix}_z"] = stack(controller.z, width)
        tensors[f"{prefix}_h_latent"] = stack(controller.h, width)
        tensors[f"{prefix}_delta"] = stack(controller.delta, width)
        tensors[f"{prefix}_resume_ids"] = torch.tensor(
            replay_resume_ids, dtype=torch.int64
        )
        tensors[f"{prefix}_resume_h"] = stack(controller.resume_hidden[:16], width)
        records.append(metadata)
        if len(records) == args.shard_size:
            flush_shard(run_dir, shard_index, records, tensors)
            shard_index += 1
            records, tensors = [], {}
    if records:
        flush_shard(run_dir, shard_index, records, tensors)


def main() -> None:
    args = parse_args()
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    gold_rows = load_dataset(args.dataset)
    source = decode_path(
        args.dataset,
        "sft" if args.task == "reference" else "glr",
        None if args.task == "reference" else args.k,
    )
    decode_rows = list(read_jsonl(source))
    validate_alignment(args.dataset, decode_rows, gold_rows, source)
    run_dir = run_dir_for(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_existing:
        verify_partial_cache(run_dir)
    done = completed_ids(run_dir) if args.resume_existing else set()
    if done and not args.resume_existing:
        raise FileExistsError(
            f"{run_dir} contains partial output; pass --resume-existing"
        )
    if len(done) == len(gold_rows) and (run_dir / "_COMPLETE").exists():
        print(f"Already complete: {run_dir} ({len(done)} examples)")
        return

    tokenizer, model = load_model(args)
    if args.task == "reference":
        cache_reference(args, tokenizer, model, gold_rows, decode_rows, run_dir, done)
    else:
        cache_learned(args, tokenizer, model, gold_rows, decode_rows, run_dir, done)

    all_metadata: list[str] = []
    for path in sorted(run_dir.glob("shard-*.metadata.jsonl")):
        all_metadata.extend(path.read_text(encoding="utf-8").splitlines())
    (run_dir / "metadata.jsonl").write_text(
        "\n".join(all_metadata) + ("\n" if all_metadata else ""),
        encoding="utf-8",
    )
    actual = len(completed_ids(run_dir))
    if actual != len(gold_rows):
        raise RuntimeError(
            f"{run_dir}: expected {len(gold_rows)} examples, found {actual}"
        )
    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "task": args.task,
        "method": "reference" if args.task == "reference" else "learned",
        "K": 0 if args.task == "reference" else args.k,
        "num_examples": actual,
        "checkpoint": str(GLR_CHECKPOINT),
        "checkpoint_config_sha256": sha256_file(GLR_CHECKPOINT / "config.json"),
        "tokenizer_sha256": sha256_file(GLR_CHECKPOINT / "tokenizer.json"),
        "test_set_sha256": sha256_file(DATASETS[args.dataset].path),
        "decode_source": str(source),
        "decode_source_sha256": sha256_file(source),
        "token_cap": DATASETS[args.dataset].cap,
        "correctness": (
            "correctness marker AND assistant token count <= token cap"
            if args.task == "learned"
            else "not used for GLR accuracy"
        ),
        "trajectory_only": args.task == "learned",
        "tensor_storage_dtype": args.tensor_storage_dtype,
    }
    if args.task == "learned":
        metadata_rows = [json.loads(line) for line in all_metadata]
        manifest["replay_audit"] = {
            "purpose": (
                "Diagnostic only. BF16/kernel-sensitive greedy continuations "
                "need not be token- or text-identical across hardware."
            ),
            "label_source": "line-aligned scored greedy decode",
            "trajectory_source": "current checkpoint K=10 short replay",
            "all_examples_have_exact_K_latent_states": all(
                int(row["actual_latent_steps"]) == args.k for row in metadata_rows
            ),
            "latent_prefix_exact_rate": sum(
                bool(row["latent_prefix_matches_supplied"]) for row in metadata_rows
            )
            / actual,
            "whole_short_prefix_exact_rate": sum(
                bool(row["replay_prefix_matches_supplied"]) for row in metadata_rows
            )
            / actual,
            "resume_prefix_exact_rate": sum(
                bool(row["replay_resume_matches_supplied"]) for row in metadata_rows
            )
            / actual,
        }
    write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Complete: {run_dir} ({actual} examples)")


if __name__ == "__main__":
    main()
