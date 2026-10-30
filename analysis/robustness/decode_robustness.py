# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Resumable inference worker for the Qwen3-1.7B robustness analysis.

This is an adaptation of the GLR decoder ``src/inference/generate_glr.py``.
It keeps the decoder unchanged while adding exact token records,
fixed sampling arguments, cache ingestion, prefix validation, and resumability.

The custom GLR generation implementation currently supports batch size one.
For matched generation semantics this worker therefore uses batch size one for
both methods. Each completed example is flushed immediately to JSONL.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig

ROBUSTNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROBUSTNESS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROBUSTNESS_DIR))

from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config
from glr_qwen3.modeling_qwen3_latent import CustomQwen3ForCausalLM
from robustness_common import (
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    GREEDY_CACHE_FILES,
    K_BY_METHOD,
    checkpoint_for,
    extract_final_assistant_turn,
    load_dataset,
    make_record,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "smoke",
            "greedy_pipeline",
            "greedy_cache",
            "sampled",
            "sampled_bundle",
        ),
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    parser.add_argument("--method", choices=("cot_sft", "glr_k10"))
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help=(
            "Sampled bundle task as DATASET:METHOD:SAMPLE_ID:SEED. " "May be repeated."
        ),
    )
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-examples", type=int, default=20)
    parser.add_argument("--smoke-examples", type=int, default=8)
    parser.add_argument("--force-direct", action="store_true")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["example_id"]) for row in read_jsonl(path)}


def eos_ids(model: CustomQwen3ForCausalLM, tokenizer: AutoTokenizer) -> set[int]:
    values = model.generation_config.eos_token_id
    if values is None:
        values = tokenizer.eos_token_id
    if isinstance(values, int):
        return {values}
    return {int(value) for value in (values or [])}


def cached_eos_ids(method: str, tokenizer: AutoTokenizer) -> set[int]:
    """Load stop IDs without materializing model weights."""
    generation_config = GenerationConfig.from_pretrained(checkpoint_for(method))
    values = generation_config.eos_token_id
    if values is None:
        values = tokenizer.eos_token_id
    if isinstance(values, int):
        return {values}
    return {int(value) for value in (values or [])}


def load_model(
    method: str, cache_dir: str | None, attn_implementation: str
) -> tuple[AutoTokenizer, CustomQwen3ForCausalLM]:
    checkpoint = checkpoint_for(method)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
    config_kwargs = {"latent_reason_token_budget": K_BY_METHOD[method]}
    config = CustomQwen3Config.from_pretrained(
        checkpoint, cache_dir=cache_dir, **config_kwargs
    )
    if method == "glr_k10":
        if not bool(config.enable_latent_reason):
            raise ValueError(f"{checkpoint}: latent reasoning is not enabled")
        if bool(config.ce_latent_tokens):
            raise ValueError(
                f"{checkpoint}: expected corrected no-CE GLR checkpoint, but "
                "ce_latent_tokens=True"
            )
    model = CustomQwen3ForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        cache_dir=cache_dir,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).eval()
    model.generation_config.use_cache = True
    return tokenizer, model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rng_state(path: Path, count: int, replicate_seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "count": count,
        "replicate_seed": replicate_seed,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_rng_state(path: Path, expected_seed: int) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state["replicate_seed"]) != expected_seed:
        raise ValueError(
            f"RNG state {path} belongs to seed {state['replicate_seed']}, "
            f"expected {expected_seed}"
        )
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    return int(state["count"])


def direct_generate(
    *,
    tokenizer: AutoTokenizer,
    model: CustomQwen3ForCausalLM,
    prompt: str,
    max_text_tokens: int,
    sampled: bool,
    rng_seed: int | None,
) -> tuple[list[int], str, bool, bool, float]:
    if rng_seed is not None:
        seed_everything(rng_seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_args: dict[str, Any] = {
        "max_new_tokens": max_text_tokens,
        "use_cache": True,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "logits_to_keep": 1,
    }
    if sampled:
        generation_args.update(
            {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 0,
            }
        )
    else:
        generation_args.update(
            {
                "do_sample": False,
                "top_p": 1.0,
                "top_k": 0,
            }
        )
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_args)
    runtime = time.perf_counter() - started
    prompt_len = int(inputs["input_ids"].shape[-1])
    raw_tokens = [int(value) for value in output[0, prompt_len:].tolist()]
    raw_completion = tokenizer.decode(raw_tokens, skip_special_tokens=False)

    completion = extract_final_assistant_turn(raw_completion)
    tokens = [
        int(value) for value in tokenizer.encode(completion, add_special_tokens=False)
    ]
    stops = eos_ids(model, tokenizer)
    stopped_on_eos = bool(raw_tokens and raw_tokens[-1] in stops)
    hit_cap = len(raw_tokens) == max_text_tokens and not stopped_on_eos
    return tokens, completion, hit_cap, stopped_on_eos, runtime


def record_direct(
    *,
    dataset: str,
    row: dict[str, Any],
    method: str,
    tokenizer: AutoTokenizer,
    model: CustomQwen3ForCausalLM,
    total_budget: int,
    sampled: bool,
    sample_id: int,
    seed: int | None,
    rng_seed: int | None = None,
    source: str = "direct",
) -> dict[str, Any]:
    tokens, text, hit_cap, stopped, runtime = direct_generate(
        tokenizer=tokenizer,
        model=model,
        prompt=str(row["q"]),
        # HF ``max_new_tokens`` already counts the GLR latent rollout slots:
        # they occupy generated positions in the serialized assistant turn.
        max_text_tokens=total_budget,
        sampled=sampled,
        rng_seed=rng_seed,
    )
    return make_record(
        dataset=dataset,
        row=row,
        method=method,
        decoding="sampled_t0.6_p0.95" if sampled else "greedy",
        sample_id=sample_id,
        seed=seed,
        total_step_budget=total_budget,
        token_ids=tokens,
        generated_text=text,
        hit_cap=hit_cap,
        stopped_on_eos=stopped,
        runtime_seconds=runtime,
        source=source,
    )


def _cache_fields(method: str) -> tuple[str, str, str]:
    if method == "cot_sft":
        return "q", "a", "decoded_output"
    if method == "glr_k10":
        return "question", "answer", "predicted"
    raise KeyError(method)


def _completion_from_cached(
    row: dict[str, Any], prompt_key: str, prediction_key: str
) -> str:
    prompt = str(row[prompt_key])
    decoded = str(row[prediction_key])
    if not decoded.startswith(prompt):
        raise ValueError(
            f"cached decoded output for id={row.get('id')} does not start with prompt"
        )
    return extract_final_assistant_turn(decoded[len(prompt) :])


def ingest_greedy_cache(
    dataset: str,
    method: str,
    tokenizer: AutoTokenizer,
    stops: set[int],
    destination: Path,
) -> None:
    rows = load_dataset(dataset)
    cache_path = GREEDY_CACHE_FILES[method][dataset]
    cached = list(read_jsonl(cache_path))
    if len(cached) != len(rows):
        raise ValueError(
            f"{dataset}/{method}: cache {cache_path} has {len(cached)} rows; "
            f"expected {len(rows)}"
        )
    if destination.exists():
        destination.unlink()
    spec = DATASETS[dataset]
    max_text = spec.full_cap
    prompt_key, answer_key, prediction_key = _cache_fields(method)
    expected_model = str(checkpoint_for(method))
    for expected, cached_row in zip(rows, cached):
        if str(cached_row[prompt_key]) != str(expected["q"]):
            raise ValueError(
                f"{dataset}/{method}: cached prompt does not match test-set row "
                f"{expected.get('id')}"
            )
        if str(cached_row[answer_key]) != str(expected["a"]):
            raise ValueError(
                f"{dataset}/{method}: cached gold answer does not match test-set "
                f"row {expected.get('id')}"
            )
        if method == "cot_sft" and str(expected.get("id")) != str(cached_row.get("id")):
            raise ValueError(f"{dataset}: cached/example ID order mismatch")
        if (
            cached_row.get("model") is not None
            and str(cached_row["model"]) != expected_model
        ):
            raise ValueError(
                f"{dataset}/{method}: cached model is {cached_row['model']!r}, "
                f"expected {expected_model!r}"
            )
        completion = _completion_from_cached(cached_row, prompt_key, prediction_key)
        all_tokens = tokenizer.encode(completion, add_special_tokens=False)
        tokens = [int(value) for value in all_tokens[:max_text]]
        stopped = bool(tokens and tokens[-1] in stops and len(all_tokens) <= max_text)
        hit_cap = len(tokens) == max_text and not stopped
        record = make_record(
            dataset=dataset,
            row=expected,
            method=method,
            decoding="greedy",
            sample_id=0,
            seed=None,
            total_step_budget=spec.full_cap,
            token_ids=tokens,
            generated_text=tokenizer.decode(tokens, skip_special_tokens=False),
            hit_cap=hit_cap,
            stopped_on_eos=stopped,
            runtime_seconds=None,
            source="cached",
        )
        append_jsonl(destination, record)


def generate_full_direct(
    dataset: str,
    method: str,
    tokenizer: AutoTokenizer,
    model: CustomQwen3ForCausalLM,
    destination: Path,
) -> None:
    done = completed_ids(destination)
    for row in load_dataset(dataset):
        if str(row["id"]) in done:
            continue
        record = record_direct(
            dataset=dataset,
            row=row,
            method=method,
            tokenizer=tokenizer,
            model=model,
            total_budget=DATASETS[dataset].full_cap,
            sampled=False,
            sample_id=0,
            seed=None,
        )
        append_jsonl(destination, record)


def load_unique_records(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"duplicate example id {key} in {path}")
        result[key] = row
    return result


def validate_and_derive_prefixes(
    *,
    dataset: str,
    method: str,
    tokenizer: AutoTokenizer,
    model: CustomQwen3ForCausalLM,
    full_path: Path,
    prefix_path: Path,
    report_path: Path,
    validation_examples: int,
) -> bool:
    rows = load_dataset(dataset)
    full = load_unique_records(full_path)
    if len(full) != len(rows):
        raise ValueError(f"{full_path}: expected {len(rows)} records, got {len(full)}")
    budget = DATASETS[dataset].constrained_budget
    max_text = budget
    checks: list[dict[str, Any]] = []
    passed = True
    for row in rows[:validation_examples]:
        direct = record_direct(
            dataset=dataset,
            row=row,
            method=method,
            tokenizer=tokenizer,
            model=model,
            total_budget=budget,
            sampled=False,
            sample_id=0,
            seed=None,
        )
        prefix = list(full[str(row["id"])]["generated_token_ids"])[:max_text]
        matches = prefix == direct["generated_token_ids"]
        passed = passed and matches
        checks.append(
            {
                "example_id": str(row["id"]),
                "matches": matches,
                "prefix_token_count": len(prefix),
                "direct_token_count": len(direct["generated_token_ids"]),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "method": method,
                "n_checked": len(checks),
                "passed": passed,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        return False

    if prefix_path.exists():
        prefix_path.unlink()
    stops = eos_ids(model, tokenizer)
    for row in rows:
        full_record = full[str(row["id"])]
        tokens = [int(value) for value in full_record["generated_token_ids"][:max_text]]
        stopped = bool(
            tokens
            and tokens[-1] in stops
            and len(full_record["generated_token_ids"]) <= max_text
        )
        hit_cap = len(tokens) == max_text and not stopped
        prefix_record = make_record(
            dataset=dataset,
            row=row,
            method=method,
            decoding="greedy",
            sample_id=0,
            seed=None,
            total_step_budget=budget,
            token_ids=tokens,
            generated_text=tokenizer.decode(tokens, skip_special_tokens=False),
            hit_cap=hit_cap,
            stopped_on_eos=stopped,
            runtime_seconds=None,
            source="prefix_derived",
        )
        append_jsonl(prefix_path, prefix_record)
    return True


def run_greedy_pipeline(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw"
    full_path = raw / f"greedy_full_{args.dataset}_{args.method}.jsonl"
    prefix_path = raw / f"greedy_prefix_{args.dataset}_{args.method}.jsonl"
    report_path = (
        args.output_dir
        / "validation"
        / f"prefix_equivalence_{args.dataset}_{args.method}.json"
    )
    tokenizer, model = load_model(args.method, args.cache_dir, args.attn_implementation)

    cache_path = GREEDY_CACHE_FILES[args.method][args.dataset]
    used_cache = cache_path.exists() and not args.force_direct
    if used_cache:
        ingest_greedy_cache(
            args.dataset,
            args.method,
            tokenizer,
            eos_ids(model, tokenizer),
            full_path,
        )
    else:
        generate_full_direct(args.dataset, args.method, tokenizer, model, full_path)

    passed = validate_and_derive_prefixes(
        dataset=args.dataset,
        method=args.method,
        tokenizer=tokenizer,
        model=model,
        full_path=full_path,
        prefix_path=prefix_path,
        report_path=report_path,
        validation_examples=args.validation_examples,
    )
    if not passed and used_cache:
        # Preserve the invalid cache-derived file for audit and switch to a
        # separate direct path before atomically replacing the active result.
        rejected = full_path.with_name(full_path.stem + "_rejected_cache.jsonl")
        full_path.replace(rejected)
        generate_full_direct(args.dataset, args.method, tokenizer, model, full_path)
        passed = validate_and_derive_prefixes(
            dataset=args.dataset,
            method=args.method,
            tokenizer=tokenizer,
            model=model,
            full_path=full_path,
            prefix_path=prefix_path,
            report_path=report_path,
            validation_examples=args.validation_examples,
        )
    if not passed:
        raise RuntimeError(
            f"{args.dataset}/{args.method}: direct constrained decoding does not "
            "match the full-cap prefix; prefix-derived scoring aborted"
        )


def derive_cached_prefixes(
    *,
    dataset: str,
    method: str,
    tokenizer: AutoTokenizer,
    stops: set[int],
    full_path: Path,
    prefix_path: Path,
    report_path: Path,
) -> None:
    """Derive the constrained greedy operating point without a GPU."""
    rows = load_dataset(dataset)
    full = load_unique_records(full_path)
    if len(full) != len(rows):
        raise ValueError(f"{full_path}: expected {len(rows)} records, got {len(full)}")
    if prefix_path.exists():
        prefix_path.unlink()
    budget = DATASETS[dataset].constrained_budget
    for row in rows:
        full_record = full[str(row["id"])]
        tokens = [int(value) for value in full_record["generated_token_ids"][:budget]]
        stopped = bool(
            tokens
            and tokens[-1] in stops
            and len(full_record["generated_token_ids"]) <= budget
        )
        record = make_record(
            dataset=dataset,
            row=row,
            method=method,
            decoding="greedy",
            sample_id=0,
            seed=None,
            total_step_budget=budget,
            token_ids=tokens,
            generated_text=tokenizer.decode(tokens, skip_special_tokens=False),
            hit_cap=len(tokens) == budget and not stopped,
            stopped_on_eos=stopped,
            runtime_seconds=None,
            source="prefix_derived",
        )
        append_jsonl(prefix_path, record)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "method": method,
                "passed": True,
                "validation_mode": "cache_alignment_and_token_prefix_derivation",
                "n_checked": len(rows),
                "direct_generation_checks": 0,
                "checks": [],
                "note": (
                    "Every cached row was matched line-by-line to the test-set "
                    "prompt and gold answer, tokenized with the checkpoint "
                    "tokenizer, and deterministically prefix-truncated. No GPU "
                    "generation was used for these cached rows."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_greedy_cache(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw"
    full_path = raw / f"greedy_full_{args.dataset}_{args.method}.jsonl"
    prefix_path = raw / f"greedy_prefix_{args.dataset}_{args.method}.jsonl"
    report_path = (
        args.output_dir
        / "validation"
        / f"prefix_equivalence_{args.dataset}_{args.method}.json"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_for(args.method), cache_dir=args.cache_dir
    )
    stops = cached_eos_ids(args.method, tokenizer)
    ingest_greedy_cache(args.dataset, args.method, tokenizer, stops, full_path)
    derive_cached_prefixes(
        dataset=args.dataset,
        method=args.method,
        tokenizer=tokenizer,
        stops=stops,
        full_path=full_path,
        prefix_path=prefix_path,
        report_path=report_path,
    )


def run_sampled_with_model(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: CustomQwen3ForCausalLM,
) -> None:
    if args.sample_id not in (0, 1, 2) or args.seed is None:
        raise ValueError("sampled mode requires --sample-id {0,1,2} and --seed")
    destination = (
        args.output_dir
        / "raw"
        / f"sampled_{args.dataset}_{args.method}_sample{args.sample_id}.jsonl"
    )
    rows = load_dataset(args.dataset)
    existing = list(read_jsonl(destination)) if destination.exists() else []
    if len(existing) > len(rows):
        raise ValueError(f"{destination}: more rows than the source dataset")
    for row_index, record in enumerate(existing):
        if str(record["example_id"]) != str(rows[row_index]["id"]):
            raise ValueError(
                f"{destination}: completed rows are not a source-order prefix"
            )

    rng_dir = (
        args.output_dir
        / "rng_states"
        / f"{args.dataset}_{args.method}_sample{args.sample_id}"
    )
    target_count = len(existing)
    replicate_seed = int(args.seed)
    state_path = rng_dir / f"{target_count:06d}.pt"
    if state_path.exists():
        restored_count = restore_rng_state(state_path, replicate_seed)
        if restored_count != target_count:
            raise ValueError(f"RNG state count mismatch in {state_path}")
    else:
        available: list[tuple[int, Path]] = []
        if rng_dir.exists():
            for candidate in rng_dir.glob("*.pt"):
                try:
                    count = int(candidate.stem)
                except ValueError:
                    continue
                if count <= target_count:
                    available.append((count, candidate))
        if available:
            replay_start, available_path = max(available)
            if restore_rng_state(available_path, replicate_seed) != replay_start:
                raise ValueError(f"RNG state count mismatch in {available_path}")
        else:
            replay_start = 0
            seed_everything(replicate_seed)
            save_rng_state(rng_dir / "000000.pt", 0, replicate_seed)
        # Normally no replay is needed. If a job died between writing a JSONL
        # row and its RNG snapshot, replay only the missing state transitions.
        for row in rows[replay_start:target_count]:
            direct_generate(
                tokenizer=tokenizer,
                model=model,
                prompt=str(row["q"]),
                max_text_tokens=DATASETS[args.dataset].constrained_budget,
                sampled=True,
                rng_seed=None,
            )
            replay_start += 1
            save_rng_state(
                rng_dir / f"{replay_start:06d}.pt",
                replay_start,
                replicate_seed,
            )

    for row_index, row in enumerate(rows[target_count:], start=target_count):
        record = record_direct(
            dataset=args.dataset,
            row=row,
            method=args.method,
            tokenizer=tokenizer,
            model=model,
            total_budget=DATASETS[args.dataset].constrained_budget,
            sampled=True,
            sample_id=args.sample_id,
            seed=int(args.seed),
            rng_seed=None,
        )
        append_jsonl(destination, record)
        save_rng_state(
            rng_dir / f"{row_index + 1:06d}.pt",
            row_index + 1,
            replicate_seed,
        )


def sampled_task_is_complete(args: argparse.Namespace) -> bool:
    destination = (
        args.output_dir
        / "raw"
        / f"sampled_{args.dataset}_{args.method}_sample{args.sample_id}.jsonl"
    )
    if not destination.exists():
        return False
    rows = list(read_jsonl(destination))
    return len(rows) == DATASETS[args.dataset].expected_count


def run_sampled(args: argparse.Namespace) -> None:
    tokenizer, model = load_model(args.method, args.cache_dir, args.attn_implementation)
    run_sampled_with_model(args, tokenizer, model)


def parse_bundle_task(
    specification: str, base: argparse.Namespace
) -> argparse.Namespace:
    fields = specification.split(":")
    if len(fields) != 4:
        raise ValueError(
            f"Invalid --task {specification!r}; expected "
            "DATASET:METHOD:SAMPLE_ID:SEED"
        )
    dataset, method, sample_id, seed = fields
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported task dataset: {dataset}")
    if method not in K_BY_METHOD:
        raise ValueError(f"Unsupported task method: {method}")
    task = copy.copy(base)
    task.dataset = dataset
    task.method = method
    task.sample_id = int(sample_id)
    task.seed = int(seed)
    if task.sample_id not in (0, 1, 2):
        raise ValueError(f"Invalid task sample ID: {task.sample_id}")
    return task


def run_sampled_bundle(args: argparse.Namespace) -> None:
    if not args.task:
        raise ValueError("sampled_bundle requires at least one --task")
    tasks = [parse_bundle_task(specification, args) for specification in args.task]
    for method in ("cot_sft", "glr_k10"):
        pending = [
            task
            for task in tasks
            if task.method == method and not sampled_task_is_complete(task)
        ]
        if not pending:
            continue
        tokenizer, model = load_model(method, args.cache_dir, args.attn_implementation)
        for task in pending:
            run_sampled_with_model(task, tokenizer, model)
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_smoke(args: argparse.Namespace) -> None:
    destination = args.output_dir / "smoke" / f"{args.dataset}_{args.method}.jsonl"
    if destination.exists():
        destination.unlink()
    tokenizer, model = load_model(args.method, args.cache_dir, args.attn_implementation)
    rows = load_dataset(args.dataset, args.smoke_examples)
    for decoding_sampled in (False, True):
        if decoding_sampled:
            seed_everything(17)
        for row_index, row in enumerate(rows):
            record = record_direct(
                dataset=args.dataset,
                row=row,
                method=args.method,
                tokenizer=tokenizer,
                model=model,
                total_budget=DATASETS[args.dataset].constrained_budget,
                sampled=decoding_sampled,
                sample_id=0,
                seed=17 if decoding_sampled else None,
                rng_seed=None,
                source="direct",
            )
            record["smoke_example_index"] = row_index
            append_jsonl(destination, record)


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "The repository GLR implementation supports batch size one; use "
            "--batch-size 1 so CoT-SFT and GLR have identical batching."
        )
    args.output_dir = args.output_dir.resolve()
    single_task_modes = {"smoke", "greedy_pipeline", "greedy_cache", "sampled"}
    if args.mode in single_task_modes and (args.dataset is None or args.method is None):
        raise ValueError(f"{args.mode} requires --dataset and --method")
    if args.mode == "greedy_pipeline":
        run_greedy_pipeline(args)
    elif args.mode == "greedy_cache":
        run_greedy_cache(args)
    elif args.mode == "sampled":
        run_sampled(args)
    elif args.mode == "sampled_bundle":
        run_sampled_bundle(args)
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()
