# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Run/resume one matched-stack timing shard on exactly one visible GPU."""

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from efficiency_common import (
    ATTENTION_IMPLEMENTATION,
    BATCH_SIZE,
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    DTYPE,
    K_BY_METHOD,
    METHODS,
    NUM_SHARDS,
    REPO_ROOT,
    SCHEMA_VERSION,
    WARMUP_EXAMPLES,
    atomic_write_json,
    canonical_hash,
    checkpoint_for,
    ensure_output_tree,
    load_dataset,
    load_shard_assignment,
    load_static_manifest,
    record_path,
    sha256_text,
    stable_order_bit,
)
from shared_decoder import SharedGreedyDecoder, validate_architecture

sys.path.insert(0, str(REPO_ROOT))
from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config  # noqa: E402
from glr_qwen3.modeling_qwen3_latent import (  # noqa: E402
    CustomQwen3ForCausalLM,
)
from analysis.robustness.robustness_common import score_generation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Diagnostic only: stop after this many prompt pairs in the shard",
    )
    return parser.parse_args()


def command_output(arguments: list[str], default: str = "unavailable") -> str:
    try:
        return subprocess.run(
            arguments, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return default


def visible_gpu_identity() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if visible.startswith("GPU-"):
        return visible
    return (
        command_output(
            ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
            default=visible,
        )
        .splitlines()[0]
        .strip()
    )


def gpu_processes(gpu_identity: str) -> list[dict[str, str]]:
    output = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        default="",
    )
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4 and (
            fields[0] == gpu_identity
            or gpu_identity in fields[0]
            or fields[0] in gpu_identity
        ):
            rows.append(
                {
                    "gpu_uuid": fields[0],
                    "pid": fields[1],
                    "process_name": fields[2],
                    "used_memory_mb": fields[3],
                }
            )
    return rows


def runtime_environment() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    gpu_identity = visible_gpu_identity()
    packages = {}
    for name in ("torch", "transformers", "accelerate", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    processes = gpu_processes(gpu_identity)
    other_processes = [
        row for row in processes if row["pid"] not in (str(os.getpid()), "N/A")
    ]
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_driver": command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "gpu_name": properties.name,
        "gpu_uuid": gpu_identity,
        "gpu_total_memory_bytes": int(total),
        "gpu_free_memory_bytes_at_start": int(free),
        "gpu_power_limit": command_output(
            ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"]
        ),
        "gpu_clocks": command_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ]
        ),
        "gpu_processes_at_start": processes,
        "known_concurrent_gpu_processes": other_processes,
        "gpu_shared_with_other_process": bool(other_processes),
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "dtype": DTYPE,
        "batch_size": BATCH_SIZE,
        "use_kv_cache": True,
        "torch_compile": False,
        "cuda_graphs": False,
        "quantization": False,
        "speculative_decoding": False,
    }


def load_model(method: str) -> CustomQwen3ForCausalLM:
    checkpoint = checkpoint_for(method)
    config = CustomQwen3Config.from_pretrained(
        checkpoint, latent_reason_token_budget=K_BY_METHOD[method]
    )
    if method == "glr_k10":
        if not bool(config.enable_latent_reason) or bool(config.ce_latent_tokens):
            raise ValueError("Expected corrected no-CE GLR checkpoint")
    else:
        config.enable_latent_reason = False
    model = CustomQwen3ForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        device_map={"": torch.cuda.current_device()},
    ).eval()
    model.generation_config.use_cache = True
    model.debug_directional = False
    validate_architecture(model.config)
    return model


def record_is_reusable(
    path: Path,
    *,
    semantic_hash: str,
    timing_hash: str,
    benchmark: str,
    example_id: str,
    method: str,
) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        record.get("status") == "complete"
        and record.get("schema_version") == SCHEMA_VERSION
        and record.get("semantic_hash") == semantic_hash
        and record.get("timing_hash") == timing_hash
        and record.get("benchmark") == benchmark
        and str(record.get("example_id")) == example_id
        and record.get("method") == method
    )


def selected_examples(
    assignment: dict[str, Any], shard_id: int
) -> list[tuple[str, int, dict[str, Any]]]:
    selected = []
    for benchmark in DATASETS:
        for index, row in enumerate(load_dataset(benchmark)):
            key = f"{benchmark}:{row['id']}"
            if int(assignment["assignment"][key]) == shard_id:
                selected.append((benchmark, index, row))
    return selected


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.num_shards != NUM_SHARDS or not 0 <= args.shard_id < NUM_SHARDS:
        raise ValueError(f"Expected shard ID in [0,{NUM_SHARDS - 1}]")
    output_dir = args.output_dir.resolve()
    ensure_output_tree(output_dir)
    static = load_static_manifest(output_dir)
    assignment = load_shard_assignment(output_dir)
    if assignment["semantic_hash"] != static["semantic_hash"]:
        raise ValueError("Shard assignment/static manifest semantic mismatch")

    environment = runtime_environment()
    if environment["gpu_shared_with_other_process"]:
        raise RuntimeError(
            "Known concurrent GPU process detected; refusing latency measurement"
        )
    timing_payload = {
        "semantic_hash": static["semantic_hash"],
        "gpu_model": environment["gpu_name"],
        "gpu_total_memory_bytes": environment["gpu_total_memory_bytes"],
        "software": environment["packages"],
        "python": environment["python"],
        "cuda_runtime": environment["torch_cuda_runtime"],
        "cuda_driver": environment["cuda_driver"],
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "dtype": DTYPE,
        "batch_size": BATCH_SIZE,
        "use_kv_cache": True,
        "shard_assignment_hash": canonical_hash(assignment["assignment"]),
    }
    timing_hash = canonical_hash(timing_payload)
    run_id = (
        f"matched_efficiency_shard{args.shard_id}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    environment.update(
        {
            "run_id": run_id,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "semantic_hash": static["semantic_hash"],
            "timing_hash": timing_hash,
        }
    )
    environment_path = (
        output_dir / "manifests" / f"environment_shard{args.shard_id}.json"
    )
    atomic_write_json(environment_path, environment)
    log_path = output_dir / "logs" / f"run_shard{args.shard_id}.log"
    append_log(log_path, f"start run_id={run_id} timing_hash={timing_hash}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_for("cot_sft"))
    glr_tokenizer = AutoTokenizer.from_pretrained(checkpoint_for("glr_k10"))
    if tokenizer.get_vocab() != glr_tokenizer.get_vocab():
        raise ValueError("Tokenizer vocabularies differ")
    models = {method: load_model(method) for method in METHODS}
    decoders = {
        method: SharedGreedyDecoder(
            model=models[method],
            tokenizer=tokenizer,
            method=method,
            k=K_BY_METHOD[method],
            max_total_steps=2048,
        )
        for method in METHODS
    }

    selected = selected_examples(assignment, args.shard_id)
    weights = assignment["predicted_weights"]
    warmups = sorted(
        selected,
        key=lambda item: (
            int(weights[f"{item[0]}:{item[2]['id']}"]),
            item[0],
            str(item[2]["id"]),
        ),
    )[:WARMUP_EXAMPLES]
    for method in METHODS:
        append_log(log_path, f"warmup method={method} n={len(warmups)}")
        for benchmark, _index, row in warmups:
            prompt_ids = tokenizer(str(row["q"]), return_tensors="pt").input_ids.to(
                models[method].device
            )
            decoders[method].max_total_steps = DATASETS[benchmark].max_total_steps
            result = decoders[method].generate(prompt_ids)
            if method == "glr_k10" and result["latent_steps"] != 10:
                raise AssertionError("GLR warm-up did not execute ten latent steps")

    executed = 0
    skipped = 0
    failed = 0
    pair_count = 0
    for benchmark, index, row in selected:
        if args.limit is not None and pair_count >= args.limit:
            break
        example_id = str(row["id"])
        order_bit = stable_order_bit(benchmark, example_id)
        order = ("cot_sft", "glr_k10") if order_bit == 0 else ("glr_k10", "cot_sft")
        pair_count += 1
        for order_position, method in enumerate(order):
            destination = record_path(output_dir, benchmark, index, example_id, method)
            if not args.force and record_is_reusable(
                destination,
                semantic_hash=static["semantic_hash"],
                timing_hash=timing_hash,
                benchmark=benchmark,
                example_id=example_id,
                method=method,
            ):
                skipped += 1
                continue
            started_request_ns = time.perf_counter_ns()
            prompt_hash = sha256_text(str(row["q"]))
            try:
                tokenization_started = time.perf_counter_ns()
                prompt_ids = tokenizer(str(row["q"]), return_tensors="pt").input_ids.to(
                    models[method].device
                )
                tokenization_ms = (
                    time.perf_counter_ns() - tokenization_started
                ) / 1_000_000.0
                decoder = decoders[method]
                decoder.max_total_steps = DATASETS[benchmark].max_total_steps
                result = decoder.generate(prompt_ids)
                detokenization_started = time.perf_counter_ns()
                generated_text = tokenizer.decode(
                    result["generated_token_ids"],
                    skip_special_tokens=False,
                )
                detokenization_ms = (
                    time.perf_counter_ns() - detokenization_started
                ) / 1_000_000.0
                # Optional request latency includes only tokenization, the
                # synchronized generation region, and detokenization.
                # Benchmark parsing/scoring outside this boundary, just as it
                # is outside the primary generation timer.
                request_latency_ms = (
                    time.perf_counter_ns() - started_request_ns
                ) / 1_000_000.0
                prediction, valid_answer, correct = score_generation(
                    benchmark, generated_text, str(row["a"])
                )
                cache_key = canonical_hash(
                    {
                        "semantic_hash": static["semantic_hash"],
                        "timing_hash": timing_hash,
                        "benchmark": benchmark,
                        "example_id": example_id,
                        "method": method,
                        "checkpoint": static["semantic_payload"]["checkpoints"][method][
                            "fingerprint"
                        ],
                        "prompt_hash": prompt_hash,
                        "max_total_steps": DATASETS[benchmark].max_total_steps,
                    }
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "complete",
                    "cache_key": cache_key,
                    "semantic_hash": static["semantic_hash"],
                    "timing_hash": timing_hash,
                    "run_id": run_id,
                    "benchmark": benchmark,
                    "split": "test",
                    "example_id": example_id,
                    "dataset_index": index,
                    "method": method,
                    "checkpoint_fingerprint": static["semantic_payload"]["checkpoints"][
                        method
                    ]["fingerprint"],
                    "K": K_BY_METHOD[method],
                    "prompt": str(row["q"]),
                    "prompt_hash": prompt_hash,
                    "prompt_token_ids": [
                        int(value) for value in prompt_ids[0].tolist()
                    ],
                    "prompt_token_count": int(prompt_ids.shape[-1]),
                    "generated_token_ids": result["generated_token_ids"],
                    "generated_text": generated_text,
                    "parsed_prediction": prediction,
                    "valid_answer": bool(valid_answer),
                    "gold_answer": str(row["a"]),
                    "correct": bool(correct),
                    "latent_step_count": result["latent_steps"],
                    "generated_text_token_count": result["text_token_count"],
                    "total_post_prompt_step_count": result["total_post_prompt_steps"],
                    "stop_reason": result["stop_reason"],
                    "max_step_truncation": result["max_step_truncation"],
                    "backbone_call_trace": result["backbone_trace"],
                    "head_call_trace": result["head_trace"],
                    "backbone_call_count": result["backbone_call_count"],
                    "vocabulary_head_call_count": result["vocabulary_head_call_count"],
                    "transition_head_call_count": result["transition_head_call_count"],
                    "vocabulary_head_positions": result["vocabulary_head_positions"],
                    "transition_head_positions": result["transition_head_positions"],
                    "generation_latency_ms": result["generation_latency_ms"],
                    "request_latency_ms": request_latency_ms,
                    "tokenization_latency_ms": tokenization_ms,
                    "detokenization_latency_ms": detokenization_ms,
                    "cuda_event_latency_ms": result["cuda_event_latency_ms"],
                    "time_to_first_text_token_ms": result[
                        "time_to_first_text_token_ms"
                    ],
                    "peak_allocated_bytes": result["peak_allocated_bytes"],
                    "peak_reserved_bytes": result["peak_reserved_bytes"],
                    "approximate_flops": result["approximate_flops"]["total"],
                    "approximate_flops_components": result["approximate_flops"],
                    "method_order": list(order),
                    "method_order_position": order_position,
                    "phase": "first" if order_position == 0 else "second",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "environment_manifest": str(
                        environment_path.relative_to(output_dir)
                    ),
                    "exception": None,
                }
                atomic_write_json(destination, record)
                executed += 1
            except Exception as error:
                failed += 1
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "semantic_hash": static["semantic_hash"],
                    "timing_hash": timing_hash,
                    "run_id": run_id,
                    "benchmark": benchmark,
                    "split": "test",
                    "example_id": example_id,
                    "dataset_index": index,
                    "method": method,
                    "K": K_BY_METHOD[method],
                    "prompt_hash": prompt_hash,
                    "gold_answer": str(row["a"]),
                    "method_order": list(order),
                    "method_order_position": order_position,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "environment_manifest": str(
                        environment_path.relative_to(output_dir)
                    ),
                    "exception": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
                atomic_write_json(destination, failure)
                append_log(
                    log_path,
                    f"failure benchmark={benchmark} id={example_id} "
                    f"method={method} error={error!r}",
                )
                raise

    for decoder in decoders.values():
        decoder.close()
    end_processes = gpu_processes(environment["gpu_uuid"])
    environment["gpu_processes_at_end"] = end_processes
    environment["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    environment["records_executed"] = executed
    environment["records_skipped_from_cache"] = skipped
    environment["records_failed"] = failed
    atomic_write_json(environment_path, environment)
    append_log(
        log_path,
        f"complete executed={executed} skipped={skipped} failed={failed}",
    )
    print(
        json.dumps(
            {
                "shard_id": args.shard_id,
                "executed": executed,
                "skipped": skipped,
                "failed": failed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
