# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Build immutable experiment metadata and cache-length-balanced shard assignments."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from efficiency_common import (
    ANALYSIS_SEED,
    ATTENTION_IMPLEMENTATION,
    BATCH_SIZE,
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    DTYPE,
    GLR_CHECKPOINT,
    GREEDY_REFERENCE_FILES,
    K_BY_METHOD,
    LATENCY_DIR,
    METHODS,
    N_BOOTSTRAP,
    NUM_SHARDS,
    REPO_ROOT,
    SCHEMA_VERSION,
    SFT_CHECKPOINT,
    WARMUP_EXAMPLES,
    atomic_write_json,
    canonical_hash,
    ensure_output_tree,
    load_dataset,
    read_jsonl,
    sha256_file,
)

ROBUSTNESS_DIR = REPO_ROOT / "analysis/robustness"
from analysis.robustness.robustness_common import (
    extract_final_assistant_turn,
    score_generation,
)  # noqa: E402

PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "flash-attn",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "matplotlib",
    "sympy",
    "math-verify",
)


def command_output(arguments: list[str], default: str = "unavailable") -> str:
    try:
        return subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return default


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def fingerprint_checkpoint(path: Path) -> dict[str, Any]:
    small_names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    )
    small = {
        name: sha256_file(path / name) for name in small_names if (path / name).exists()
    }
    weights = []
    for candidate in sorted(path.glob("*.safetensors")):
        stat = candidate.stat()
        weights.append(
            {
                "name": candidate.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "absolute_path": str(path.resolve()),
        "small_file_sha256": small,
        "weight_metadata": weights,
        "construction": (
            "SHA256 for configuration/tokenizer metadata; absolute path, size, "
            "and mtime_ns for safetensor weights."
        ),
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def reference_completion(method: str, row: dict[str, Any]) -> str:
    if method == "cot_sft":
        prompt, decoded = str(row["q"]), str(row["decoded_output"])
    else:
        prompt, decoded = str(row["question"]), str(row["predicted"])
    if not decoded.startswith(prompt):
        raise ValueError("Reference decode does not begin with its prompt")
    return extract_final_assistant_turn(decoded[len(prompt) :])


def expected_step_weights(
    tokenizer: AutoTokenizer,
) -> tuple[dict[str, int], dict[str, Any]]:
    weights: dict[str, int] = {}
    reference_summary: dict[str, Any] = {}
    for benchmark, spec in DATASETS.items():
        source = load_dataset(benchmark)
        per_method: dict[str, list[int]] = {}
        reference_summary[benchmark] = {}
        for method in METHODS:
            cached = list(read_jsonl(GREEDY_REFERENCE_FILES[method][benchmark]))
            if len(cached) != len(source):
                raise ValueError(f"{method}/{benchmark}: reference row-count mismatch")
            lengths: list[int] = []
            correctness: list[bool] = []
            for expected, reference in zip(source, cached):
                prompt_key = "q" if method == "cot_sft" else "question"
                answer_key = "a" if method == "cot_sft" else "answer"
                if str(reference[prompt_key]) != str(expected["q"]):
                    raise ValueError(f"{method}/{benchmark}: prompt mismatch")
                if str(reference[answer_key]) != str(expected["a"]):
                    raise ValueError(f"{method}/{benchmark}: gold mismatch")
                completion = reference_completion(method, reference)
                token_ids = tokenizer.encode(completion, add_special_tokens=False)
                token_ids = token_ids[: spec.max_total_steps]
                lengths.append(len(token_ids))
                truncated_text = tokenizer.decode(token_ids, skip_special_tokens=False)
                _prediction, _valid, correct = score_generation(
                    benchmark, truncated_text, str(expected["a"])
                )
                correctness.append(bool(correct))
            per_method[method] = lengths
            reference_summary[benchmark][method] = {
                "n": len(lengths),
                "accuracy_percent": 100 * sum(correctness) / len(correctness),
                "mean_total_steps": statistics.fmean(lengths),
                "median_total_steps": statistics.median(lengths),
                "source": "scored greedy decode truncated to latency cap",
            }
        for index, row in enumerate(source):
            weights[f"{benchmark}:{row['id']}"] = (
                per_method["cot_sft"][index] + per_method["glr_k10"][index]
            )
    return weights, reference_summary


def build_shards(weights: dict[str, int]) -> dict[str, Any]:
    shards = [
        {"shard_id": shard_id, "predicted_total_steps": 0, "keys": []}
        for shard_id in range(NUM_SHARDS)
    ]
    for key, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
        target = min(
            shards,
            key=lambda shard: (
                shard["predicted_total_steps"],
                len(shard["keys"]),
                shard["shard_id"],
            ),
        )
        target["keys"].append(key)
        target["predicted_total_steps"] += int(weight)
    assignment = {
        key: int(shard["shard_id"]) for shard in shards for key in shard["keys"]
    }
    return {
        "num_shards": NUM_SHARDS,
        "weight_definition": (
            "Sum of CoT-SFT and GLR cached greedy token counts, each truncated "
            "to the benchmark total-step cap; used only for runtime balancing."
        ),
        "predicted_weights": weights,
        "assignment": assignment,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    ensure_output_tree(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    glr_tokenizer = AutoTokenizer.from_pretrained(GLR_CHECKPOINT)
    if tokenizer.get_vocab() != glr_tokenizer.get_vocab():
        raise ValueError("CoT-SFT and GLR tokenizer vocabularies differ")
    if (
        tokenizer.special_tokens_map != glr_tokenizer.special_tokens_map
        or tokenizer.chat_template != glr_tokenizer.chat_template
    ):
        raise ValueError("CoT-SFT and GLR tokenizer special/template settings differ")
    sft_generation = json.loads(
        (SFT_CHECKPOINT / "generation_config.json").read_text(encoding="utf-8")
    )
    glr_generation = json.loads(
        (GLR_CHECKPOINT / "generation_config.json").read_text(encoding="utf-8")
    )
    if sft_generation.get("eos_token_id") != glr_generation.get("eos_token_id"):
        raise ValueError("CoT-SFT and GLR stopping-token IDs differ")

    checkpoints = {
        "cot_sft": fingerprint_checkpoint(SFT_CHECKPOINT),
        "glr_k10": fingerprint_checkpoint(GLR_CHECKPOINT),
    }
    dataset_payload = {}
    for name, spec in DATASETS.items():
        rows = load_dataset(name)
        dataset_payload[name] = {
            "path": str(spec.path),
            "sha256": sha256_file(spec.path),
            "split": "test",
            "expected_count": spec.expected_count,
            "max_total_steps": spec.max_total_steps,
            "ordered_example_id_fingerprint": canonical_hash(
                [str(row["id"]) for row in rows]
            ),
            "prompt_fingerprint": canonical_hash([str(row["q"]) for row in rows]),
        }

    source_paths = {
        "shared_decoder": LATENCY_DIR / "shared_decoder.py",
        "worker": LATENCY_DIR / "run_matched_efficiency.py",
        "transition_model": REPO_ROOT / "glr_qwen3/modeling_qwen3_latent.py",
        "answer_adapter": ROBUSTNESS_DIR / "robustness_common.py",
        "gsm8k_evaluator": REPO_ROOT / "src/metric/gsm8k_accuracy.py",
        "svamp_evaluator": REPO_ROOT / "src/metric/amc23_score.py",
        "math500_evaluator": REPO_ROOT / "src/metric/minerva_math500_score.py",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    git_status = command_output(["git", "status", "--porcelain"], "")
    semantic_payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoints": checkpoints,
        "datasets": dataset_payload,
        "K": K_BY_METHOD,
        "decoder": {
            "stack": "shared manual HuggingFace/PyTorch autoregressive decoder",
            "greedy": True,
            "sampling": False,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "use_kv_cache": True,
            "batch_size": BATCH_SIZE,
            "torch_compile": False,
            "cuda_graphs": False,
            "quantization": False,
            "speculative_decoding": False,
            "eos_token_ids": sft_generation["eos_token_id"],
            "prompt_construction": "raw q field, without an added chat template",
            "prompt_template_fingerprint": canonical_hash(
                {
                    "construction": "raw q field",
                    "chat_template_unused": tokenizer.chat_template,
                }
            ),
            "tokenizer_special_tokens": tokenizer.special_tokens_map,
            "glr_call_sequence": (
                "Prefill selects <think>; one shared backbone call consumes its "
                "embedding and transition 1 constructs latent input 1; ten "
                "latent backbone updates are run, with transitions 2--10 after "
                "updates 1--9; the shared vocabulary head is first called again "
                "on latent update 10. Thus there are exactly ten transition-head "
                "calls and no vocabulary projection during updates 1--9."
            ),
        },
        "source_hashes": source_hashes,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty_state_hash": canonical_hash(git_status.splitlines()),
    }
    semantic_hash = canonical_hash(semantic_payload)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_hash": semantic_hash,
        "semantic_payload": semantic_payload,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "analysis": {
            "bootstrap_replicates": N_BOOTSTRAP,
            "analysis_seed": ANALYSIS_SEED,
        },
        "warmup_examples_per_loaded_method": WARMUP_EXAMPLES,
        "timing_definition": (
            "Synchronized wall clock from prefill start through final token/"
            "stopping event; excludes tokenization, detokenization, scoring, "
            "model loading, dataset loading, and file writes."
        ),
        "flop_definition": (
            "Approximate dense-matmul FLOPs from actual q/s backbone traces plus "
            "instrumented vocabulary and transition head positions."
        ),
        "architecture": {
            key: json.loads(
                (GLR_CHECKPOINT / "config.json").read_text(encoding="utf-8")
            )[key]
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
            )
        },
        "reference_decodes": {
            method: {
                benchmark: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for benchmark, path in paths.items()
                if benchmark in DATASETS
            }
            for method, paths in GREEDY_REFERENCE_FILES.items()
        },
    }
    destination = output_dir / "manifests" / "static_manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("semantic_hash") != semantic_hash:
            completed_records = list(
                (output_dir / "cache" / "generations").glob("*/*.json")
            )
            if completed_records:
                raise ValueError(
                    f"{destination} exists with a different semantic hash and "
                    "timing records exist; use a new OUTPUT_DIR"
                )
            atomic_write_json(destination, manifest)
    else:
        atomic_write_json(destination, manifest)

    weights, reference_summary = expected_step_weights(tokenizer)
    manifest["reference_summary"] = reference_summary
    atomic_write_json(destination, manifest)
    shards = build_shards(weights)
    shards["semantic_hash"] = semantic_hash
    atomic_write_json(output_dir / "manifests" / "shard_assignment.json", shards)
    print(destination)
    for shard in shards["shards"]:
        print(
            f"shard={shard['shard_id']} examples={len(shard['keys'])} "
            f"predicted_steps={shard['predicted_total_steps']}"
        )


if __name__ == "__main__":
    main()
