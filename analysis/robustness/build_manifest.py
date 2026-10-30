# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Build the immutable-before-inference manifest for the robustness analysis."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robustness_common import (
    ANALYSIS_SEED,
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    GLR_CHECKPOINT,
    N_BOOTSTRAP,
    REPO_ROOT,
    SEEDS,
    GREEDY_CACHE_FILES,
    CORRECTNESS_FIELDS,
    SFT_CHECKPOINT,
    dataset_revision,
    sha256_file,
)

PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "matplotlib",
    "sympy",
    "math-verify",
    "antlr4-python3-runtime",
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
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed-in-current-environment"
    return versions


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    model_file = path / "model.safetensors"
    config_file = path / "config.json"
    generation_config_file = path / "generation_config.json"
    return {
        "path": str(path),
        "model_sha256": sha256_file(model_file),
        "model_size_bytes": model_file.stat().st_size,
        "config_sha256": sha256_file(config_file),
        "generation_config_sha256": sha256_file(generation_config_file),
    }


def gpu_models() -> list[str]:
    output = command_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        default="",
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def base_manifest() -> dict[str, Any]:
    status = command_output(["git", "status", "--porcelain"], default="")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        },
        "model_size": "1.7B",
        "checkpoints": {
            "cot_sft": checkpoint_metadata(SFT_CHECKPOINT),
            "glr_k10": checkpoint_metadata(GLR_CHECKPOINT),
        },
        "datasets": {
            name: {
                "path": str(spec.path),
                "revision": dataset_revision(name),
                "split": spec.split,
                "expected_count": spec.expected_count,
                "constrained_total_step_budget": spec.constrained_budget,
                "full_total_step_cap": spec.full_cap,
            }
            for name, spec in DATASETS.items()
        },
        "cached_generation_candidates": {
            method: {
                name: {
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": sha256_file(path) if path.exists() else None,
                    "source_cap_text_tokens": 8192,
                    "provenance": (
                        "User-confirmed greedy decode from the corresponding "
                        "fixed checkpoint; rows follow the test-set order."
                    ),
                }
                for name, path in method_paths.items()
            }
            for method, method_paths in GREEDY_CACHE_FILES.items()
        },
        "generation": {
            "K": {"cot_sft": 0, "glr_k10": 10},
            "total_step_accounting": {
                "definition": (
                    "Tokenizer length of the final assistant turn after the "
                    "question/prompt, with add_special_tokens=False."
                ),
                "assistant_turn_extraction": (
                    "Same delimiter split and strip operation as "
                    "plots/plot_common.py."
                ),
                "generated_wrappers_and_eos_included": True,
                "prompt_tokens_included": False,
                "glr_latent_steps": (
                    "The K=10 latent positions are already serialized inside "
                    "the generated assistant turn and are not added again."
                ),
            },
            "greedy": {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "num_beams": 1,
                "repetition_penalty": 1.0,
            },
            "sampled": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 0,
                "num_beams": 1,
                "repetition_penalty": 1.0,
                "sample_ids": [0, 1, 2],
                "seeds": list(SEEDS),
                "rng_resume_policy": (
                    "Each ordered replicate is seeded exactly once. Python, Torch "
                    "CPU, and Torch CUDA RNG states are saved after each completed "
                    "problem and restored on resume."
                ),
            },
            "dtype": "bfloat16",
            "use_cache": True,
            "stop_strings": [],
            "eos_handling": (
                "Use the checkpoint generation_config eos_token_id list; EOS is "
                "retained in generated_token_ids and prevents hit_cap=True."
            ),
            "batch_size": 1,
            "batching_note": (
                "The repository GLR generation path supports batch size one; "
                "CoT-SFT uses the same policy for matched semantics."
            ),
            "greedy_cache_policy": (
                "Reuse the six scored greedy decodes. Validate every row against "
                "the test-set prompt and gold answer, tokenize with the checkpoint "
                "tokenizer, and compute budgeted accuracy as full-generation "
                "correctness AND complete post-prompt length <= budget."
            ),
            "direct_greedy_validation_examples_per_method_dataset": 0,
            "sampled_gpu_shards_maximum": 5,
            "sampled_cache_policy": (
                "Resume any normalized per-seed JSONL prefix already present and "
                "schedule only missing examples. Never mix pilot decodes whose "
                "seed or exact sampling configuration is unknown."
            ),
        },
        "analysis": {
            "seed": ANALYSIS_SEED,
            "bootstrap_replicates": N_BOOTSTRAP,
            "bootstrap_unit_greedy": "problem",
            "bootstrap_unit_sampled": "problem cluster retaining all 3 samples",
            "correctness_fields": CORRECTNESS_FIELDS,
            "sampled_budget_rule": (
                "Use the same benchmark/method-specific evaluator; cap-reaching "
                "unfinished trajectories receive no credit."
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "cuda_available": "pending runtime inspection",
            "cuda_version": "pending runtime inspection",
            "gpu_models": [],
        },
    }


def update_runtime(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["environment"]["packages"] = package_versions()
    manifest["environment"]["python"] = sys.version
    manifest["environment"]["platform"] = platform.platform()
    manifest["environment"]["gpu_models"] = gpu_models()
    try:
        import torch

        manifest["environment"]["cuda_available"] = torch.cuda.is_available()
        manifest["environment"]["cuda_version"] = torch.version.cuda
        manifest["environment"]["torch_version"] = torch.__version__
    except Exception as error:
        manifest["environment"]["cuda_available"] = False
        manifest["environment"]["cuda_error"] = repr(error)
    manifest["runtime_inspected_at_utc"] = datetime.now(timezone.utc).isoformat()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-update", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "run_manifest.json"
    if args.runtime_update:
        if not destination.exists():
            raise FileNotFoundError(
                f"{destination} must be built before the runtime update"
            )
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        manifest = update_runtime(manifest)
    else:
        manifest = base_manifest()
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
