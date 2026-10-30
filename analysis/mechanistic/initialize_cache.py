# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Create the immutable experiment manifest and fixed K-star record."""

import argparse
import platform
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from mechanistic_common import (
    DATASETS,
    DEFAULT_CACHE_ROOT,
    GLR_CHECKPOINT,
    K_STAR,
    K_VALUES,
    SFT_CHECKPOINT,
    sha256_file,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--skip-model-weight-hash", action="store_true")
    args = parser.parse_args()
    args.cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": {
            "model": "Qwen3-1.7B",
            "datasets": list(DATASETS),
            "example_selection": "all",
            "k_values": list(K_VALUES),
            "k_coverage": {
                "gsm8k": list(K_VALUES),
                "math500": list(K_VALUES),
                "svamp": [K_STAR],
            },
            "k_star": K_STAR,
            "k_star_selection": "fixed_main_setting_not_test_selected",
            "training": "none",
            "control_subset": {
                "size_per_dataset": 100,
                "seed": 2026,
                "selection": "deterministic random without replacement",
            },
        },
        "checkpoints": {
            "glr": str(GLR_CHECKPOINT),
            "sft": str(SFT_CHECKPOINT),
            "sft_config_sha256": sha256_file(SFT_CHECKPOINT / "config.json"),
            "sft_tokenizer_sha256": sha256_file(SFT_CHECKPOINT / "tokenizer.json"),
            "glr_config_sha256": sha256_file(GLR_CHECKPOINT / "config.json"),
            "glr_tokenizer_sha256": sha256_file(GLR_CHECKPOINT / "tokenizer.json"),
            "glr_weights_sha256": (
                None
                if args.skip_model_weight_hash
                else sha256_file(GLR_CHECKPOINT / "model.safetensors")
            ),
        },
        "datasets": {
            name: {
                "path": str(spec.path),
                "sha256": sha256_file(spec.path),
                "expected_count": spec.expected_count,
                "generation_cap": spec.cap,
            }
            for name, spec in DATASETS.items()
        },
        "decoding": {
            "greedy": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "caps": {name: spec.cap for name, spec in DATASETS.items()},
            "learned_accuracy_source": "line-aligned scored greedy decodes",
            "correct_at_cap_rule": (
                "correctness marker AND generation ended (EOS) AND assistant token count <= cap"
            ),
        },
        "reference": {
            "kind": "CoT-SFT trace teacher-forced through GLR checkpoint",
            "purpose": "explicit CoT token embeddings and GLR contextual states",
        },
        "runtime_at_manifest_creation": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    write_json(args.cache_root / "manifest.json", manifest)
    write_json(
        args.cache_root / "k_star.json",
        {
            "K": K_STAR,
            "source": "main GLR setting K=10; not selected on test",
        },
    )
    print(f"Initialized {args.cache_root}")


if __name__ == "__main__":
    main()
