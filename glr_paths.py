# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Single place for all local paths and model locations used by the GLR scripts.

Defaults work out of the box: base models are pulled from the Hugging Face Hub
and all data/outputs live under this repository. To use your own locations,
create `glr_paths_local.py` next to this file (it is git-ignored) and override
any subset of the entries, e.g.

    MODELS = {"qwen3_1.7b": "/data/models/Qwen3-1.7B"}
    PATHS = {"ckpt_dir": "/scratch/glr/ckpts", "hf_cache_dir": "/scratch/hf_cache"}

Python:  from glr_paths import get;  get("ckpt_dir")
Bash:    ckpt_dir=$(python glr_paths.py ckpt_dir)
         python glr_paths.py            # print all entries
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Base models: Hugging Face Hub IDs or local model directories.
MODELS = {
    "qwen3_0.6b": "Qwen/Qwen3-0.6B",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
}

# Data and output locations. Relative paths are resolved against the repository root.
PATHS = {
    # Hugging Face cache, exported as HF_HOME by scripts/common.sh; empty means the default (~/.cache/huggingface).
    "hf_cache_dir": "",
    # Training data: raw Mixture-of-Thoughts download, chat-formatted, and the filtered 10K subset.
    "mot_raw_dir": "data/train/Mixture-of-Thoughts",
    "mot_tokenized_dir": "data/train/math_tokenized",
    "mot_train_10k_dir": "data/train/math_tokenized_filt_10K_8192tok_100cot",
    # Evaluation sets shipped with this repository.
    "eval_dir": "data/eval",
    # Training checkpoints.
    "ckpt_dir": "ckpts",
    # Raw generations, scored generations, figures, analyses, and baselines.
    "decode_dir": "outputs/decodes",
    "scored_dir": "outputs/scored",
    "figure_dir": "outputs/figures",
    "analysis_dir": "outputs/analysis",
    "baseline_dir": "outputs/baselines",
}


def _load_local_overrides() -> None:
    local_file = REPO_ROOT / "glr_paths_local.py"
    if not local_file.exists():
        return
    spec = importlib.util.spec_from_file_location("glr_paths_local", local_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODELS.update(getattr(module, "MODELS", {}))
    PATHS.update(getattr(module, "PATHS", {}))


_load_local_overrides()


def get(key: str) -> str:
    """Return a model location or an absolute path for `key`."""
    if key in MODELS:
        return MODELS[key]
    if key in PATHS:
        value = PATHS[key]
        if not value:
            return ""
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else REPO_ROOT / path)
    raise KeyError(f"Unknown key '{key}'. Available: {sorted(MODELS) + sorted(PATHS)}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for k in list(MODELS) + list(PATHS):
            print(f"{k}={get(k)}")
    elif len(sys.argv) == 2:
        try:
            print(get(sys.argv[1]))
        except KeyError as e:
            sys.exit(e.args[0])
    else:
        sys.exit("usage: python glr_paths.py [key]")
