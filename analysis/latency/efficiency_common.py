# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Shared, model-free configuration for the matched-stack efficiency study."""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LATENCY_DIR = Path(__file__).resolve().parent
REPO_ROOT = LATENCY_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from glr_paths import get  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(get("analysis_dir")) / "latency"
TEST_SET_DIR = Path(get("eval_dir"))

PYTHON_BIN = "python"
SFT_CHECKPOINT = Path(get("ckpt_dir")) / "cot_sft_qwen3-1.7B"
GLR_CHECKPOINT = Path(get("ckpt_dir")) / "glr_qwen3-1.7B"

SCHEMA_VERSION = "efficiency-qwen3-1.7B-v1"
METHODS = ("cot_sft", "glr_k10")
K_BY_METHOD = {"cot_sft": 0, "glr_k10": 10}
ANALYSIS_SEED = 20260726
N_BOOTSTRAP = 10_000
ATTENTION_IMPLEMENTATION = "flash_attention_2"
DTYPE = "bfloat16"
BATCH_SIZE = 1
NUM_SHARDS = 5
WARMUP_EXAMPLES = 10


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    expected_count: int
    max_total_steps: int

    @property
    def path(self) -> Path:
        return TEST_SET_DIR / self.filename


ALL_DATASETS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec("gsm8k", "gsm8k_zeroshot.jsonl", 1319, 2048),
    "svamp": DatasetSpec("svamp", "svamp.jsonl", 300, 2048),
    "math500": DatasetSpec(
        "math500", "minerva_math500_zeroshot.jsonl", 500, 4096
    ),
}


def selected_dataset_names() -> tuple[str, ...]:
    """Return a validated, ordered benchmark selection.

    The default is all benchmarks. Launchers can select
    complete benchmarks without changing any dataset rows or generation caps.
    """

    raw = os.environ.get("LATENCY_BENCHMARKS", "").strip()
    if not raw:
        return tuple(ALL_DATASETS)
    names = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("LATENCY_BENCHMARKS selected no benchmarks")
    if len(names) != len(set(names)):
        raise ValueError(f"LATENCY_BENCHMARKS contains duplicates: {raw}")
    unknown = [name for name in names if name not in ALL_DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown LATENCY_BENCHMARKS entries {unknown}; "
            f"choose from {tuple(ALL_DATASETS)}"
        )
    return names


SELECTED_DATASET_NAMES = selected_dataset_names()
DATASETS: dict[str, DatasetSpec] = {
    name: ALL_DATASETS[name] for name in SELECTED_DATASET_NAMES
}

# Scored greedy generations (scripts/score/score_decodes.sh).
GREEDY_REFERENCE_FILES: dict[str, dict[str, Path]] = {
    "cot_sft": {
        "gsm8k": Path(get("scored_dir")) / "cot_sft_qwen3-1.7B" / "gsm8k_zeroshot.jsonl",
        "math500": Path(get("scored_dir")) / "cot_sft_qwen3-1.7B" / "minerva_math500_zeroshot.jsonl",
        "svamp": Path(get("scored_dir")) / "cot_sft_qwen3-1.7B" / "svamp.jsonl",
    },
    "glr_k10": {
        "gsm8k": Path(get("scored_dir")) / "glr_qwen3-1.7B" / "gsm8k_zeroshot_latent10.jsonl",
        "math500": Path(get("scored_dir")) / "glr_qwen3-1.7B" / "minerva_math500_zeroshot_latent10.jsonl",
        "svamp": Path(get("scored_dir")) / "glr_qwen3-1.7B" / "svamp_latent10.jsonl",
    },
}


def checkpoint_for(method: str) -> Path:
    if method == "cot_sft":
        return SFT_CHECKPOINT
    if method == "glr_k10":
        return GLR_CHECKPOINT
    raise KeyError(method)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def load_dataset(name: str) -> list[dict[str, Any]]:
    spec = DATASETS[name]
    rows = list(read_jsonl(spec.path))
    if len(rows) != spec.expected_count:
        raise ValueError(
            f"{spec.path}: expected {spec.expected_count}, got {len(rows)}"
        )
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{spec.path}: duplicate example IDs")
    return rows


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def stable_order_bit(benchmark: str, example_id: str) -> int:
    return int(sha256_text(f"{benchmark}:{example_id}")[:16], 16) % 2


def safe_example_component(example_id: str) -> str:
    return sha256_text(example_id)[:12]


def record_path(
    output_dir: Path, benchmark: str, index: int, example_id: str, method: str
) -> Path:
    return (
        output_dir
        / "cache"
        / "generations"
        / benchmark
        / f"{index:04d}_{safe_example_component(example_id)}_{method}.json"
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_output_tree(output_dir: Path) -> None:
    for relative in (
        "manifests",
        "cache/generations",
        "cache/timing",
        "traces",
        "summaries",
        "tables",
        "figures",
        "logs",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def load_static_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "manifests" / "static_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Build the manifest first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_shard_assignment(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "manifests" / "shard_assignment.json"
    if not path.exists():
        raise FileNotFoundError(f"Build the shard assignment first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
