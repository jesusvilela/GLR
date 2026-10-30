# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Shared configuration and official scoring adapters for the Qwen3-1.7B robustness analysis.

All experiment-specific code lives in ``analysis/robustness``.  The scoring
adapters call the repository metric implementations (``src/metric``), so
truncated prefixes and sampled generations use the same evaluators as the main results.
"""

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from glr_paths import get  # noqa: E402

ROBUSTNESS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path(get("analysis_dir")) / "robustness" / "qwen3-1.7B"
TEST_SET_DIR = Path(get("eval_dir"))

GLR_CHECKPOINT = Path(get("ckpt_dir")) / "glr_qwen3-1.7B"
SFT_CHECKPOINT = Path(get("ckpt_dir")) / "cot_sft_qwen3-1.7B"

SEEDS = (17, 29, 43)
ANALYSIS_SEED = 20260726
N_BOOTSTRAP = 10_000
K_BY_METHOD = {"cot_sft": 0, "glr_k10": 10}
ASSISTANT_TURN_DELIMITER = "<|im_end|>\n<|im_start|>assistant\n"
# Correctness field per benchmark and method, as in plots/plot_benchmark.py.
# SVAMP uses numeric exact match for CoT-SFT and math_verify for GLR-10.
CORRECTNESS_FIELDS: dict[str, dict[str, str]] = {
    "gsm8k": {"cot_sft": "correct", "glr_k10": "correct"},
    "math500": {
        "cot_sft": "math_verify_correct",
        "glr_k10": "math_verify_correct",
    },
    "svamp": {
        "cot_sft": "exact_match_correct",
        "glr_k10": "math_verify_correct",
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    split: str
    expected_count: int
    constrained_budget: int
    full_cap: int

    @property
    def path(self) -> Path:
        return TEST_SET_DIR / self.filename


DATASETS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        "gsm8k", "gsm8k_zeroshot.jsonl", "test", 1319, 256, 2048
    ),
    "math500": DatasetSpec(
        "math500", "minerva_math500_zeroshot.jsonl", "test", 500, 512, 4096
    ),
    "svamp": DatasetSpec("svamp", "svamp.jsonl", "test", 300, 256, 2048),
}

# Scored greedy generations (scripts/score/score_decodes.sh).
GREEDY_CACHE_FILES: dict[str, dict[str, Path]] = {
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

REQUIRED_RECORD_FIELDS = (
    "dataset",
    "dataset_revision",
    "split",
    "example_id",
    "prompt_hash",
    "method",
    "model_size",
    "checkpoint_path_or_hash",
    "K",
    "decoding",
    "sample_id",
    "seed",
    "total_step_budget",
    "latent_steps",
    "text_steps",
    "total_steps",
    "generated_token_ids",
    "generated_text",
    "gold_answer",
    "predicted_answer",
    "valid_answer",
    "correct",
    "hit_cap",
    "stopped_on_eos",
    "repetition_flag",
    "runtime_seconds",
    "source",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield value


def load_dataset(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    spec = DATASETS[name]
    rows = list(read_jsonl(spec.path))
    if len(rows) != spec.expected_count:
        raise ValueError(
            f"{name}: expected {spec.expected_count} rows in {spec.path}, got {len(rows)}"
        )
    ids = [str(row.get("id", idx)) for idx, row in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate example IDs in {spec.path}")
    return rows if limit is None else rows[:limit]


def checkpoint_for(method: str) -> Path:
    return SFT_CHECKPOINT if method == "cot_sft" else GLR_CHECKPOINT


@lru_cache(maxsize=None)
def dataset_revision(name: str) -> str:
    return f"sha256:{sha256_file(DATASETS[name].path)}"


def repetition_flag(token_ids: list[int]) -> bool:
    """Conservative token diagnostic from section 7 of the specification."""
    run = 0
    previous: int | None = None
    for token in token_ids:
        if token == previous:
            run += 1
        else:
            previous = token
            run = 1
        if run >= 20:
            return True

    tail = token_ids[-256:]
    if len(tail) >= 32:
        counts: dict[tuple[int, ...], int] = {}
        for start in range(len(tail) - 7):
            block = tuple(tail[start : start + 8])
            counts[block] = counts.get(block, 0) + 1
            if counts[block] >= 4:
                return True
    return False


def extract_final_assistant_turn(text: str | None) -> str:
    """Apply the post-question extraction used for the figures (plots/plot_common.py).

    The terminal ``<|im_end|>`` is intentionally retained.  It is a generated
    token in the step accounting, whereas the question/prompt
    preceding the final assistant turn is excluded.
    """
    if not text:
        return ""
    if ASSISTANT_TURN_DELIMITER in text:
        output = text.split(ASSISTANT_TURN_DELIMITER)[-1]
    elif "<|im_start|>assistant\n" in text:
        output = text.split("<|im_start|>assistant\n")[-1]
    else:
        output = text
    output = output.replace("<|im_start|>assistant", "")
    return output.strip()


def _load_metric_module(module_name: str):
    metric_dir = REPO_ROOT / "src" / "metric"
    repo_str = str(REPO_ROOT)
    metric_str = str(metric_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    if metric_str not in sys.path:
        sys.path.insert(0, metric_str)
    return importlib.import_module(module_name)


def score_generation(
    dataset: str, generated_text: str, gold_answer: str
) -> tuple[str | None, bool, bool]:
    """Return (canonical prediction, valid extraction, official correctness)."""
    if dataset == "gsm8k":
        metric = _load_metric_module("gsm8k_accuracy")
        pred = metric.extract_gsm8k_final_answer(
            generated_text, mode="flexible-extract"
        )
        pred_norm = metric._normalize_exact_match(
            pred,
            ignore_case=True,
            regexes_to_ignore=metric.DEFAULT_REGEXES_TO_IGNORE,
        )
        gold_norm = metric._normalize_exact_match(
            gold_answer,
            ignore_case=True,
            regexes_to_ignore=metric.DEFAULT_REGEXES_TO_IGNORE,
        )
        pred_num = metric._parse_numeric(pred_norm)
        gold_num = metric._parse_numeric(gold_norm)
        valid = pred != "[invalid]" and pred_num is not None
        if pred_num is not None and gold_num is not None:
            try:
                # Decimal avoids OverflowError for pathological repetitive
                # outputs containing integers with thousands of digits while
                # preserving the official evaluator's 1e-9 tolerance.
                correct = (
                    abs(Decimal(pred_norm) - Decimal(gold_norm))
                    < Decimal("1e-9")
                )
            except InvalidOperation:
                correct = False
        else:
            correct = valid and pred_norm == gold_norm
        return (pred_norm if valid else None), bool(valid), bool(correct)

    if dataset == "svamp":
        metric = _load_metric_module("amc23_score")
        correct, pred_raw, pred_norm, _ = metric._exact_match_from_candidates(
            generated_text, gold_answer, extraction_mode="flexible-extract"
        )
        valid = pred_raw not in ("[invalid]", "[invalidanswer]") and pred_norm not in (
            "[invalid]",
            "[invalidanswer]",
        )
        return (pred_norm if valid else None), bool(valid), bool(correct)

    if dataset == "math500":
        metric = _load_metric_module("minerva_math500_score")
        pred_raw = metric.extract_minerva_final_answer(generated_text)
        pred_norm = metric.normalize_final_answer(pred_raw)
        gold_norm = metric.normalize_final_answer(gold_answer)
        valid = pred_raw != "[invalidanswer]" and pred_norm != "[invalidanswer]"
        correct = valid and metric._is_equiv_latex(pred_norm, gold_norm)
        return (pred_norm if valid else None), bool(valid), bool(correct)

    raise KeyError(f"Unsupported dataset: {dataset}")


def answers_equivalent(dataset: str, left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        # Missing/unfinished answers are abstentions, not a voteable answer
        # category.  In particular, two cap-reaching trajectories must not
        # manufacture an incorrect "majority" by both mapping to None.
        return False
    if dataset != "math500":
        return left == right
    metric = _load_metric_module("minerva_math500_score")
    return bool(metric._is_equiv_latex(left, right))


def majority_vote(
    dataset: str, answers: list[str | None], correctness: list[bool]
) -> tuple[bool, bool]:
    """Return (consensus_exists, voted_answer_is_correct) for exactly 3 samples."""
    if len(answers) != 3 or len(correctness) != 3:
        raise ValueError("Majority@3 requires exactly three samples")
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if answers_equivalent(dataset, answers[left], answers[right]):
            # Equivalent answers have the same official correctness except for
            # pathological evaluator timeouts. Recheck both labels defensively.
            return True, bool(correctness[left] or correctness[right])
    return False, False


def make_record(
    *,
    dataset: str,
    row: dict[str, Any],
    method: str,
    decoding: str,
    sample_id: int,
    seed: int | None,
    total_step_budget: int,
    token_ids: list[int],
    generated_text: str,
    hit_cap: bool,
    stopped_on_eos: bool,
    runtime_seconds: float | None,
    source: str,
) -> dict[str, Any]:
    k = K_BY_METHOD[method]
    predicted, valid, correct = score_generation(
        dataset, generated_text, str(row["a"])
    )
    record = {
        "dataset": dataset,
        "dataset_revision": dataset_revision(dataset),
        "split": DATASETS[dataset].split,
        "example_id": str(row.get("id")),
        "prompt_hash": sha256_text(str(row["q"])),
        "method": method,
        "model_size": "1.7B",
        "checkpoint_path_or_hash": str(checkpoint_for(method)),
        "K": k,
        "decoding": decoding,
        "sample_id": sample_id,
        "seed": seed,
        "total_step_budget": total_step_budget,
        # For GLR, the K latent rollout positions are already represented in
        # the post-prompt generated sequence. ``latent_steps`` records the
        # subset size; it must not be added to ``text_steps`` a second time.
        "latent_steps": k,
        "text_steps": len(token_ids),
        "total_steps": len(token_ids),
        "generated_token_ids": [int(value) for value in token_ids],
        "generated_text": generated_text,
        "gold_answer": str(row["a"]),
        "predicted_answer": predicted,
        "valid_answer": valid,
        "correct": correct,
        "hit_cap": bool(hit_cap),
        "stopped_on_eos": bool(stopped_on_eos),
        "repetition_flag": repetition_flag(token_ids),
        "runtime_seconds": runtime_seconds,
        "source": source,
    }
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Record missing fields: {missing}")
    if record["total_steps"] != record["text_steps"]:
        raise ValueError(
            "total_steps != post-prompt tokenizer length (text_steps)"
        )
    if record["text_steps"] != len(record["generated_token_ids"]):
        raise ValueError("text_steps != len(generated_token_ids)")
    if record["total_steps"] > record["total_step_budget"]:
        raise ValueError(
            f"total steps {record['total_steps']} exceed budget "
            f"{record['total_step_budget']}"
        )
    expected_k = K_BY_METHOD[record["method"]]
    if record["latent_steps"] != expected_k or record["K"] != expected_k:
        raise ValueError(f"Invalid K/latent_steps for {record['method']}")
