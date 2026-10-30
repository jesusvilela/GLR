# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Shared configuration and cache helpers for the 1.7B GLR analysis.

It uses the repository metric implementations (``src/metric``) unchanged.
"""

import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from glr_paths import get  # noqa: E402

TEST_SET_DIR = Path(get("eval_dir"))
DEFAULT_CACHE_ROOT = Path(get("analysis_dir")) / "mechanistic"

GLR_CHECKPOINT = Path(get("ckpt_dir")) / "glr_qwen3-1.7B"
SFT_CHECKPOINT = Path(get("ckpt_dir")) / "cot_sft_qwen3-1.7B"
ABS_CHECKPOINT_DEFAULT = Path(get("ckpt_dir")) / "absolute_prediction"
SCORED_DIR = Path(get("scored_dir"))

# K=10 is the main-paper setting and the only setting in this analysis.
K_VALUES = (10,)
K_STAR = 10
BOOTSTRAP_SEED = 2026
CONTROL_SEEDS = {
    "shuffled_transition": (0, 1, 2),
    "matched_norm_random_direction": (0, 1, 2, 3, 4),
    "random_head_norm_matched": (0, 1, 2),
    "noisy_transition": (0, 1, 2, 3, 4),
}
NOISE_SIGMAS = (0.25, 0.5, 1.0)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    expected_count: int
    cap: int
    sft_decode: str
    glr_decode_dir: str
    glr_k10_decode: str

    @property
    def path(self) -> Path:
        return TEST_SET_DIR / self.filename


DATASETS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        "gsm8k",
        "gsm8k_zeroshot.jsonl",
        1319,
        2048,
        str(SCORED_DIR / "cot_sft_qwen3-1.7B" / "gsm8k_zeroshot.jsonl"),
        str(SCORED_DIR / "glr_qwen3-1.7B"),
        str(SCORED_DIR / "glr_qwen3-1.7B" / "gsm8k_zeroshot_latent10.jsonl"),
    ),
    "math500": DatasetSpec(
        "math500",
        "minerva_math500_zeroshot.jsonl",
        500,
        4096,
        str(SCORED_DIR / "cot_sft_qwen3-1.7B" / "minerva_math500_zeroshot.jsonl"),
        str(SCORED_DIR / "glr_qwen3-1.7B"),
        str(SCORED_DIR / "glr_qwen3-1.7B" / "minerva_math500_zeroshot_latent10.jsonl"),
    ),
    "svamp": DatasetSpec(
        "svamp",
        "svamp.jsonl",
        300,
        2048,
        str(SCORED_DIR / "cot_sft_qwen3-1.7B" / "svamp.jsonl"),
        str(SCORED_DIR / "glr_qwen3-1.7B"),
        str(SCORED_DIR / "glr_qwen3-1.7B" / "svamp_latent10.jsonl"),
    ),
}

METHOD_LABELS = {
    "learned": "GLR learned transition",
    "zero_transition": "Zero transition",
    "repeated_think": "Repeated `<think>` embedding",
    "shuffled_transition": "Shuffled learned transition",
    "matched_norm_random_direction": "Matched-norm random direction",
    "random_head_norm_matched": "Norm-matched random head",
    "noisy_transition": "Noisy transition",
    "absolute_prediction": "Absolute embedding predictor",
}

DELIMITER = "<|im_end|>\n<|im_start|>assistant\n"
EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: expected JSON object")
            yield row


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    spec = DATASETS[name]
    rows = list(read_jsonl(spec.path))
    if len(rows) != spec.expected_count:
        raise ValueError(f"{name}: expected {spec.expected_count}, found {len(rows)}")
    ids = [str(row.get("id", i)) for i, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate example IDs")
    return rows if limit is None else rows[:limit]


def extract_assistant_turn(text: str | None) -> str:
    """Match the token-count extraction used for the figures (plots/plot_common.py)."""
    if not text:
        return ""
    if DELIMITER in text:
        text = text.split(DELIMITER)[-1]
    elif "<|im_start|>assistant\n" in text:
        text = text.split("<|im_start|>assistant\n")[-1]
    return text.replace("<|im_start|>assistant", "").strip()


def generation_finished(text: str | None) -> bool:
    """Match the figures (plots/plot_common.py): a generation cut off at the decoding limit did not finish."""
    return bool(text) and text.endswith(EOS_TOKENS)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def supplied_correct(row: dict[str, Any]) -> bool:
    """Generic fallback for logs whose benchmark-specific key is unknown."""
    for key in (
        "correct",
        "is_correct",
        "math_verify_correct",
        "exact_match_correct",
    ):
        if key in row:
            value = parse_bool(row[key])
            if value is not None:
                return value
    return False


def marked_correct(
    dataset: str, row: dict[str, Any], method: str = "glr"
) -> tuple[bool, str]:
    """Return the correctness marker used for the figures.

    ``method`` is either ``glr`` or ``sft``.  The distinction matters only for
    SVAMP: ``math_verify_correct`` is used for GLR and
    ``exact_match_correct`` for CoT-SFT.
    """
    if dataset == "gsm8k":
        candidates = ("correct", "is_correct")
    elif dataset == "math500":
        candidates = ("math_verify_correct",)
    elif dataset == "svamp" and method == "sft":
        candidates = ("exact_match_correct",)
    elif dataset == "svamp":
        candidates = ("math_verify_correct",)
    else:
        raise KeyError(dataset)
    for key in candidates:
        if key not in row:
            continue
        value = parse_bool(row[key])
        if value is not None:
            return value, key
    raise KeyError(
        f"{dataset}/{method}: missing correctness key(s) {candidates}"
    )


def marked_control_correct(
    dataset: str, row: dict[str, Any]
) -> tuple[bool, str]:
    """Correctness marker comparable to dependency-free control scoring.

    The fresh controls use the repository's exact/numeric equivalence scorer.
    Learned results therefore use their exact-match annotation for this
    secondary comparison, while the main mechanistic analysis uses the marker
    returned by :func:`marked_correct`.
    """
    candidates = (
        ("correct", "is_correct")
        if dataset == "gsm8k"
        else ("exact_match_correct",)
    )
    for key in candidates:
        if key in row:
            value = parse_bool(row[key])
            if value is not None:
                return value, key
    raise KeyError(f"{dataset}: missing control correctness key(s) {candidates}")


def decode_path(dataset: str, method: str, k: int | None = None) -> Path:
    """Resolve one of the supplied line-aligned greedy decoding files."""
    spec = DATASETS[dataset]
    if method == "sft":
        relative = spec.sft_decode
    elif method == "glr":
        if k is None:
            raise ValueError("GLR decode path requires k")
        if dataset == "gsm8k":
            relative = f"{spec.glr_decode_dir}/gsm8k_zeroshot_latent{k}.jsonl"
        elif dataset == "math500":
            relative = f"{spec.glr_decode_dir}/minerva_math500_zeroshot_latent{k}.jsonl"
        elif dataset == "svamp" and k == K_STAR:
            relative = spec.glr_k10_decode
        else:
            raise FileNotFoundError(
                f"No supplied {dataset} GLR K={k} decode is in scope"
            )
    else:
        raise ValueError(f"unknown decode method: {method}")
    path = REPO_ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_metric_module(name: str):
    metric_dir = REPO_ROOT / "src" / "metric"
    for path in (REPO_ROOT, metric_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module(name)


def score_generation(
    dataset: str, generated_text: str, gold_answer: str
) -> tuple[str | None, bool, bool]:
    """Use the same extraction/equivalence code as the main evaluation."""
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
            # Control decodes can emit pathological integer strings with
            # thousands of digits. Converting those integers to float raises
            # OverflowError. Decimal preserves the evaluator's
            # absolute 1e-9 tolerance while making such predictions safely
            # score as incorrect rather than aborting the cache job.
            try:
                difference = abs(
                    Decimal(str(pred_num)) - Decimal(str(gold_num))
                )
                correct = difference < Decimal("1e-9")
            except (InvalidOperation, ValueError):
                correct = False
        else:
            correct = valid and pred_norm == gold_norm
        return (pred_norm if valid else None), bool(valid), bool(correct)
    if dataset == "svamp":
        metric = _load_metric_module("amc23_score")
        correct, pred_raw, pred_norm, _ = metric._exact_match_from_candidates(
            generated_text, gold_answer, extraction_mode="flexible-extract"
        )
        invalid = {"[invalid]", "[invalidanswer]"}
        valid = pred_raw not in invalid and pred_norm not in invalid
        return (pred_norm if valid else None), bool(valid), bool(correct)
    if dataset == "math500":
        metric = _load_metric_module("minerva_math500_score")
        pred_raw = metric.extract_minerva_final_answer(generated_text)
        pred_norm = metric.normalize_final_answer(pred_raw)
        gold_norm = metric.normalize_final_answer(gold_answer)
        valid = pred_raw != "[invalidanswer]" and pred_norm != "[invalidanswer]"
        return (
            pred_norm if valid else None,
            bool(valid),
            bool(valid and metric._is_equiv_latex(pred_norm, gold_norm)),
        )
    raise KeyError(dataset)


def cache_run_dir(
    cache_root: Path,
    dataset: str,
    method: str,
    k: int,
    seed: int | None = None,
    noise_sigma: float | None = None,
    forced: bool = False,
) -> Path:
    if method == "reference":
        return cache_root / "references" / dataset / "cot_sft_glr_encoded"
    path = cache_root / "runs" / dataset / method / f"K={k}"
    if noise_sigma is not None:
        path = path / f"sigma={noise_sigma:g}"
    path = path / f"seed={0 if seed is None else seed}"
    if forced:
        path = path / "forced_continuation"
    return path


def expected_ids(dataset: str) -> list[str]:
    return [str(row.get("id", i)) for i, row in enumerate(load_dataset(dataset))]
