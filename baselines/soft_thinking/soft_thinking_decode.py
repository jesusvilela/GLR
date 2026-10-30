#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Decode CoT-SFT Qwen3 checkpoints with the Soft Thinking baseline.

This decoder does not use the GLR model or its transition head.  It performs an
ordinary prompt prefill, waits for the model's discrete ``<think>`` marker, and
then feeds probability-weighted input embeddings through the same KV-cached
Qwen3 backbone.

Each output is JSONL and retains the three fields written by
``src/inference/generate_glr.py`` (``question``, ``answer``, and ``predicted``),
followed by per-example Soft Thinking metadata.  Existing valid rows are skipped
on reruns.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from glr_paths import get  # noqa: E402

TEST_SET_DIR = Path(get("eval_dir"))

# Score with the repository evaluators (src/metric) through the robustness adapter.
from analysis.robustness.robustness_common import repetition_flag, score_generation  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    expected_count: int
    total_step_budget: int

    @property
    def path(self) -> Path:
        return TEST_SET_DIR / self.filename


DATASETS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec("gsm8k_zeroshot.jsonl", 1319, 2048),
    "svamp": DatasetSpec("svamp.jsonl", 300, 2048),
    "math500": DatasetSpec("minerva_math500_zeroshot.jsonl", 500, 4096),
}

MODE_NAMES = {
    "greedy": "ordinary_greedy_cot_sft",
    "fixed": "soft_thinking_fixed_k",
    "until_close": "soft_thinking_until_close",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-size", choices=("0.6B", "1.7B"), required=True)
    parser.add_argument("--fixed-k", type=int, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_NAMES),
        default=["fixed", "until_close"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(get("baseline_dir")) / "soft_thinking",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k-filter", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.001)
    parser.add_argument("--mix-k", type=int, default=10)
    parser.add_argument("--soft-cap", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Diagnostic only: decode the first N rows of each dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace selected output files instead of resuming them.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help=(
            "Do not regenerate model-level summaries after this worker. "
            "Used by parallel benchmark shards to avoid summary write races."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Regenerate summaries from existing JSONL without loading a model.",
    )
    args = parser.parse_args()
    if args.skip_summary and args.summary_only:
        parser.error("--skip-summary and --summary-only are mutually exclusive")
    if args.fixed_k < 0:
        parser.error("--fixed-k must be non-negative")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k_filter < 1:
        parser.error("--top-k-filter must be at least 1")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if not 0 <= args.min_p <= 1:
        parser.error("--min-p must be in [0, 1]")
    if args.mix_k < 1:
        parser.error("--mix-k must be at least 1")
    if args.soft_cap < 1:
        parser.error("--soft-cap must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield row


def load_dataset(name: str, limit: int | None) -> list[dict[str, Any]]:
    spec = DATASETS[name]
    rows = list(read_jsonl(spec.path))
    if len(rows) != spec.expected_count:
        raise ValueError(
            f"{name}: expected {spec.expected_count} rows in {spec.path}, "
            f"found {len(rows)}"
        )
    ids = [str(row.get("id", index)) for index, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate example IDs")
    return rows if limit is None else rows[:limit]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def marker_ids(tokenizer: Any, marker: str) -> list[int]:
    ids = [int(value) for value in tokenizer.encode(marker, add_special_tokens=False)]
    if not ids:
        raise ValueError(f"Tokenizer produced no IDs for {marker!r}")
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    if decoded != marker:
        raise ValueError(
            f"Tokenizer marker round trip failed for {marker!r}: "
            f"ids={ids}, decoded={decoded!r}"
        )
    return ids


def contains_subsequence(values: list[int], pattern: list[int]) -> bool:
    if not pattern or len(values) < len(pattern):
        return False
    return any(
        values[start : start + len(pattern)] == pattern
        for start in range(len(values) - len(pattern) + 1)
    )


def suffix_matches(values: list[int], pattern: list[int]) -> bool:
    return len(values) >= len(pattern) and values[-len(pattern) :] == pattern


def eos_ids(model: Any, tokenizer: Any) -> set[int]:
    configured = model.generation_config.eos_token_id
    if configured is None:
        configured = tokenizer.eos_token_id
    if isinstance(configured, int):
        return {int(configured)}
    return {int(value) for value in (configured or [])}


def past_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    # Legacy tuple cache: [layer][key/value], key shape is B,H,S,D.
    return int(past_key_values[0][0].shape[-2])


def filtered_mixture(
    logits: torch.Tensor,
    input_embedding: Any,
    *,
    temperature: float,
    top_k_filter: int,
    top_p: float,
    min_p: float,
    mix_k: int,
) -> tuple[torch.Tensor, list[int], list[float]]:
    """Apply the v2 filters in order and return one weighted input embedding.

    Selecting top-k logits before softmax is algebraically identical to taking
    top-k from the full softmax and then renormalizing, while avoiding a dense
    full-vocabulary probability tensor.
    """

    scores = logits.reshape(-1).float() / temperature
    filter_k = min(top_k_filter, int(scores.numel()))
    selected_scores, selected_ids = torch.topk(
        scores, k=filter_k, largest=True, sorted=True
    )
    probabilities = torch.softmax(selected_scores, dim=-1)

    if top_p < 1:
        cumulative = probabilities.cumsum(dim=-1)
        # Keep the first element whose inclusion crosses top_p.
        nucleus_mask = cumulative - probabilities < top_p
        selected_ids = selected_ids[nucleus_mask]
        probabilities = probabilities[nucleus_mask]
        probabilities = probabilities / probabilities.sum()

    if min_p > 0:
        min_mask = probabilities >= min_p * probabilities.max()
        selected_ids = selected_ids[min_mask]
        probabilities = probabilities[min_mask]
        probabilities = probabilities / probabilities.sum()

    final_k = min(mix_k, int(probabilities.numel()))
    probabilities, final_indices = torch.topk(
        probabilities, k=final_k, largest=True, sorted=True
    )
    selected_ids = selected_ids[final_indices]
    probabilities = probabilities / probabilities.sum()

    token_embeddings = input_embedding(selected_ids)
    # Accumulate in fp32, then feed the model in its embedding dtype.
    mixture = torch.sum(token_embeddings.float() * probabilities[:, None], dim=0).to(
        dtype=token_embeddings.dtype
    )
    return (
        mixture.reshape(1, 1, -1),
        [int(value) for value in selected_ids.tolist()],
        [float(value) for value in probabilities.tolist()],
    )


class SoftThinkingDecoder:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        temperature: float,
        top_k_filter: int,
        top_p: float,
        min_p: float,
        mix_k: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k_filter = top_k_filter
        self.top_p = top_p
        self.min_p = min_p
        self.mix_k = mix_k
        self.embedding = model.get_input_embeddings()
        self.device = self.embedding.weight.device
        self.think_start_ids = marker_ids(tokenizer, "<think>")
        self.think_close_ids = marker_ids(tokenizer, "</think>")
        self.stop_ids = eos_ids(model, tokenizer)

    def _forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        query_length = int(
            input_ids.shape[-1]
            if input_ids is not None
            else inputs_embeds.shape[-2]  # type: ignore[union-attr]
        )
        old_length = past_length(past_key_values)
        cache_position = torch.arange(
            old_length,
            old_length + query_length,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones(
            (1, old_length + query_length),
            dtype=torch.long,
            device=self.device,
        )
        outputs = self.model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        final_hidden = outputs.last_hidden_state[:, -1:, :]
        logits = self.model.lm_head(final_hidden)[:, -1, :]
        return logits, outputs.past_key_values

    def _token_embedding(self, token_id: int) -> torch.Tensor:
        ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        return self.embedding(ids)

    def _soft_embedding(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, list[int], list[float]]:
        return filtered_mixture(
            logits,
            self.embedding,
            temperature=self.temperature,
            top_k_filter=self.top_k_filter,
            top_p=self.top_p,
            min_p=self.min_p,
            mix_k=self.mix_k,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        mode: str,
        fixed_k: int,
        soft_cap: int,
        total_step_budget: int,
    ) -> dict[str, Any]:
        if mode not in MODE_NAMES:
            raise ValueError(mode)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(self.device)
        prompt_token_ids = [int(value) for value in prompt_ids[0].tolist()]

        started = time.perf_counter()
        logits, past = self._forward(input_ids=prompt_ids)
        generated_ids: list[int] = []
        soft_steps = 0
        soft_support_sizes: list[int] = []
        soft_phase_started = suffix_matches(prompt_token_ids, self.think_start_ids)
        soft_phase_done = mode == "greedy"
        soft_phase_reason = (
            "not_applicable" if mode == "greedy" else "think_start_not_reached"
        )
        soft_cap_reached = False
        natural_soft_closure = False
        pending_close_prefix = 0
        stopped_on_eos = False
        overall_reason = "max_total_step_budget"

        while len(generated_ids) + soft_steps < total_step_budget:
            # Fixed-K uses exactly K mixtures after <think>, before choosing
            # the first resumed textual token.
            if mode == "fixed" and soft_phase_started and not soft_phase_done:
                if soft_steps >= fixed_k:
                    soft_phase_done = True
                    soft_phase_reason = "fixed_k_complete"
                else:
                    mixture, support, _weights = self._soft_embedding(logits)
                    soft_support_sizes.append(len(support))
                    logits, past = self._forward(
                        inputs_embeds=mixture, past_key_values=past
                    )
                    soft_steps += 1
                    continue

            # Until-close checks the unfiltered greedy prediction before any
            # distribution filtering. Qwen3 uses one close-marker ID. The
            # prefix state also implements the specified deterministic
            # multi-ID fallback for other compatible tokenizers.
            if mode == "until_close" and soft_phase_started and not soft_phase_done:
                if soft_steps >= soft_cap:
                    soft_cap_reached = True
                    soft_phase_done = True
                    soft_phase_reason = "soft_think_cap_reached"
                else:
                    greedy_id = int(torch.argmax(logits, dim=-1).item())
                    expected_close_id = self.think_close_ids[pending_close_prefix]
                    if greedy_id == expected_close_id:
                        generated_ids.append(greedy_id)
                        pending_close_prefix += 1
                        if pending_close_prefix == len(self.think_close_ids):
                            natural_soft_closure = True
                            soft_phase_done = True
                            soft_phase_reason = "predicted_think_close"
                            pending_close_prefix = 0
                        if len(generated_ids) + soft_steps >= total_step_budget:
                            overall_reason = "max_total_step_budget"
                            break
                        logits, past = self._forward(
                            inputs_embeds=self._token_embedding(greedy_id),
                            past_key_values=past,
                        )
                        continue
                    if pending_close_prefix:
                        # A partial multi-token close prefix was already
                        # appended discretely. Resume soft reasoning from the
                        # logits after that unmatched prefix.
                        pending_close_prefix = 0
                        if greedy_id == self.think_close_ids[0]:
                            continue
                    mixture, support, _weights = self._soft_embedding(logits)
                    soft_support_sizes.append(len(support))
                    logits, past = self._forward(
                        inputs_embeds=mixture, past_key_values=past
                    )
                    soft_steps += 1
                    continue

            # Ordinary greedy text decoding: before <think>, after fixed K,
            # after natural closure, or as the safety-cap fallback.
            next_token = int(torch.argmax(logits, dim=-1).item())
            generated_ids.append(next_token)
            if next_token in self.stop_ids:
                stopped_on_eos = True
                overall_reason = "eos_or_end_of_turn"
                break
            if len(generated_ids) + soft_steps >= total_step_budget:
                overall_reason = "max_total_step_budget"
                break

            logits, past = self._forward(
                inputs_embeds=self._token_embedding(next_token),
                past_key_values=past,
            )

            if (
                mode != "greedy"
                and not soft_phase_started
                and suffix_matches(
                    prompt_token_ids + generated_ids, self.think_start_ids
                )
            ):
                soft_phase_started = True
                soft_phase_reason = "soft_phase_active"
                if mode == "fixed" and fixed_k == 0:
                    soft_phase_done = True
                    soft_phase_reason = "fixed_k_complete"

        if mode != "greedy" and not soft_phase_started:
            soft_phase_reason = "think_start_not_reached"

        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        predicted = self.tokenizer.decode(
            prompt_token_ids + generated_ids, skip_special_tokens=False
        )
        close_generated = contains_subsequence(generated_ids, self.think_close_ids)
        if soft_cap_reached:
            overall_reason = f"soft_think_cap_reached_then_{overall_reason}"

        return {
            "generated_token_ids": generated_ids,
            "generated_text": completion,
            "predicted": predicted,
            "num_soft_steps": soft_steps,
            "num_generated_textual_tokens": len(generated_ids),
            "total_sequential_steps": soft_steps + len(generated_ids),
            "thinking_started": soft_phase_started,
            "thinking_closed_naturally": close_generated,
            "natural_soft_closure": natural_soft_closure,
            "soft_think_cap_reached": soft_cap_reached,
            "soft_phase_termination_reason": soft_phase_reason,
            "termination_reason": overall_reason,
            "stopped_on_eos": stopped_on_eos,
            "soft_mixture_support_min": (
                min(soft_support_sizes) if soft_support_sizes else None
            ),
            "soft_mixture_support_max": (
                max(soft_support_sizes) if soft_support_sizes else None
            ),
            "runtime_seconds": time.perf_counter() - started,
        }


def output_path(
    output_dir: Path,
    *,
    model_size: str,
    dataset: str,
    mode: str,
    fixed_k: int,
    soft_cap: int,
) -> Path:
    model_slug = f"qwen3-{model_size}"
    suffix = {
        "greedy": "greedy",
        "fixed": f"softthink_fixed_k{fixed_k}",
        "until_close": f"softthink_until_close_kmax{soft_cap}",
    }[mode]
    return output_dir / model_slug / f"{dataset}_{suffix}.jsonl"


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: list[str] = []
    for row in read_jsonl(path):
        if row.get("status") == "complete":
            ids.append(str(row["example_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate completed example IDs")
    return set(ids)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def make_record(
    *,
    row: dict[str, Any],
    dataset: str,
    dataset_revision: str,
    model_path: Path,
    model_size: str,
    mode: str,
    fixed_k: int,
    soft_cap: int,
    total_step_budget: int,
    result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    extracted, valid, correct = score_generation(
        dataset, result["predicted"], str(row["a"])
    )
    relative_checkpoint = (
        str(model_path.resolve().relative_to(REPO_ROOT.resolve()))
        if model_path.resolve().is_relative_to(REPO_ROOT.resolve())
        else str(model_path.resolve())
    )
    return {
        # Same fields as src/inference/generate_glr.py.
        "question": str(row["q"]),
        "answer": str(row["a"]),
        "predicted": result["predicted"],
        # Identity and reproducibility.
        "status": "complete",
        "model_size": model_size,
        "checkpoint_identifier": relative_checkpoint,
        "benchmark": dataset,
        "dataset_revision": f"sha256:{dataset_revision}",
        "example_id": str(row.get("id")),
        "inference_mode": MODE_NAMES[mode],
        "K": fixed_k if mode == "fixed" else (0 if mode == "greedy" else None),
        "K_max": soft_cap if mode == "until_close" else None,
        "total_step_budget": total_step_budget,
        "decoding": "greedy",
        "soft_temperature": args.temperature,
        "top_k_filter": args.top_k_filter,
        "p_nucleus": args.top_p,
        "p_min": args.min_p,
        "k_mix": args.mix_k,
        # Text, score, and accounting.
        "generated_text": result["generated_text"],
        "generated_token_ids": result["generated_token_ids"],
        "extracted_answer": extracted,
        "valid_answer": valid,
        "correct": correct,
        "correctness": correct,
        "num_soft_steps": result["num_soft_steps"],
        "num_generated_textual_tokens": result["num_generated_textual_tokens"],
        "total_sequential_steps": result["total_sequential_steps"],
        "thinking_started": result["thinking_started"],
        "thinking_closed_naturally": result["thinking_closed_naturally"],
        "natural_soft_closure": result["natural_soft_closure"],
        "soft_think_cap_reached": result["soft_think_cap_reached"],
        "soft_phase_termination_reason": result["soft_phase_termination_reason"],
        "termination_reason": result["termination_reason"],
        "stopped_on_eos": result["stopped_on_eos"],
        "hit_total_step_budget": (
            result["termination_reason"].endswith("max_total_step_budget")
        ),
        "soft_mixture_support_min": result["soft_mixture_support_min"],
        "soft_mixture_support_max": result["soft_mixture_support_max"],
        "repetition_flag": repetition_flag(result["generated_token_ids"]),
        "runtime_seconds": result["runtime_seconds"],
    }


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {model_path}")
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, local_files_only=True
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the full decoder. Run it in the project GPU "
            "environment or use --help for argument validation."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": torch.cuda.current_device()},
    ).eval()
    model.requires_grad_(False)
    model.generation_config.use_cache = True
    transition = getattr(model.model, "direction_delta_predictor", None)
    if transition is not None:
        raise ValueError(
            f"{model_path}: CoT-SFT baseline unexpectedly has a GLR transition head"
        )
    return tokenizer, model


def write_model_summary(args: argparse.Namespace) -> None:
    """Regenerate the requested per-model accuracy-efficiency table."""

    rows: list[dict[str, Any]] = []
    model_slug = f"qwen3-{args.model_size}"
    for mode in args.modes:
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        combined: list[dict[str, Any]] = []
        for dataset in args.datasets:
            path = output_path(
                args.output_dir,
                model_size=args.model_size,
                dataset=dataset,
                mode=mode,
                fixed_k=args.fixed_k,
                soft_cap=args.soft_cap,
            )
            records = list(read_jsonl(path)) if path.exists() else []
            by_dataset[dataset] = records
            combined.extend(records)

        naturally_closed = [
            record for record in combined if bool(record["natural_soft_closure"])
        ]
        rows.append(
            {
                "method": MODE_NAMES[mode],
                "K": (
                    args.fixed_k
                    if mode == "fixed"
                    else (args.soft_cap if mode == "until_close" else 0)
                ),
                "gsm8k_accuracy": (
                    sum(bool(row["correct"]) for row in by_dataset["gsm8k"])
                    / len(by_dataset["gsm8k"])
                    if "gsm8k" in by_dataset and by_dataset["gsm8k"]
                    else None
                ),
                "svamp_accuracy": (
                    sum(bool(row["correct"]) for row in by_dataset["svamp"])
                    / len(by_dataset["svamp"])
                    if "svamp" in by_dataset and by_dataset["svamp"]
                    else None
                ),
                "math500_accuracy": (
                    sum(bool(row["correct"]) for row in by_dataset["math500"])
                    / len(by_dataset["math500"])
                    if "math500" in by_dataset and by_dataset["math500"]
                    else None
                ),
                "mean_soft_steps": (
                    statistics.fmean(int(row["num_soft_steps"]) for row in combined)
                    if combined
                    else None
                ),
                "mean_text_tokens": (
                    statistics.fmean(
                        int(row["num_generated_textual_tokens"]) for row in combined
                    )
                    if combined
                    else None
                ),
                "mean_total_steps": (
                    statistics.fmean(
                        int(row["total_sequential_steps"]) for row in combined
                    )
                    if combined
                    else None
                ),
                "natural_closure_rate": (
                    len(naturally_closed) / len(combined)
                    if mode == "until_close" and combined
                    else None
                ),
                "natural_closure_mean_soft_steps": (
                    statistics.fmean(
                        int(row["num_soft_steps"]) for row in naturally_closed
                    )
                    if naturally_closed
                    else None
                ),
                "natural_closure_median_soft_steps": (
                    statistics.median(
                        int(row["num_soft_steps"]) for row in naturally_closed
                    )
                    if naturally_closed
                    else None
                ),
                "num_examples": len(combined),
                "num_examples_by_benchmark": {
                    dataset: len(records) for dataset, records in by_dataset.items()
                },
            }
        )

    model_dir = args.output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_size": args.model_size,
        "checkpoint_identifier": str(args.model_path),
        "rows": rows,
    }
    json_path = model_dir / "summary.json"
    json_temporary = json_path.with_suffix(".json.tmp")
    json_temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    json_temporary.replace(json_path)

    def display(value: Any) -> str:
        return "—" if value is None else f"{float(value):.4f}"

    markdown_lines = [
        f"# Soft Thinking v2: Qwen3-{args.model_size}",
        "",
        (
            "| Method | GSM8K Acc. | SVAMP Acc. | MATH500 Acc. | "
            "Mean Soft Steps | Mean Text Tokens | Mean Total Steps | "
            "Natural Closure Rate |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                (
                    str(row["method"]),
                    display(row["gsm8k_accuracy"]),
                    display(row["svamp_accuracy"]),
                    display(row["math500_accuracy"]),
                    display(row["mean_soft_steps"]),
                    display(row["mean_text_tokens"]),
                    display(row["mean_total_steps"]),
                    display(row["natural_closure_rate"]),
                )
            )
            + " |"
        )
    markdown_lines.extend(
        [
            "",
            (
                "The JSON summary additionally records the mean and median "
                "soft-step counts among naturally closed until-close examples."
            ),
            "",
        ]
    )
    markdown_path = model_dir / "summary.md"
    markdown_temporary = markdown_path.with_suffix(".md.tmp")
    markdown_temporary.write_text("\n".join(markdown_lines), encoding="utf-8")
    markdown_temporary.replace(markdown_path)
    print(f"Wrote summaries: {json_path}, {markdown_path}", flush=True)


def main() -> None:
    args = parse_args()
    expected_k = {"0.6B": 20, "1.7B": 10}[args.model_size]
    if args.fixed_k != expected_k:
        raise ValueError(
            f"The requested experiment fixes K={expected_k} for "
            f"{args.model_size}, received --fixed-k {args.fixed_k}"
        )

    if args.summary_only:
        write_model_summary(args)
        return

    tokenizer, model = load_model(args)
    decoder = SoftThinkingDecoder(
        model=model,
        tokenizer=tokenizer,
        temperature=args.temperature,
        top_k_filter=args.top_k_filter,
        top_p=args.top_p,
        min_p=args.min_p,
        mix_k=args.mix_k,
    )
    print(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "model_size": args.model_size,
                "datasets": args.datasets,
                "modes": args.modes,
                "fixed_k": args.fixed_k,
                "soft_cap": args.soft_cap,
                "think_start_ids": decoder.think_start_ids,
                "think_close_ids": decoder.think_close_ids,
                "eos_ids": sorted(decoder.stop_ids),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for dataset in args.datasets:
        spec = DATASETS[dataset]
        rows = load_dataset(dataset, args.limit)
        revision = sha256_file(spec.path)
        for mode in args.modes:
            destination = output_path(
                args.output_dir,
                model_size=args.model_size,
                dataset=dataset,
                mode=mode,
                fixed_k=args.fixed_k,
                soft_cap=args.soft_cap,
            )
            if args.overwrite and destination.exists():
                destination.unlink()
            done = completed_ids(destination)
            pending = [
                row
                for index, row in enumerate(rows)
                if str(row.get("id", index)) not in done
            ]
            print(
                f"{args.model_size} {dataset} {mode}: "
                f"{len(done)} complete, {len(pending)} pending -> {destination}",
                flush=True,
            )
            for index, row in enumerate(
                tqdm(pending, desc=f"{args.model_size}:{dataset}:{mode}")
            ):
                result = decoder.generate(
                    str(row["q"]),
                    mode=mode,
                    fixed_k=args.fixed_k,
                    soft_cap=args.soft_cap,
                    total_step_budget=spec.total_step_budget,
                )
                record = make_record(
                    row=row,
                    dataset=dataset,
                    dataset_revision=revision,
                    model_path=args.model_path,
                    model_size=args.model_size,
                    mode=mode,
                    fixed_k=args.fixed_k,
                    soft_cap=args.soft_cap,
                    total_step_budget=spec.total_step_budget,
                    result=result,
                    args=args,
                )
                if record["total_sequential_steps"] > record["total_step_budget"]:
                    raise AssertionError("Decoder exceeded total step budget")
                append_jsonl(destination, record)
            print(f"Completed {destination}", flush=True)
    if not args.skip_summary:
        write_model_summary(args)


if __name__ == "__main__":
    main()
