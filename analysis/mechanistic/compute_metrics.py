# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Compute alignment, drift, and semantic metrics from trajectory caches only."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from mechanistic_common import (
    DATASETS,
    DEFAULT_CACHE_ROOT,
    GLR_CHECKPOINT,
    load_dataset,
    score_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--nearest-neighbors", type=int, default=10)
    parser.add_argument("--future-window", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nn-device", default="cpu")
    parser.add_argument("--nn-vocab-chunk", type=int, default=8192)
    return parser.parse_args()


class ParquetSink:
    def __init__(self, path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.pq = pq
        self.path = path
        self.writer = None

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = self.pa.Table.from_pylist(rows)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = self.pq.ParquetWriter(
                self.path, table.schema, compression="zstd"
            )
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def iter_run(run_dir: Path) -> Iterator[tuple[dict[str, Any], dict[str, torch.Tensor]]]:
    for metadata_path in sorted(run_dir.glob("shard-*.metadata.jsonl")):
        tensor_path = metadata_path.with_name(
            metadata_path.name.replace(".metadata.jsonl", ".safetensors")
        )
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            rows = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
            ]
            for row in rows:
                prefix = row["tensor_prefix"]
                tensors = {
                    key[len(prefix) + 1 :]: handle.get_tensor(key)
                    for key in handle.keys()
                    if key.startswith(prefix + "_")
                }
                yield row, tensors


def load_reference(cache_root: Path, dataset: str):
    run_dir = cache_root / "references" / dataset / "cot_sft_glr_encoded"
    if not (run_dir / "_COMPLETE").exists():
        raise FileNotFoundError(f"incomplete reference cache: {run_dir}")
    return {
        str(row["example_id"]): (row, tensors) for row, tensors in iter_run(run_dir)
    }


def complete_run_dirs(cache_root: Path, dataset: str) -> list[Path]:
    root = cache_root / "runs" / dataset
    if not root.exists():
        return []
    return sorted(path.parent for path in root.rglob("_COMPLETE"))


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def monotonic_alignment(latent: np.ndarray, discrete: np.ndarray) -> np.ndarray:
    """O(Km) monotonic alignment with repeated discrete indices allowed."""
    if len(latent) == 0 or len(discrete) == 0:
        return np.empty((0,), dtype=np.int64)
    cost = 1.0 - normalize(latent) @ normalize(discrete).T
    k_size, m_size = cost.shape
    dp = cost[0].copy()
    back = np.zeros((k_size, m_size), dtype=np.int32)
    for k in range(1, k_size):
        prefix_value = np.empty(m_size, dtype=np.float64)
        prefix_index = np.empty(m_size, dtype=np.int32)
        best_value = float(dp[0])
        best_index = 0
        for j in range(m_size):
            if float(dp[j]) < best_value:
                best_value, best_index = float(dp[j]), j
            prefix_value[j] = best_value
            prefix_index[j] = best_index
        dp = cost[k] + prefix_value
        back[k] = prefix_index
    aligned = np.empty(k_size, dtype=np.int64)
    aligned[-1] = int(np.argmin(dp))
    for k in range(k_size - 1, 0, -1):
        aligned[k - 1] = int(back[k, aligned[k]])
    return aligned


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(normalize(left) * normalize(right), axis=-1)


def relative_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(left - right, axis=-1) / np.maximum(
        np.linalg.norm(right, axis=-1), 1e-12
    )


class VocabularyProjector:
    def __init__(
        self,
        checkpoint: Path,
        tokenizer: AutoTokenizer,
        top_k: int,
        device: str,
        chunk_size: int,
    ):
        tensor_path = checkpoint / "model.safetensors"
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            key = "model.embed_tokens.weight"
            if key not in handle.keys():
                candidates = [
                    name
                    for name in handle.keys()
                    if name.endswith("embed_tokens.weight")
                ]
                if len(candidates) != 1:
                    raise KeyError(
                        f"could not identify embedding matrix in {tensor_path}"
                    )
                key = candidates[0]
            self.embedding_cpu = handle.get_tensor(key).float()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self.requested_top_k = top_k
        self.top_k = top_k + len(self.special_ids)
        self.embedding_device = None
        if self.device.type == "cuda":
            weight = self.embedding_cpu.to(self.device)
            self.embedding_device = weight / weight.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

    def embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        return self.embedding_cpu[torch.as_tensor(token_ids, dtype=torch.long)].numpy()

    def project(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = torch.from_numpy(z).float().to(self.device)
        query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        best_values = torch.full(
            (query.shape[0], self.top_k), -torch.inf, device=self.device
        )
        best_ids = torch.full(
            (query.shape[0], self.top_k), -1, dtype=torch.long, device=self.device
        )
        source = (
            self.embedding_device
            if self.embedding_device is not None
            else self.embedding_cpu
        )
        for start in range(0, source.shape[0], self.chunk_size):
            weight = source[start : start + self.chunk_size]
            if self.embedding_device is None:
                weight = weight.to(self.device)
                weight = weight / weight.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores = query @ weight.T
            ids = (
                torch.arange(start, start + weight.shape[0], device=self.device)
                .view(1, -1)
                .expand(query.shape[0], -1)
            )
            values = torch.cat((best_values, scores), dim=1)
            indices = torch.cat((best_ids, ids), dim=1)
            best_values, order = torch.topk(values, self.top_k, dim=1)
            best_ids = torch.gather(indices, 1, order)
        return best_ids.cpu().numpy(), best_values.cpu().numpy()


def substantive_close(
    tokenizer: AutoTokenizer, ids: list[int], close_ids: set[int]
) -> bool:
    for token_id in ids:
        if tokenizer.decode([token_id], skip_special_tokens=False).strip() == "":
            continue
        return token_id in close_ids
    return False


def canonical_gold_text(dataset: str, value: str) -> str:
    if dataset == "gsm8k" and "####" in value:
        return value.rsplit("####", 1)[-1].strip()
    return value.strip()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.cache_root / "derived"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(GLR_CHECKPOINT)
    projector = VocabularyProjector(
        GLR_CHECKPOINT,
        tokenizer,
        args.nearest_neighbors,
        args.nn_device,
        args.nn_vocab_chunk,
    )
    trajectory_sink = ParquetSink(output_dir / "trajectory_steps.parquet")
    neighbor_sink = ParquetSink(output_dir / "nearest_neighbors.parquet")
    summary_sink = ParquetSink(output_dir / "per_example_summary.parquet")
    close_ids = set(int(x) for x in [151668])

    for dataset in args.datasets:
        references = load_reference(args.cache_root, dataset)
        gold_rows = {
            str(row.get("id", index)): row
            for index, row in enumerate(load_dataset(dataset))
        }
        token_counts: Counter[int] = Counter()
        for _, tensors in references.values():
            token_counts.update(int(x) for x in tensors["reasoning_ids"].tolist())
        frequency_total = max(sum(token_counts.values()), 1)

        for run_dir in complete_run_dirs(args.cache_root, dataset):
            for metadata, tensors in iter_run(run_dir):
                example_id = str(metadata["example_id"])
                noise_sigma = (
                    float(metadata["noise_sigma"])
                    if metadata.get("noise_sigma") is not None
                    else math.nan
                )
                if example_id not in references:
                    raise KeyError(
                        f"{dataset}/{example_id}: no GLR-encoded CoT-SFT reference"
                    )
                ref_meta, ref = references[example_id]
                h_discrete = ref["h_discrete"].float().numpy()
                reasoning_ids = ref["reasoning_ids"].numpy().astype(np.int64)
                if len(h_discrete) == 0:
                    continue
                z = tensors["z"].float().numpy()
                h_latent = tensors["h_latent"].float().numpy()
                delta = tensors["delta"].float().numpy()
                aligned = monotonic_alignment(h_latent, h_discrete)
                matched_h = h_discrete[aligned]
                matched_e = projector.embeddings(reasoning_ids[aligned])
                drift_e = cosine_distance(z, matched_e)
                drift_h = cosine_distance(h_latent, matched_h)
                l2_e = relative_l2(z, matched_e)
                l2_h = relative_l2(h_latent, matched_h)
                nn_ids, nn_values = projector.project(z)
                gold_tokens = set(
                    tokenizer.encode(
                        canonical_gold_text(dataset, str(gold_rows[example_id]["a"])),
                        add_special_tokens=False,
                    )
                )
                trajectory_rows: list[dict[str, Any]] = []
                neighbor_rows: list[dict[str, Any]] = []
                future_hits: list[float] = []
                future_recalls: list[float] = []
                future_hits_no_special: list[float] = []
                future_recalls_no_special: list[float] = []
                random_hits: list[float] = []
                for step in range(len(z)):
                    matched_at = int(aligned[step])
                    window = set(
                        int(x)
                        for x in reasoning_ids[
                            matched_at : matched_at + args.future_window
                        ].tolist()
                    )
                    ranked_all = [
                        (int(token_id), float(similarity))
                        for token_id, similarity in zip(nn_ids[step], nn_values[step])
                    ]
                    ranked_with_special = ranked_all[: args.nearest_neighbors]
                    ranked_no_special = [
                        item
                        for item in ranked_all
                        if item[0] not in projector.special_ids
                    ][: args.nearest_neighbors]
                    nearest = set(token_id for token_id, _ in ranked_with_special)
                    nearest_no_special = set(
                        token_id for token_id, _ in ranked_no_special
                    )
                    hit = float(bool(nearest & window))
                    recall = len(nearest & window) / max(len(window), 1)
                    hit_no_special = float(bool(nearest_no_special & window))
                    recall_no_special = len(nearest_no_special & window) / max(
                        len(window), 1
                    )
                    frequency_mass = (
                        sum(token_counts[token] for token in window) / frequency_total
                    )
                    random_hit = 1.0 - (1.0 - frequency_mass) ** args.nearest_neighbors
                    future_hits.append(hit)
                    future_recalls.append(recall)
                    future_hits_no_special.append(hit_no_special)
                    future_recalls_no_special.append(recall_no_special)
                    random_hits.append(random_hit)
                    trajectory_rows.append(
                        {
                            "example_id": example_id,
                            "dataset": dataset,
                            "method": metadata["method"],
                            "K": int(metadata["K"]),
                            "actual_latent_steps": int(
                                metadata.get("actual_latent_steps", len(z))
                            ),
                            "seed": int(metadata["seed"]),
                            "noise_sigma": noise_sigma,
                            "forced_continuation": bool(
                                metadata.get("forced_continuation", False)
                            ),
                            "k": step + 1,
                            "normalized_k": (step + 1) / len(z),
                            "normalized_step_bin": min(
                                4, math.ceil(4 * (step + 1) / len(z))
                            ),
                            "correct": bool(metadata["correct"]),
                            "control_correct": bool(
                                metadata.get("control_correct", metadata["correct"])
                            ),
                            "original_correct": bool(
                                metadata.get("original_correct", metadata["correct"])
                            ),
                            "correctness_key": str(
                                metadata.get("correctness_key", "recomputed")
                            ),
                            "discrete_thought_length": len(reasoning_ids),
                            "aligned_j": matched_at + 1,
                            "progress": (matched_at + 1) / len(reasoning_ids),
                            "cot_embedding_cosine_drift": float(drift_e[step]),
                            "hidden_cosine_drift": float(drift_h[step]),
                            "cot_embedding_relative_l2_drift": float(l2_e[step]),
                            "hidden_relative_l2_drift": float(l2_h[step]),
                            "latent_norm": float(np.linalg.norm(z[step])),
                            "transition_norm": float(np.linalg.norm(delta[step])),
                            "nearest_vocab_cosine": float(nn_values[step, 0]),
                            "nearest_vocab_distance": float(1.0 - nn_values[step, 0]),
                            "future_hit_at_10": hit,
                            "future_recall_at_10": float(recall),
                            "answer_hit_at_10": float(bool(nearest & gold_tokens)),
                            "future_hit_at_10_no_special": hit_no_special,
                            "future_recall_at_10_no_special": float(recall_no_special),
                            "answer_hit_at_10_no_special": float(
                                bool(nearest_no_special & gold_tokens)
                            ),
                            "frequency_random_future_hit_at_10": float(random_hit),
                        }
                    )
                    for variant, ranked in (
                        ("with_special", ranked_with_special),
                        ("no_special", ranked_no_special),
                    ):
                        for rank, (token_id, similarity) in enumerate(ranked, 1):
                            neighbor_rows.append(
                                {
                                    "example_id": example_id,
                                    "dataset": dataset,
                                    "method": metadata["method"],
                                    "K": int(metadata["K"]),
                                    "actual_latent_steps": int(
                                        metadata.get("actual_latent_steps", len(z))
                                    ),
                                    "seed": int(metadata["seed"]),
                                    "noise_sigma": noise_sigma,
                                    "k": step + 1,
                                    "normalized_k": (step + 1) / len(z),
                                    "projection_variant": variant,
                                    "rank": rank,
                                    "token_id": token_id,
                                    "token_string": tokenizer.decode(
                                        [token_id], skip_special_tokens=False
                                    ),
                                    "cosine_similarity": float(similarity),
                                    "is_special_token": token_id
                                    in projector.special_ids,
                                    "in_matched_future_window": token_id in window,
                                    "in_gold_answer_tokens": token_id in gold_tokens,
                                    "aligned_j": matched_at + 1,
                                    "cot_embedding_cosine_drift": float(drift_e[step]),
                                    "correct": bool(metadata["correct"]),
                                }
                            )
                trajectory_sink.write(trajectory_rows)
                neighbor_sink.write(neighbor_rows)

                resume_h = tensors["resume_h"].float().numpy()
                resume_alignment = monotonic_alignment(resume_h, h_discrete)
                resume_progress = (
                    float((resume_alignment[0] + 1) / len(reasoning_ids))
                    if len(resume_alignment)
                    else math.nan
                )
                resume_end = (
                    float((resume_alignment[-1] + 1) / len(reasoning_ids))
                    if len(resume_alignment)
                    else math.nan
                )
                resume_ids = [int(x) for x in tensors["resume_ids"].tolist()]
                immediate = substantive_close(tokenizer, resume_ids, close_ids)
                prefix_text = tokenizer.decode(
                    resume_ids[:16], skip_special_tokens=False
                )
                if metadata.get("trajectory_only_replay", False):
                    answer16_valid = False
                else:
                    _, answer16_valid, _ = score_generation(
                        dataset, prefix_text, str(gold_rows[example_id]["a"])
                    )
                if immediate:
                    category = "immediate_close"
                elif not math.isfinite(resume_progress):
                    category = "unmatched_or_too_short"
                elif resume_progress < 0.33:
                    category = "early_resume"
                elif resume_progress < 0.66:
                    category = "middle_resume"
                else:
                    category = "late_resume"
                generated_count = int(metadata["generated_token_count"])
                summary_sink.write(
                    [
                        {
                            "example_id": example_id,
                            "dataset": dataset,
                            "method": metadata["method"],
                            "K": int(metadata["K"]),
                            "actual_latent_steps": int(
                                metadata.get("actual_latent_steps", len(z))
                            ),
                            "seed": int(metadata["seed"]),
                            "noise_sigma": noise_sigma,
                            "correct": bool(metadata["correct"]),
                            "control_correct": bool(
                                metadata.get("control_correct", metadata["correct"])
                            ),
                            "original_correct": bool(
                                metadata.get("original_correct", metadata["correct"])
                            ),
                            "correctness_key": str(
                                metadata.get("correctness_key", "recomputed")
                            ),
                            "parsed_answer": str(metadata.get("parsed_answer") or ""),
                            "gold_answer": metadata["gold_answer"],
                            "total_steps": generated_count,
                            "text_tokens": max(
                                0,
                                generated_count
                                - int(metadata.get("actual_latent_steps", len(z))),
                            ),
                            "valid_answer": bool(metadata["valid_answer"]),
                            "truncated": bool(metadata["truncated"]),
                            "immediate_close": immediate,
                            "answer_within_16": bool(answer16_valid),
                            "final_progress": float(
                                (aligned[-1] + 1) / len(reasoning_ids)
                            ),
                            "final_cot_embedding_drift": float(drift_e[-1]),
                            "mean_cot_embedding_drift": float(drift_e.mean()),
                            "max_cot_embedding_drift": float(drift_e.max()),
                            "final_hidden_drift": float(drift_h[-1]),
                            "resume_progress": resume_progress,
                            "resume_end_progress": resume_end,
                            "resume_category": category,
                            "future_hit_at_10": float(np.mean(future_hits)),
                            "future_recall_at_10": float(np.mean(future_recalls)),
                            "future_hit_at_10_no_special": float(
                                np.mean(future_hits_no_special)
                            ),
                            "future_recall_at_10_no_special": float(
                                np.mean(future_recalls_no_special)
                            ),
                            "nearest_vocab_cosine": float(np.mean(nn_values[:, 0])),
                            "frequency_random_future_hit_at_10": float(
                                np.mean(random_hits)
                            ),
                            "forced_continuation": bool(
                                metadata.get("forced_continuation", False)
                            ),
                        }
                    ]
                )

    trajectory_sink.close()
    neighbor_sink.close()
    summary_sink.close()
    print(f"Derived cache written to {output_dir}")


if __name__ == "__main__":
    main()
