# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Plan balanced, resumable shards of sampled-inference tasks."""

import argparse
import json
from pathlib import Path
from typing import Any

from robustness_common import DATASETS, DEFAULT_OUTPUT_DIR, SEEDS, read_jsonl

METHODS = ("cot_sft", "glr_k10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--max-shards", type=int, default=5)
    parser.add_argument("--write-json", type=Path)
    return parser.parse_args()


def validate_existing_prefix(
    path: Path,
    *,
    dataset: str,
    method: str,
    sample_id: int,
    seed: int,
) -> int:
    if not path.exists():
        return 0
    source_rows = DATASETS[dataset].expected_count
    rows = list(read_jsonl(path))
    if len(rows) > source_rows:
        raise ValueError(
            f"{path}: has {len(rows)} rows; expected at most {source_rows}"
        )
    seen: set[str] = set()
    for row in rows:
        if row.get("dataset") != dataset or row.get("method") != method:
            raise ValueError(f"{path}: dataset/method metadata mismatch")
        if int(row.get("sample_id", -1)) != sample_id:
            raise ValueError(f"{path}: sample_id metadata mismatch")
        if int(row.get("seed", -1)) != seed:
            raise ValueError(f"{path}: seed metadata mismatch")
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"{path}: duplicate example_id {example_id}")
        seen.add(example_id)
    return len(rows)


def pending_tasks(
    output_dir: Path, datasets: list[str], methods: list[str]
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for dataset in datasets:
        spec = DATASETS[dataset]
        for method in methods:
            for sample_id, seed in enumerate(SEEDS):
                path = (
                    output_dir
                    / "raw"
                    / f"sampled_{dataset}_{method}_sample{sample_id}.jsonl"
                )
                completed = validate_existing_prefix(
                    path,
                    dataset=dataset,
                    method=method,
                    sample_id=sample_id,
                    seed=seed,
                )
                remaining = spec.expected_count - completed
                if remaining:
                    tasks.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "sample_id": sample_id,
                            "seed": seed,
                            "completed_examples": completed,
                            "remaining_examples": remaining,
                            "max_remaining_steps": remaining * spec.constrained_budget,
                            "spec": f"{dataset}:{method}:{sample_id}:{seed}",
                        }
                    )
    return tasks


def balance(tasks: list[dict[str, Any]], max_shards: int) -> list[dict[str, Any]]:
    if not tasks:
        return []
    count = min(max_shards, len(tasks))
    shards = [
        {"shard_id": index, "max_remaining_steps": 0, "tasks": []}
        for index in range(count)
    ]
    for task in sorted(
        tasks,
        key=lambda item: (-int(item["max_remaining_steps"]), str(item["spec"])),
    ):
        target = min(
            shards,
            key=lambda shard: (
                int(shard["max_remaining_steps"]),
                len(shard["tasks"]),
                int(shard["shard_id"]),
            ),
        )
        target["tasks"].append(task)
        target["max_remaining_steps"] += int(task["max_remaining_steps"])
    return shards


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_shards <= 5:
        raise ValueError("--max-shards must be between 1 and 5")
    output_dir = args.output_dir.resolve()
    tasks = pending_tasks(output_dir, args.datasets, args.methods)
    shards = balance(tasks, args.max_shards)
    payload = {
        "output_dir": str(output_dir),
        "max_shards": args.max_shards,
        "pending_task_count": len(tasks),
        "shards": shards,
    }
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for shard in shards:
        specifications = " ".join(str(task["spec"]) for task in shard["tasks"])
        print(f"{shard['shard_id']}|{shard['max_remaining_steps']}|{specifications}")


if __name__ == "__main__":
    main()
