"""Post-hoc, paired GSM8K evaluation for the R1 full-model JSONL output.

No answer/gold is used by generation, candidate selection or the Telos runtime
gate. This script is intentionally separate from the inference runner.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean
import tempfile

from experiments.swn_r1.teloi import EVALUATION_TELOI
from src.metric.gsm8k_accuracy import score_file


REFERENCE_PAIRS = (
    ("swn", "glr"),
    ("random", "glr"),
    ("swn_cayley", "glr"),
    ("orientation_cayley", "glr"),
    ("swn_cayley", "orientation_cayley"),
    ("glr_2pass", "glr"),
    ("swn_2pass", "glr_2pass"),
    ("random_2pass", "glr_2pass"),
    ("swn_cayley_2pass", "glr_2pass"),
    ("orientation_cayley_2pass", "glr_2pass"),
    ("swn_cayley_2pass", "orientation_cayley_2pass"),
)


def summarize(path: Path, *, n_bootstrap: int = 1000, seed: int = 2026) -> dict:
    rows: dict[str, list] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["arm"]].append(r)
    if not rows:
        raise ValueError("empty output")
    sample_sets = {arm: {str(r["id"]) for r in recs} for arm, recs in rows.items()}
    if len({frozenset(v) for v in sample_sets.values()}) != 1:
        raise ValueError("arms must have identical sample id sets (paired evaluation)")
    scored = {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm, records in rows.items():
            ids = [str(r["id"]) for r in records]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicated sample id in arm " + arm)
            records = sorted(records, key=lambda r: str(r["id"]))
            local = Path(tmp) / (arm + ".jsonl")
            local.write_text("\n".join(json.dumps({
                "predicted": r["predicted"], "answer": r["answer"],
            }) for r in records) + "\n", encoding="utf-8")
            results, accuracy = score_file(local)
            if len(results) != len(records):
                raise ValueError("scorer did not return every record")
            metrics = {
                "accuracy": float(accuracy),
                "mean_steps": float(mean(r["generated_steps"] for r in records)),
                "mean_seconds": float(mean(r["wall_seconds"] for r in records)),
                "fallbacks_per_example": float(mean(r["fallbacks"] for r in records)),
            }
            scored[arm] = {
                "n": len(records), "metrics": metrics,
                "correct_by_id": {str(r["id"]): bool(v.correct)
                                  for r, v in zip(records, results)},
            }
    pairs = []
    rng = random.Random(seed)
    for proposed, baseline in REFERENCE_PAIRS:
        if proposed not in scored or baseline not in scored:
            continue
        x, y = scored[proposed], scored[baseline]
        ids = sorted(x["correct_by_id"])
        d = [(int(x["correct_by_id"][i]) - int(y["correct_by_id"][i]))
             for i in ids]
        mean_diff = sum(d) / len(d)
        boot = sorted(sum(rng.choices(d, k=len(d))) / len(d)
                      for _ in range(n_bootstrap)) if n_bootstrap else []
        ci = [boot[int(.025 * (len(boot)-1))],
              boot[int(.975 * (len(boot)-1))]] if boot else None
        pairs.append({
            "proposed": proposed, "control": baseline,
            "teloi": EVALUATION_TELOI.compare(x["metrics"], y["metrics"]),
            "paired_accuracy_delta": mean_diff,
            "bootstrap_accuracy_delta_95pct": ci,
            "correctness_gains": sum(1 for e in d if e == 1),
            "correctness_losses": sum(1 for e in d if e == -1),
            "correctness_ties": sum(1 for e in d if e == 0),
        })
    return {
        "metric": "GSM8K official repository exact-match scorer",
        "sample_scope": "observed inputs only; never proof of generalization",
        "arms": {k: {"n": v["n"], **v["metrics"]} for k, v in scored.items()},
        "paired_comparisons": pairs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    if args.bootstrap < 0:
        p.error("bootstrap must be nonnegative")
    result = summarize(args.predictions, n_bootstrap=args.bootstrap)
    args.summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
