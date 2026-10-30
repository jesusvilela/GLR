#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# All paper plots from the scored generations: every model on every benchmark, and the K=0 ablation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/common.sh"
source scripts/decode/benchmarks.sh

for model in 0.6B 1.7B; do
  for benchmark in "${BENCHMARKS[@]}"; do
    echo "========> Qwen3-${model} ${benchmark}"
    python plots/plot_benchmark.py --model ${model} --benchmark ${benchmark}
  done
done

echo "========> K=0 ablation (Qwen3-1.7B)"
python plots/plot_k0_ablation.py
