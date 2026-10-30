#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Statistical robustness of the Qwen3-1.7B constrained-budget comparison (CoT-SFT vs. GLR-10) on GSM8K, MATH500,
# and SVAMP:
#   greedy:  paired bootstrap confidence intervals (10,000 resamples), exact McNemar tests with Holm correction,
#            and generation-length reductions;
#   sampled: three trajectories per problem (temperature 0.6, top-p 0.95, seeds 17/29/43) at the 256/512/256-step
#            budgets: sampled pass@1, majority@3, and pass@3.
# Requires the scored greedy generations of cot_sft_qwen3-1.7B and glr_qwen3-1.7B (scripts/score/score_decodes.sh)
# and one GPU for the sampled trajectories. Resumable. Output: analysis_dir/robustness/qwen3-1.7B.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common.sh"

output_dir=$(python glr_paths.py analysis_dir)/robustness/qwen3-1.7B
datasets=(gsm8k math500 svamp)
methods=(cot_sft glr_k10)

python analysis/robustness/build_manifest.py --output-dir ${output_dir}

echo "========> greedy generations: full cap and constrained prefixes"
for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    python analysis/robustness/decode_robustness.py \
      --mode greedy_cache \
      --dataset ${dataset} \
      --method ${method} \
      --batch-size 1 \
      --output-dir ${output_dir}
  done
done

echo "========> sampled trajectories"
mapfile -t shard_lines < <(
  python analysis/robustness/plan_sampled_shards.py \
    --output-dir ${output_dir} \
    --datasets "${datasets[@]}" \
    --methods "${methods[@]}" \
    --max-shards 1
)
for shard_line in "${shard_lines[@]}"; do
  IFS='|' read -r shard_id max_remaining_steps task_specs <<< "${shard_line}"
  task_args=()
  for task_spec in ${task_specs}; do
    task_args+=(--task ${task_spec})
  done
  python analysis/robustness/build_manifest.py --runtime-update --output-dir ${output_dir}
  python analysis/robustness/decode_robustness.py \
    --mode sampled_bundle \
    --batch-size 1 \
    --output-dir ${output_dir} \
    "${task_args[@]}"
done

echo "========> statistics, tables, and figures"
python analysis/robustness/compute_statistics.py \
  --output-dir ${output_dir} \
  --bootstrap-replicates 10000 \
  --analysis-seed 20260726
python analysis/robustness/validate_outputs.py \
  --output-dir ${output_dir} \
  --compile-latex-if-available
