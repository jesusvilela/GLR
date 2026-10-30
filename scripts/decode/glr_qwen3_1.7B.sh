#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Decoding of the GLR Qwen3-1.7B model: K latent steps, then token decoding.
# Optional arguments select benchmarks (default: all six), e.g. `bash scripts/decode/glr_qwen3_1.7B.sh svamp amc23`.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
source scripts/decode/benchmarks.sh
select_benchmarks "$@"

model_name=glr_qwen3-1.7B
model_path=$(python glr_paths.py ckpt_dir)/${model_name}
out_dir=$(python glr_paths.py decode_dir)/${model_name}

# Latent steps K evaluated in the paper.
declare -A latent_token_budgets=(
  [gsm8k_zeroshot]="5 10 20 50 80 100"
  [minerva_math500_zeroshot]="5 10 20 50 80 100"
  [svamp]="10"
  [multiarith]="10"
  [amc23]="10"
  [olympiadbench_oe_to_maths_en_comp_train]="10"
)

for benchmark in "${SELECTED[@]}"; do
  echo "========> ${benchmark}: ${model_name}, K = ${latent_token_budgets[${benchmark}]}, max ${MAX_NEW_TOKENS[${benchmark}]} steps"

  python src/inference/generate_glr.py \
    --model_path ${model_path} \
    --out_dir ${out_dir} \
    --latent_token_budget ${latent_token_budgets[${benchmark}]} \
    --max_gen_tokens ${MAX_NEW_TOKENS[${benchmark}]} \
    --sampling_temp 0.0 \
    --all_test_set_paths ${benchmark}.jsonl
done
