#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# K=0 ablation: the GLR Qwen3-1.7B model decoded without latent steps (greedy).
# Written to its own folder so it is kept apart from the main K grid.
# vLLM version of glr_qwen3_1.7B_k0_ablation.sh (vllm_glr_qwen3/): same settings and output files, much faster.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
source scripts/decode/benchmarks.sh
if [[ $# -eq 0 ]]; then set -- gsm8k_zeroshot minerva_math500_zeroshot; fi
select_benchmarks "$@"

model_name=glr_qwen3-1.7B
model_path=$(python glr_paths.py ckpt_dir)/${model_name}
out_dir=$(python glr_paths.py decode_dir)/${model_name}_k0_ablation

for benchmark in "${SELECTED[@]}"; do
  echo "========> ${benchmark}: ${model_name}, K = 0, max ${MAX_NEW_TOKENS[${benchmark}]} steps"

  python vllm_glr_qwen3/generate.py \
    --model_path ${model_path} \
    --out_dir ${out_dir} \
    --latent_token_budget 0 \
    --max_gen_tokens ${MAX_NEW_TOKENS[${benchmark}]} \
    --sampling_temp 0.0 \
    --all_test_set_paths ${benchmark}.jsonl
done
