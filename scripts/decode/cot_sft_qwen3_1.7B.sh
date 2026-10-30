#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Decoding of the CoT-SFT Qwen3-1.7B model with vLLM.
# Optional arguments select benchmarks (default: all six), e.g. `bash scripts/decode/cot_sft_qwen3_1.7B.sh svamp amc23`.
# MODEL_NAME selects another checkpoint folder in ckpt_dir, e.g. MODEL_NAME=cot_sft_qwen3-1.7B_embed_unfreeze.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
source scripts/decode/benchmarks.sh
select_benchmarks "$@"

model_name=${MODEL_NAME:-cot_sft_qwen3-1.7B}
model_dir=$(python glr_paths.py ckpt_dir)/${model_name}
base_out_dir=$(python glr_paths.py decode_dir)/${model_name}
base_test_dir=$(python glr_paths.py eval_dir)

for benchmark in "${SELECTED[@]}"; do
  echo "========> ${benchmark}: ${model_name}, max ${MAX_NEW_TOKENS[${benchmark}]} new tokens"

  python src/inference/generate_vllm_qwen3.py \
    --input ${base_test_dir}/${benchmark}.jsonl \
    --out-dir ${base_out_dir} \
    --save-file ${benchmark} \
    --prompt-field q \
    --models ${model_dir} \
    --max-new-tokens ${MAX_NEW_TOKENS[${benchmark}]} \
    --trust-remote-code \
    --save-finish-reason
done
