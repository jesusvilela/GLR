#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Soft Thinking baseline on the CoT-SFT checkpoints, on GSM8K, SVAMP, and MATH500:
# fixed K soft-token steps (K=20 for Qwen3-0.6B, K=10 for Qwen3-1.7B) and soft steps until </think> (K_max=256).
# Soft Thinking settings: tau=0.6, top-k 30, top-p 0.95, min-p 0.001, mix-k 10.
# Outputs: baseline_dir/soft_thinking/qwen3-{0.6B,1.7B}/*.jsonl (resumable; OVERWRITE=1 starts over).
#
# Optional arguments select model sizes (default: 0.6B 1.7B).
# Environment overrides: DATASETS ("gsm8k svamp math500"), MODES ("fixed until_close"), LIMIT, OVERWRITE.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common.sh"

read -r -a datasets <<< "${DATASETS:-gsm8k svamp math500}"
read -r -a modes <<< "${MODES:-fixed until_close}"
extra_args=()
if [[ -n "${LIMIT:-}" ]]; then
  extra_args+=(--limit "${LIMIT}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi

declare -A fixed_k=([0.6B]=20 [1.7B]=10)
ckpt_dir=$(python glr_paths.py ckpt_dir)
if [[ $# -eq 0 ]]; then set -- 0.6B 1.7B; fi
export PYTHONUNBUFFERED=1

for model_size in "$@"; do
  echo "========> Soft Thinking, CoT-SFT Qwen3-${model_size}, K=${fixed_k[${model_size}]}"
  python -u baselines/soft_thinking/soft_thinking_decode.py \
    --model-path ${ckpt_dir}/cot_sft_qwen3-${model_size} \
    --model-size ${model_size} \
    --fixed-k ${fixed_k[${model_size}]} \
    --datasets "${datasets[@]}" \
    --modes "${modes[@]}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    "${extra_args[@]}"
done
