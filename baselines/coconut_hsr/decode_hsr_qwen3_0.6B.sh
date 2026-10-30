#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Greedy decoding of the Coconut-style HSR baseline (Qwen3-0.6B) on GSM8K, SVAMP, and MATH500:
# fixed K = 10 and 100 hidden-state feedback steps, and feedback until the model emits </think>.
# Outputs: baseline_dir/coconut_hsr_qwen3-0.6B/*.jsonl (resumable; OVERWRITE=1 starts over).
# Environment overrides: DATASETS ("gsm8k svamp math500"), LIMIT, OVERWRITE.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common.sh"

read -r -a datasets <<< "${DATASETS:-gsm8k svamp math500}"
extra_args=()
if [[ -n "${LIMIT:-}" ]]; then
  extra_args+=(--limit "${LIMIT}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi

model_path=$(python glr_paths.py ckpt_dir)/coconut_hsr_qwen3-0.6B
output_dir=$(python glr_paths.py baseline_dir)/coconut_hsr_qwen3-0.6B
export PYTHONUNBUFFERED=1

python -u baselines/coconut_hsr/decode_hsr.py \
  --model-path ${model_path} \
  --output-dir ${output_dir} \
  --latent-steps 10 100 \
  --datasets "${datasets[@]}" \
  --modes fixed until_close \
  --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  "${extra_args[@]}"
