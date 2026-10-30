#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Train the Coconut-style hidden-state replacement (HSR) baseline on Qwen3-0.6B (one H100 80GB).
# Hyperparameters match the GLR training scripts (scripts/train/glr_qwen3_0.6B.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common.sh"

base_model=$(python glr_paths.py qwen3_0.6b)
train_file_path=$(python glr_paths.py mot_train_10k_dir)
save_dir=$(python glr_paths.py ckpt_dir)/coconut_hsr_qwen3-0.6B

mkdir -p ${save_dir}
export PYTHONUNBUFFERED=1

python -m baselines.coconut_hsr.train_hsr \
  --block_size=8192 \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps=16 \
  --dataloader_num_workers=4 \
  --dataset_num_proc=4 \
  --num_train_epochs=5 \
  --max_steps=-1 \
  --train_file_path="${train_file_path}" \
  --model_name="${base_model}" \
  --warmup_ratio=0.05 \
  --bf16=True \
  --eval_strategy="no" \
  --logging_steps=10 \
  --save_strategy="steps" \
  --save_steps=100 \
  --save_total_limit=2 \
  --lr_scheduler_type="cosine" \
  --learning_rate=4e-5 \
  --weight_decay=1e-4 \
  --adam_beta1=0.9 \
  --adam_beta2=0.95 \
  --output_dir="${save_dir}" \
  --push_to_hub=False \
  --save_only_model=True \
  --gradient_checkpointing=True \
  --freeze_input_embeddings=True \
  "$@" 2>&1 | tee ${save_dir}/train.log
