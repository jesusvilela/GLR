#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Train the CoT-SFT baseline on Qwen3-0.6B (one H100 80GB).
# Input embeddings are frozen by default; FREEZE_INPUT_EMBEDDINGS=False trains the unfrozen-embedding variant.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

export PYTHONUNBUFFERED=1

freeze_input_embeddings=${FREEZE_INPUT_EMBEDDINGS:-True}

base_model=$(python glr_paths.py qwen3_0.6b)
train_file_path=$(python glr_paths.py mot_train_10k_dir)
save_dir=$(python glr_paths.py ckpt_dir)/cot_sft_qwen3-0.6B
if [[ "${freeze_input_embeddings}" == "False" ]]; then
  save_dir=${save_dir}_embed_unfreeze
fi

mkdir -p ${save_dir}

lr=4e-5
epochs=5
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=16
push_to_hub=false
block_size=8192

python \
    src/train/train.py \
    --block_size=${block_size} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --dataloader_num_workers=4 \
    --dataset_num_proc=4 \
    --num_train_epochs=${epochs} \
    --train_file_path=${train_file_path} \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=10 \
    --save_strategy="steps" \
    --save_steps=100 \
    --save_total_limit=2 \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir=${save_dir} \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --enable_latent_reason=False \
    --latent_reason_type="directional" \
    --latent_reason_discount_factor=0.999 \
    --latent_reason_pred_delta_embed=False \
    --latent_reason_teacher_forcing=False \
    --latent_reason_criteria="mse" \
    --freeze_input_embeddings=${freeze_input_embeddings} \
    "$@" 2>&1 | tee ${save_dir}/train.log
