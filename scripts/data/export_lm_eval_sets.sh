#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Export the GSM8K and MATH500 eval sets with lm-evaluation-harness, as originally done for the paper.
# scripts/data/prepare_eval_sets.sh builds the same files without the harness (see data/eval/SHA256SUMS);
# this script documents where the prompts come from and overwrites the two files in eval_dir.
#
# Requires lm-evaluation-harness at commit 03c44ad with vLLM, ideally in a separate environment:
#   pip install "lm-eval[vllm] @ git+https://github.com/EleutherAI/lm-evaluation-harness@03c44adc0586f88bb343a74da1a1c602103536dd"
# The two custom tasks in lm_eval_tasks/ are copied into the installed harness (minerva_math500_zeroshot
# includes the harness's own minerva_math_algebra.yaml). Generations are not used, only the logged prompts.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

model_path=$(python glr_paths.py qwen3_0.6b)
eval_dir=$(python glr_paths.py eval_dir)
out_dir=$(python glr_paths.py decode_dir)/lm_eval_export

lm_eval_tasks_dir=$(python -c "import lm_eval, os; print(os.path.join(os.path.dirname(lm_eval.__file__), 'tasks'))")
cp lm_eval_tasks/gsm8k/gsm8k_zeroshot.yaml ${lm_eval_tasks_dir}/gsm8k/
cp lm_eval_tasks/minerva_math/minerva_math500_zeroshot.yaml ${lm_eval_tasks_dir}/minerva_math/

model_args="pretrained="
model_args+="${model_path},"
model_args+="tokenizer=${model_path},"
model_args+="dtype=bfloat16,"
model_args+="enable_thinking=True,max_model_len=32768"

VLLM_WORKER_MULTIPROC_METHOD=spawn lm_eval --model vllm \
  --model_args ${model_args} \
  --tasks gsm8k_zeroshot,minerva_math500_zeroshot \
  --batch_size auto --apply_chat_template \
  --output_path ${out_dir} \
  --log_samples \
  --gen_kwargs max_gen_toks=256

for task in gsm8k_zeroshot minerva_math500_zeroshot; do
  samples=$(ls -t ${out_dir}/*/samples_${task}_*.jsonl | head -n 1)
  python src/data/dump_test_sets_from_lm_eval.py \
    --inp_path ${samples} \
    --out_path ${eval_dir}/${task}.jsonl
done
