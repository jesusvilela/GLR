#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Score the generations in decode_dir and write annotated copies (per-example correctness) to scored_dir.
# Optional arguments select model folders (default: every folder in decode_dir), e.g.
#   bash scripts/score/score_decodes.sh glr_qwen3-1.7B cot_sft_qwen3-1.7B
# Output: scored_dir/<model>/<benchmark>.jsonl (CoT-SFT) and scored_dir/<model>/<benchmark>_latent<K>.jsonl (GLR).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
source scripts/decode/benchmarks.sh

declare -A SCORER=(
  [gsm8k_zeroshot]=gsm8k_accuracy.py
  [minerva_math500_zeroshot]=minerva_math500_score.py
  [svamp]=svamp_score.py
  [multiarith]=multiarith_score.py
  [amc23]=amc23_score.py
  [olympiadbench_oe_to_maths_en_comp_train]=olympiadbench_oe_to_maths_en_comp_score.py
)

decode_dir=$(python glr_paths.py decode_dir)
scored_dir=$(python glr_paths.py scored_dir)
eval_dir=$(python glr_paths.py eval_dir)

# Print the number of examples in the test set and in the prediction file.
report_counts() {
  local n_test n_pred
  n_test=$(wc -l < ${eval_dir}/$1.jsonl)
  n_pred=$(wc -l < $2)
  echo "           test set: ${n_test} examples, predictions: ${n_pred} examples"
  if [[ ${n_pred} -ne ${n_test} ]]; then
    echo "           WARNING: prediction file is incomplete or does not match the test set" >&2
  fi
}

if [[ $# -gt 0 ]]; then
  models=("$@")
else
  models=()
  for d in ${decode_dir}/{cot_sft,glr}_*/; do
    [[ -d ${d} ]] && models+=("$(basename ${d})")
  done
fi

shopt -s nullglob
for model in "${models[@]}"; do
  mkdir -p ${scored_dir}/${model}
  case ${model} in
    cot_sft_*)
      for benchmark in "${BENCHMARKS[@]}"; do
        for pred_file in ${decode_dir}/${model}/${benchmark}_*.jsonl; do
          echo "========> ${model} ${benchmark}"
          report_counts ${benchmark} ${pred_file}
          python src/metric/${SCORER[${benchmark}]} \
            --pred-key decoded_output \
            --gold-solution-file ${eval_dir}/${benchmark}.jsonl \
            --pred-file ${pred_file} \
            --save-file ${scored_dir}/${model}/${benchmark}.jsonl
        done
      done
      ;;
    glr_*)
      for pred_file in ${decode_dir}/${model}/*_latent*_gen*_temp0.0.json; do
        name=$(basename ${pred_file})
        benchmark=${name%%_latent*}
        latent=${name#*_latent}
        latent=${latent%%_*}
        echo "========> ${model} ${benchmark} K=${latent}"
        report_counts ${benchmark} ${pred_file}
        python src/metric/${SCORER[${benchmark}]} \
          --pred-file ${pred_file} \
          --save-file ${scored_dir}/${model}/${benchmark}_latent${latent}.jsonl
      done
      ;;
    *)
      echo "Skipping ${model}: not a cot_sft_* or glr_* folder" >&2
      ;;
  esac
done
