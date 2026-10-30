# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Evaluation benchmarks (files in eval_dir) and their maximum number of generation steps.
BENCHMARKS=(
  gsm8k_zeroshot
  minerva_math500_zeroshot
  svamp
  multiarith
  amc23
  olympiadbench_oe_to_maths_en_comp_train
)

declare -A MAX_NEW_TOKENS=(
  [gsm8k_zeroshot]=2048
  [svamp]=2048
  [multiarith]=2048
  [minerva_math500_zeroshot]=4096
  [amc23]=4096
  [olympiadbench_oe_to_maths_en_comp_train]=4096
)

# Benchmarks selected on the command line (default: all), validated against BENCHMARKS.
select_benchmarks() {
  if [[ $# -eq 0 ]]; then
    SELECTED=("${BENCHMARKS[@]}")
  else
    SELECTED=("$@")
  fi
  for benchmark in "${SELECTED[@]}"; do
    if [[ ! -v MAX_NEW_TOKENS[${benchmark}] ]]; then
      echo "Unknown benchmark '${benchmark}'. Choose from: ${BENCHMARKS[*]}" >&2
      exit 1
    fi
  done
}
