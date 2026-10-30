#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Apply the Qwen3 chat template to Mixture-of-Thoughts math and count tokens.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

inp_dataset=$(python glr_paths.py mot_raw_dir)/math
out_dataset=$(python glr_paths.py mot_tokenized_dir)
tokenizer_path=$(python glr_paths.py qwen3_0.6b)

include_system_prompt=False


python src/data/process_mixture_of_thoughts_math.py \
  --out_dataset ${out_dataset} \
  --tokenizer_path ${tokenizer_path} \
  --dataset_inp_dir ${inp_dataset} \
  --split train \
  --num_proc 8 \
  --include_system_prompt ${include_system_prompt}
