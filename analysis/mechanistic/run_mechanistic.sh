#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Mechanistic analysis of Qwen3-1.7B GLR-10 on GSM8K, MATH500, and SVAMP:
#   1. teacher-force the CoT-SFT traces through the GLR checkpoint and replay the GLR-10 latent trajectories;
#   2. causal controls on 100 fixed examples per benchmark: zero transition, repeated <think>,
#      norm-matched random head, and noisy transition (sigma = 1);
#   3. alignment progress, CoT-embedding drift, FutureHit@10, logistic regression, tables, and figure;
#   4. accuracy and generation length of the controls with the benchmark evaluators.
# Requires the scored generations of cot_sft_qwen3-1.7B and glr_qwen3-1.7B (scripts/score/score_decodes.sh)
# and one GPU. Every step resumes from the cache. Output: analysis_dir/mechanistic.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common.sh"

cache_root=$(python glr_paths.py analysis_dir)/mechanistic
control_n=${CONTROL_N:-100}
common=(--cache-root ${cache_root} --resume-existing --verify-existing)

python analysis/mechanistic/initialize_cache.py --cache-root ${cache_root}
python analysis/mechanistic/ingest_existing_decodings.py --output-dir ${cache_root}/rendered

for dataset in gsm8k math500 svamp; do
  echo "========> ${dataset}: CoT-SFT reference and GLR-10 replay"
  python analysis/mechanistic/cache_replays.py \
    --dataset ${dataset} --task reference "${common[@]}"
  python analysis/mechanistic/cache_replays.py \
    --dataset ${dataset} --task learned --k 10 "${common[@]}"

  echo "========> ${dataset}: causal controls (N=${control_n})"
  for method in zero_transition repeated_think random_head_norm_matched; do
    python analysis/mechanistic/cache_trajectories.py \
      --dataset ${dataset} --method ${method} --k 10 \
      --subset-size ${control_n} --subset-seed 2026 "${common[@]}"
  done
  python analysis/mechanistic/cache_trajectories.py \
    --dataset ${dataset} --method noisy_transition --k 10 \
    --noise-sigma 1.0 --subset-size ${control_n} --subset-seed 2026 \
    "${common[@]}"
done

echo "========> metrics, tables, and figure"
python analysis/mechanistic/compute_metrics.py \
  --cache-root ${cache_root} \
  --datasets gsm8k math500 svamp \
  --nearest-neighbors 10 \
  --future-window 32 \
  --nn-device ${NN_DEVICE:-cuda} \
  --output-dir ${cache_root}/derived
python analysis/mechanistic/render_outputs.py \
  --cache-root ${cache_root} \
  --results-dir ${cache_root}/derived \
  --output-dir ${cache_root}/rendered \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 2026
python analysis/mechanistic/validate_cache.py --cache-root ${cache_root}

echo "========> causal controls with the benchmark evaluators"
python analysis/mechanistic/score_controls.py --result-root ${cache_root}
