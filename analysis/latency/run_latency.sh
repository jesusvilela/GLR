#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Matched-stack wall-clock latency and FLOPs of CoT-SFT and GLR-10 (Qwen3-1.7B): the same Hugging Face decoder,
# BF16, FlashAttention-2, KV caching, greedy decoding, batch size 1, on one H100 80GB GPU.
# Default benchmarks: MATH500 (cap 4,096) and SVAMP (cap 2,048); LATENCY_BENCHMARKS=gsm8k,math500,svamp adds GSM8K.
# Requires the scored generations of cot_sft_qwen3-1.7B and glr_qwen3-1.7B (scripts/score/score_decodes.sh).
# Resumable. STAGE=analysis only re-validates, re-analyzes, and re-renders. Output: analysis_dir/latency.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common.sh"

output_dir=$(python glr_paths.py analysis_dir)/latency
export LATENCY_BENCHMARKS="${LATENCY_BENCHMARKS:-math500,svamp}"

if [[ "${STAGE:-all}" == "all" ]]; then
  python analysis/latency/build_efficiency_manifest.py --output-dir ${output_dir}
  for shard_id in 0 1 2 3 4; do
    echo "========> shard ${shard_id}/4"
    python analysis/latency/run_matched_efficiency.py \
      --output-dir ${output_dir} \
      --shard-id ${shard_id} \
      --num-shards 5
  done
fi

echo "========> validation, analysis, tables, and figure"
python analysis/latency/validate_efficiency.py --output-dir ${output_dir} --stage cache
python analysis/latency/analyze_efficiency.py \
  --output-dir ${output_dir} \
  --bootstrap-replicates 10000 \
  --analysis-seed 20260726
python analysis/latency/validate_efficiency.py --output-dir ${output_dir} --stage outputs --compile-latex-if-available
echo "Table: ${output_dir}/tables/efficiency_table.md"
