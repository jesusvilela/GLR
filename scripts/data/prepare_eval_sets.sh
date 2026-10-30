#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Build the six eval sets from the Hugging Face Hub (pinned revisions) into eval_dir and check them
# against data/eval/SHA256SUMS, the files used in the paper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

python src/data/prepare_gsm8k_test.py
python src/data/prepare_math500_test.py
python src/data/prepare_svamp_test.py
python src/data/prepare_multiarith_test.py
python src/data/prepare_amc23_test.py
python src/data/prepare_olympiadbench_test.py --configs OE_TO_maths_en_COMP

checksums=$(pwd)/data/eval/SHA256SUMS
(cd "$(python glr_paths.py eval_dir)" && sha256sum -c "${checksums}")
