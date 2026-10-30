#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Download the math split of open-r1/Mixture-of-Thoughts at the revision used in the paper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

revision=e55fa28006c0d0ec60fb3547520f775dd42d02cd
out_dir=$(python glr_paths.py mot_raw_dir)

hf download open-r1/Mixture-of-Thoughts \
  --repo-type dataset \
  --revision ${revision} \
  --include "math/*" \
  --local-dir ${out_dir}

# Check that the files are the ones used in the paper.
cd ${out_dir}
sha256sum -c <<EOF
f3875a97e04d614a4e45929ba9321ec492d1aebecdc35c31faf8a2293955c480  math/train-00000-of-00004.parquet
6d152451dfeef388b6a41c99a92c900f5b85eabc90180fadaf85e9fcc56d64cb  math/train-00001-of-00004.parquet
9343a9833f6fc910891c0984bbe2804214f6284e7fbee9a91b9d490d6a724381  math/train-00002-of-00004.parquet
c7ea52f68357a22bf03ed53496727ff599f5a4ae84d1741de9ecac0597428641  math/train-00003-of-00004.parquet
EOF
