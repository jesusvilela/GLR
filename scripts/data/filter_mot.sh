#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

# Filter the formatted Mixture-of-Thoughts math data and keep a random 10K subset (seed 42).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

python src/data/filter_datasets_mot.py
