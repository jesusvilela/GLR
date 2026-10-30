# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional
import torch


@dataclass
class BaseModelOutputWithPastWithLatentReason(BaseModelOutputWithPast):
    direction_delta: Optional[torch.FloatTensor] = (
        None  # (batch_size, seq_len, hidden_size)
    )
    latent_loss: Optional[torch.FloatTensor] = None
