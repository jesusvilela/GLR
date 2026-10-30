# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Lets vLLM load a GLR checkpoint as a plain Qwen3 model.

A GLR checkpoint is a Qwen3 model plus the transition head. vLLM only decodes text
tokens, so it loads the Qwen3 weights and skips the transition head; the latent
steps are computed in Hugging Face (see generate.py) and passed in as prompt embeddings.
"""

from transformers import AutoConfig
from vllm import ModelRegistry
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config

GLR_ONLY_WEIGHTS = (
    "model.direction_delta_predictor.",
    "model.latent_reason_discount_vector",
)


class GLRQwen3ForCausalLM(Qwen3ForCausalLM):
    def load_weights(self, weights):
        return super().load_weights(
            (name, w) for name, w in weights if not name.startswith(GLR_ONLY_WEIGHTS)
        )


def register():
    """Register the GLR config and architecture with transformers and vLLM (in this process)."""
    AutoConfig.register(CustomQwen3Config.model_type, CustomQwen3Config, exist_ok=True)
    ModelRegistry.register_model("CustomQwen3ForCausalLM", GLRQwen3ForCausalLM)
