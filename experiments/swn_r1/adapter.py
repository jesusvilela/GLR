"""Model-level, opt-in SWN adapter for Idiap GLR's Hugging Face decoder.

Does not patch source or checkpoint weights. Only intercepts the learned
direction delta when GLR's existing latent-prefix condition is satisfied.
No backward attention, no oracle access, and no claims of frame invariance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import torch

from experiments.swn_r0.transport import spin_woven_delta, spinor_data


@dataclass(frozen=True)
class RuntimeTelos:
    """Feasibility envelope, not a learned scalar utility."""
    max_relative_perturbation: float = 0.20

    def __post_init__(self):
        if not (isfinite(self.max_relative_perturbation)
                and 0 < self.max_relative_perturbation <= 1):
            raise ValueError("max_relative_perturbation must be in (0,1]")


@dataclass
class TransitionAudit:
    proposed: int = 0
    accepted: int = 0
    reverted: int = 0
    max_observed_ratio: float = 0.0
    reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(proposed=self.proposed, accepted=self.accepted,
                    reverted=self.reverted, max_observed_ratio=self.max_observed_ratio,
                    remainder_reasons=dict(self.reasons))


class LatentDeltaBridge:
    """Interception context for CustomQwen3ForCausalLM.prepare_inputs_for_generation.

    All variants leave GLR's native forward, KV cache and generation state
    implementation untouched. The wrapper temporarily substitutes only the
    most recently predicted delta, then restores it even on exceptions.
    """

    def __init__(self, model: Any, *, mode: str = "glr",
                 strength: float = 0.02, random_seed: int = 2026,
                 telos: RuntimeTelos | None = None):
        if mode not in ("glr", "swn", "random"):
            raise ValueError("mode must be glr, swn or random")
        if not isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and >= 0")
        if not getattr(model.config, "enable_latent_reason", False):
            raise ValueError("a GLR-trained checkpoint with latent reasoning is required")
        if not getattr(model.config, "latent_reason_pred_delta_embed", False):
            raise ValueError("requires GLR's additive predicted-delta mode")
        self.model = model
        self.mode = mode
        self.strength = strength
        self.random_seed = random_seed
        self.telos = telos or RuntimeTelos()
        self.audit = TransitionAudit()
        self._original = None

    def _active(self, ids: torch.Tensor) -> bool:
        if ids is None or not hasattr(ids, "size"):
            return False
        if self.model.model._last_direction_delta is None:
            return False
        start, end = self.model.model.get_start_end_think_idx(ids, train=False)
        n = ids.size(-1)
        budget = self.model.config.latent_reason_token_budget
        return (start >= 1 and end >= n and budget is not None
                and end - start + 1 <= budget and n - start >= 0)

    def _candidate(self, original: torch.Tensor) -> torch.Tensor:
        if self.mode == "glr" or self.strength == 0:
            return original
        if original.shape[-1] % 4:
            raise ValueError("GLR hidden dimension must be a multiple of 4")
        if self.mode == "swn":
            return spin_woven_delta(original, self.strength)

        # Fixed random skew mixer, with PER-BLOCK perturbation magnitude
        # matched to the SWN arm on the same original delta.
        _, _, c, g = spinor_data(dtype=original.dtype, device=original.device)
        gen = torch.Generator(device="cpu").manual_seed(self.random_seed)
        a = torch.randn(4, 4, generator=gen, dtype=torch.float64)
        rnd = (a - a.T).to(device=original.device, dtype=original.dtype)
        ref = spin_woven_delta(original, self.strength) - original
        alt = spin_woven_delta(original, self.strength,
                               commutator=rnd, metric=g) - original
        ref_blocks = ref.reshape(*original.shape[:-1], -1, 4)
        alt_blocks = alt.reshape(*original.shape[:-1], -1, 4)
        ref_n = torch.linalg.vector_norm(ref_blocks.float(), dim=-1, keepdim=True)
        alt_n = torch.linalg.vector_norm(alt_blocks.float(), dim=-1, keepdim=True)
        scale = torch.where(alt_n > 1e-20, ref_n / alt_n.clamp_min(1e-20),
                            torch.zeros_like(alt_n))
        return original + (alt_blocks.float() * scale).reshape_as(original).to(original.dtype)

    def _govern(self, source: torch.Tensor) -> torch.Tensor:
        if self.mode == "glr" or self.strength == 0:
            return source
        self.audit.proposed += 1
        if not bool(torch.isfinite(source).all().item()):
            self._reject("nonfinite_source")
            return source
        proposed = self._candidate(source)
        if not bool(torch.isfinite(proposed).all().item()):
            self._reject("nonfinite_candidate")
            return source
        diff = torch.linalg.vector_norm((proposed - source).float(), dim=-1)
        denom = torch.linalg.vector_norm(source.float(), dim=-1).clamp_min(1e-12)
        ratio = float((diff / denom).amax().item())
        self.audit.max_observed_ratio = max(self.audit.max_observed_ratio, ratio)
        if ratio > self.telos.max_relative_perturbation:
            self._reject("telos_norm_envelope")
            return source
        self.audit.accepted += 1
        return proposed

    def _reject(self, reason: str):
        self.audit.reverted += 1
        self.audit.reasons[reason] = self.audit.reasons.get(reason, 0) + 1

    def __enter__(self):
        if self._original is not None:
            raise RuntimeError("bridge cannot be entered recursively")
        self._original = self.model.prepare_inputs_for_generation
        original_prepare = self._original

        def prepare(*args, **kwargs):
            ids = args[0] if args else kwargs.get("input_ids")
            if not self._active(ids):
                return original_prepare(*args, **kwargs)
            old_delta = self.model.model._last_direction_delta
            next_delta = self._govern(old_delta)
            self.model.model._last_direction_delta = next_delta
            try:
                return original_prepare(*args, **kwargs)
            finally:
                self.model.model._last_direction_delta = old_delta

        self.model.prepare_inputs_for_generation = prepare
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.prepare_inputs_for_generation = self._original
        self._original = None
        return False
