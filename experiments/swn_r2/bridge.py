"""A constrained, opt-in bridge from the R2 Cayley operator into trained GLR.

The checkpoint and original Qwen causal decoder are unchanged. The bridge
only replaces the already-predicted latent delta during an active GLR prefix.
"""
from __future__ import annotations

import torch
from experiments.swn_r1.adapter import LatentDeltaBridge, RuntimeTelos
from experiments.swn_r2.cayley import cayley_delta, random_orientations


class CayleyDeltaBridge(LatentDeltaBridge):
    def __init__(self, model, *, mode: str = "swn_cayley",
                 strength: float = 0.02, random_seed: int = 2026,
                 telos: RuntimeTelos | None = None):
        if mode not in ("swn_cayley", "orientation_cayley"):
            raise ValueError("invalid R2 Cayley arm")
        super().__init__(model, mode="swn", strength=strength,
                         random_seed=random_seed, telos=telos)
        self.cayley_mode = mode

    def _candidate(self, original: torch.Tensor) -> torch.Tensor:
        signs = (random_orientations(original, self.random_seed)
                 if self.cayley_mode == "orientation_cayley" else None)
        return cayley_delta(original, self.strength, sign_blocks=signs)
