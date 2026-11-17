"""SWN x GLR R0: isolated, differentiable spinor-block transition pilot.

The S3 spinor generators are transcribed from SpinWoven/S3Spinor.lean.
This is an optional transformation of GLR's predicted delta, not a claim
that Qwen hidden coordinates carry a canonical S3 action.

No original GLR code is modified; this file is an independent experiment.
"""
from __future__ import annotations

import torch

S = torch.tensor([[1, 0, 0, 0], [0, -1, 1, 0],
                  [0, 0, 1, 0], [0, 0, 0, -1]], dtype=torch.float64)
T = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0],
                  [0, 1, -1, 0], [0, 0, 0, 1]], dtype=torch.float64)
I = torch.eye(4, dtype=torch.float64)


def spinor_data(*, dtype: torch.dtype = torch.float64, device=None):
    """Return generator matrices, commutator and S3-invariant positive metric.

    The averaging metric is derived from the finite six-element group;
    it is not the indefinite (2,2) form on the separate hyperbolic double.
    """
    s, t = S.to(device=device, dtype=dtype), T.to(device=device, dtype=dtype)
    eye = torch.eye(4, dtype=dtype, device=device)
    orbit = [eye, s, t, t @ t, s @ t, s @ t @ t]
    metric = sum((g.T @ g for g in orbit), torch.zeros_like(eye)) / 6
    return s, t, s @ t - t @ s, metric


def spin_woven_delta(delta: torch.Tensor, strength: float = 0.0,
                     *, commutator: torch.Tensor | None = None,
                     metric: torch.Tensor | None = None) -> torch.Tensor:
    """Transform (...,d) deltas as d/4 explicit spinor blocks.

    F(x) = x + strength * c(x) * Cx, with C=[s,t],
    c(x)=x^T G x / (1+x^T G x). The chosen decomposition is not canonical.

    A change of frame is covariant only when C and G are transported too.
    This operator is not asserted to commute with the S3 action.
    """
    if delta.ndim < 1 or delta.shape[-1] % 4:
        raise ValueError("last dimension must be divisible by four")
    if not delta.is_floating_point():
        raise TypeError("delta must be floating-point")
    if strength == 0.0:
        return delta  # bitwise-identical GLR baseline
    if commutator is None or metric is None:
        _, _, default_c, default_g = spinor_data(dtype=delta.dtype, device=delta.device)
        commutator = default_c if commutator is None else commutator
        metric = default_g if metric is None else metric
    if commutator.shape != (4, 4) or metric.shape != (4, 4):
        raise ValueError("commutator and metric must be 4x4")
    c = commutator.to(device=delta.device, dtype=delta.dtype)
    g = metric.to(device=delta.device, dtype=delta.dtype)
    x = delta.reshape(*delta.shape[:-1], -1, 4)
    energy = torch.einsum('...bi,ij,...bj->...b', x, g, x)
    # Clamp is a numerical safety measure, not an algebraic identity.
    energy = energy.clamp_min(0)
    weight = energy / (1 + energy)
    changed = x + strength * weight.unsqueeze(-1) * (x @ c.T)
    return changed.reshape_as(delta)
