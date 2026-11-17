"""R2 spin-woven reversible pilot for GLR deltas (not an intrinsic Qwen geometry).

Exact concrete identities from the SWN S3 4x4 spinor matrix representation:
G = mean_g rho(g)^T rho(g) > 0, C=[S,T], A=[S,C],
C^T G = G C, A^T G + G A = 0 and A^3=-12A.

The Cayley transform is evaluated analytically, no batched matrix inversion.
Each 4-coordinate block is a CHOSEN chart; model-level performance is open.
"""
from __future__ import annotations
import math
import torch

S = torch.tensor([[1,0,0,0],[0,-1,1,0],[0,0,1,0],[0,0,0,-1]],dtype=torch.float64)
T = torch.tensor([[1,0,0,0],[0,0,-1,0],[0,1,-1,0],[0,0,0,1]],dtype=torch.float64)
I = torch.eye(4,dtype=torch.float64)
ORBIT=(I,S,T,T@T,S@T,S@T@T)
G=sum((g.T@g for g in ORBIT),torch.zeros_like(I))/6
C=S@T-T@S
A=S@C-C@S


def quadratic_energy(x: torch.Tensor, metric: torch.Tensor | None=None) -> torch.Tensor:
    """Blockwise x^T G x, not scalar confidence or reasoning quality."""
    if x.ndim<1 or x.shape[-1]%4:
        raise ValueError("last coordinate dimension must be divisible by four")
    y=x.reshape(-1,4)
    g=G.to(device=y.device,dtype=y.dtype) if metric is None else metric.to(device=y.device,dtype=y.dtype)
    return (y*(y@g.T)).sum(dim=-1).reshape(*x.shape[:-1],-1)


def random_orientations(x: torch.Tensor, seed: int=2026) -> torch.Tensor:
    """Deterministic +/- orientations; exact magnitude-matched null control."""
    if x.numel()%4:
        raise ValueError("four-channel blocks required")
    generator=torch.Generator(device="cpu").manual_seed(seed)
    return 2*torch.randint(0,2,(x.numel()//4,),generator=generator,dtype=torch.int64)-1


def cayley_delta(x: torch.Tensor, strength: float=0.02, *,
                 direction: int=1, generator: torch.Tensor | None=None,
                 metric: torch.Tensor | None=None,
                 sign_blocks: torch.Tensor | None=None) -> torch.Tensor:
    """Nonlinear, reversible and G-isometric blockwise transport.

    u = direction * strength/2 * q(x)/(1+q(x))
    Cayley_u(A)x = x + [2u/(1+12u^2)] Ax + [2u^2/(1+12u^2)] A^2x.

    Deweave: call with direction=-1 on the output. This works because
    q is invariant and the same u is reconstituted from the output.
    """
    if x.ndim<1 or x.shape[-1]%4:
        raise ValueError("last dimension must be divisible by four")
    if not x.is_floating_point():
        raise TypeError("expected a real floating point tensor")
    if not math.isfinite(strength) or strength<0 or direction not in (-1,1):
        raise ValueError("invalid strength or direction")
    if not bool(torch.isfinite(x).all().item()):
        raise ArithmeticError("nonfinite source")
    if strength==0:
        return x
    dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
    z=x.to(dtype).reshape(-1,4)
    a=A.to(device=z.device,dtype=dtype) if generator is None else generator.to(device=z.device,dtype=dtype)
    g=G.to(device=z.device,dtype=dtype) if metric is None else metric.to(device=z.device,dtype=dtype)
    if a.shape!=(4,4) or g.shape!=(4,4):
        raise ValueError("invalid carrier shape")
    if not bool(torch.allclose(a@a@a,-12*a,rtol=1e-5,atol=1e-5)):
        raise ValueError("generator does not satisfy A^3=-12A")
    if not bool(torch.allclose(a.T@g+g@a,torch.zeros_like(g),rtol=1e-5,atol=1e-5)):
        raise ValueError("generator is not G-skew")
    q=(z*(z@g.T)).sum(-1).clamp_min(0)
    u=0.5*direction*strength*(1-1/(1+q))
    if sign_blocks is not None:
        signs=sign_blocks.reshape(-1).to(device=z.device,dtype=dtype)
        if signs.numel()!=z.shape[0] or not bool(((signs==1)|(signs==-1)).all().item()):
            raise ValueError("one +/- orientation required for each block")
        u=u*signs
    den=1+12*u.square()
    y=z+(2*u/den)[:,None]*(z@a.T)+(2*u.square()/den)[:,None]*(z@(a@a).T)
    if not bool(torch.isfinite(y).all().item()):
        raise ArithmeticError("nonfinite output")
    return y.reshape_as(x).to(x.dtype)
