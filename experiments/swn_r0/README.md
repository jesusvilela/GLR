# SWN x GLR / R0: a noncommutative pilot, not a performance claim

An independent experimental adapter for the fork, **not upstream code**.
The original [Idiap GLR](https://github.com/idiap/GLR) mathematics, code, citation and baseline remain unchanged.

## Grounding

- GLR's `latent_reason_pred_delta_embed=True` learns a vector increment from the last Qwen3 hidden state; its generation update adds this increment to the current latent input embedding.
- Spin Woven Numbers supplies concrete `S3` spinor matrices `s` (reflection) and `t` (3-cycle), from [`SpinWoven/S3Spinor.lean`](https://github.com/Dojo-1/spin-woven-numbers/blob/main/SpinWoven/S3Spinor.lean).
- This R0 pilot chooses a **noncanonical** partition of a real embedding delta into four-component blocks. Each block is treated as a spinor under that explicitly chosen representation. There is *no evidence* that these blocks reflect learned Qwen3 features.
- `C=st-ts` exposes noncommutativity. `G=|S3|^-1 sum_g rho(g)^T rho(g)` is an invariant *positive-definite auxiliary metric*, not the separate signature (2,2) carrier metric.

For `x` a four-vector, write `q(x)=x^T G x` and `c(x)=q(x)/(1+q(x))`. The optional output is:

```text
F_epsilon(x) = x + epsilon * c(x) * C x
```

A full latent delta is processed blockwise. At `epsilon=0`, this is exactly GLR's original delta. The nonlinearity comes from `c(x)`. For an invertible frame `B`, covariance is tested *only when* `x'=Bx`, `C'=BCB^-1` and `G'=B^-T G B^-1`. It is **not** asserted that the operator is equivariant under fixed `S3` actions or preserves a conserved spinor energy.

## Minimal CPU check

From the repository root (requires PyTorch):

```bash
python -m unittest discover -s tests -p 'test_swn_r0.py' -v
```

## Proposed model experiment (NOT YET RUN)

Apply `spin_woven_delta(predicted_delta, epsilon)` at the *latent generation update* in both the Hugging Face and vLLM-prefix decoders, keeping all checkpoints and model weights fixed. Do not alter training or the paper's baseline. Preregister epsilon, prompt subset and seeds before inspecting task scores. Compare:

1. Original GLR (`epsilon=0`).
2. Spin-woven update (preselected epsilon values, e.g. 0.01 and 0.05).
3. Norm-matched random mixing and coordinate-permutation controls.
4. Same parameter/compute-budget alternatives, including an equally simple linear mixer.

Track exact-match accuracy, completion length (including latent steps), latency, numerical stability and per-example correctness flips, with independent held-out problems. Reject the geometry-as-performance hypothesis if gains disappear against the controls or fail to replicate. A fixed checkpoint may not accommodate transformed deltas; a later *separately registered* training experiment may be necessary.

## Boundary / provenance

- Experimental **H**: SWN-style blockwise latent transport may improve GLR outcomes.
- Established by source inspection: the two projects supply the original mechanisms and concrete finite matrices.
- Unit tests establish local numerical properties only, **not** model capability or an intrinsic geometric representation.
- SWN's `H^1` 19-dimensional model, 13 blind directions, arithmetic primes and exact Clifford work are **not** transferred to the Qwen3 latent space by this adapter. Any such bridge is a new mathematical obligation.
