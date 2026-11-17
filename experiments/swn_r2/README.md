# R2 — Reversible SWN transport, GLR and Teloi

**Status:** finite algebra identity checked exactly over rational matrices; numerical CPU tests passed locally. Reasoning advantage remains **H**.

## The obstruction and correction

For the concrete SWN \`S3\` spinor generators \`S,T\`, take the \`S3\`-averaged positive metric \`G\` and \`C=ST-TS\`. An important negative result is **\`C^T G=GC\`**: \`C\` is **G-self-adjoint**, so its G-skew projection is **zero**. It does not generate a nontrivial metric-preserving Cayley flow.

The iterated commutator \`A=[S,C]=[S,[S,T]]\` is nonzero and satisfies:

\`\`\`text
A = [[0,0,0,0],[0,2,-4,0],[0,4,-2,0],[0,0,0,0]]
G = [[1,0,0,0],[0,4/3,-2/3,0],[0,-2/3,4/3,0],[0,0,0,1]]
A^T G + G A = 0
A^3 = -12 A
\`\`\`

These are exact finite matrix identities (also independently checked with Python rational arithmetic); \`A\` acts only on the 2-dimensional standard sector, leaving both 1D spinor sectors fixed.

## Reversible nonlinear Cayley step

On each four-coordinate block \`x\`, let \`q=x^T G x\`, \`u=(epsilon/2)*q/(1+q)\`. The R2 candidate replaces the GLR predicted delta block via:

\`\`\`text
F_eps(x) = (I-uA)^-1 (I+uA) x
         = x + 2u/(1+12u^2) Ax + 2u^2/(1+12u^2) A^2x
\`\`\`

By the skew identity, \`q(F_eps(x))=q(x)\`; the inverse uses \`-u\` (which is reconstructed from unchanged \`q\`). A moving frame is compatible only when \`x'=Bx\`, \`A'=BAB^-1\` and \`G'=B^-T G B^-1\`.

The four-coordinate partition in Qwen's embedding space remains **A: an assumed chart**, not a discovered neural spinor representation. Conservation of an auxiliary metric does not imply accuracy.

## Controls / model-level entry

- Original \`glr\` (no delta replacement).
- R0 \`swn\` additive commutator and R0 \`random\` mixer.
- New \`swn_cayley\` reversible transport.
- New \`orientation_cayley\` signs \`+/- A\` across blocks, preselected by seed and exactly matched in per-block G-distance.
- Each has a two-pass variant with identical protocol and all tokens/seconds counted.

\`CayleyDeltaBridge\` intercepts only GLR's active learned latent prefix, records runtime Telos reversions, and restores the model's original cached direction. Source weights and Qwen's causal attention stay untouched. A **trained GLR checkpoint** is mandatory for any meaningful model-level benchmark. A plain pretrained Qwen3 with random transition head must be rejected.

## Tests and falsifiability

Run \`python -m unittest discover -s tests -p test_swn_r2.py -v\`. Check invariance, inverse, frame covariance, orientations, gradients, complete restoration and invalid inputs. Hold out the full official benchmark until the method and seed schedule are fixed. Compare paired correctness at equal compute and equal passes; do not conflate the two-pass benefit with a geometric effect.

**Evidence:** finite exact identities (P for these matrices); CPU numerical properties (M); Qwen latent spinor semantics and reasoning improvement (H). No claim of Idiap endorsement. Original GLR: [idiap/GLR](https://github.com/idiap/GLR); SWN matrix source: [Spin Woven Numbers](https://github.com/Dojo-1/spin-woven-numbers/blob/main/SpinWoven/S3Spinor.lean).
