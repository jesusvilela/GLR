<!--
SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch

SPDX-License-Identifier: MIT
-->

# <img src="assets/glr_icon.svg" height="32" align="top" alt=""> Geometric Latent Reasoning Induces Shorter Generations in LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2606.02248-b31b1b.svg)](https://arxiv.org/abs/2606.02248)
[![Project page](https://img.shields.io/badge/Project_page-glr--llm.github.io-2f6feb.svg)](https://glr-llm.github.io/)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-6f42c1.svg)](https://neurips.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSES/MIT.txt)

**Shashi Kumar**<sup>1,2,\*</sup>, **Yacouba Kaloga**<sup>1,\*</sup>, Petr Motlicek<sup>1,3</sup>, Ina Kodrasi<sup>1</sup>, Andrea Cavallaro<sup>2</sup><br>
<sup>1</sup>Idiap Research Institute, Switzerland &nbsp; <sup>2</sup>EPFL, Switzerland &nbsp; <sup>3</sup>Brno University of Technology, Czech Republic &nbsp; <sup>\*</sup>Equal contribution

This is the official implementation of the NeurIPS 2026 paper **Geometric Latent Reasoning Induces Shorter Generations in LLMs**.

GLR treats chain-of-thought as a path through the model's token-embedding space. A small linear transition head predicts the direction of each step, so the model takes *K* continuous latent steps before it resumes ordinary decoding, which yields much shorter generations without any explicit length penalty.

> [!NOTE]
> The [**project page**](https://glr-llm.github.io/) presents the paper's highlights: an animated walkthrough of the method, the main results, latency, and analyses.

This repository reproduces the paper's results with Qwen3-0.6B and Qwen3-1.7B: data preparation, training, decoding, scoring, figures, baselines, and analyses.

## Quick start

Requirements: Linux x86_64, an NVIDIA GPU with a CUDA 12.x driver (all experiments used one H100 80GB), and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/idiap/GLR.git && cd GLR
uv sync && source .venv/bin/activate

bash scripts/data/download_mot.sh && bash scripts/data/tokenize_mot.sh && bash scripts/data/filter_mot.sh
bash scripts/data/prepare_eval_sets.sh
for size in 0.6B 1.7B; do
  bash scripts/train/sft_qwen3_${size}.sh && bash scripts/train/glr_qwen3_${size}.sh
  bash scripts/decode/cot_sft_qwen3_${size}.sh && bash scripts/decode/glr_qwen3_${size}_vllm.sh
done
bash scripts/decode/glr_qwen3_1.7B_k0_ablation_vllm.sh
bash scripts/score/score_decodes.sh
bash plots/run_all_plots.sh
```

`uv sync` creates `.venv/` with Python 3.10 and the pinned dependencies (PyTorch 2.8, Transformers 4.57, vLLM 0.11, FlashAttention 2.8).

> [!TIP]
> The quick start decodes GLR with vLLM (the `_vllm` scripts), which takes minutes instead of hours. The paper's results use the Hugging Face decoder (`glr_qwen3_${size}.sh`), with the same settings and output files. See [Decoding](#4-decoding).

<details>
<summary><b>Configuring paths</b></summary>

<br>

All locations come from [`glr_paths.py`](glr_paths.py): base models are downloaded from the Hugging Face Hub, and data, checkpoints and outputs are stored inside the repository. To use your own locations, create a git-ignored `glr_paths_local.py` next to it and override only what you need:

```python
MODELS = {"qwen3_1.7b": "/path/to/Qwen3-1.7B"}
PATHS = {"ckpt_dir": "/scratch/glr/ckpts", "hf_cache_dir": "/scratch/hf_cache"}
```

`python glr_paths.py` prints every entry, `python glr_paths.py ckpt_dir` a single one.

</details>

## Models and data

The scripts download the models and data from the Hugging Face Hub; none of them is redistributed in this repository.

| Name | Location | Revision used | License | Role |
|---|---|---|---|---|
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de` | Apache-2.0 | base model |
| Qwen3-1.7B | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `70d244c` | Apache-2.0 | base model |
| Mixture-of-Thoughts (`math` split) | [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) | `e55fa28` | not specified (built from OpenR1-Math-220k, Apache-2.0) | training data |

The evaluation sets and their licenses are listed in [Evaluation sets](#2-evaluation-sets), and third-party code in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Reproducing the paper

Every step is one script, run from the repository root with the environment active.

| Step | Scripts | Output |
|---|---|---|
| [1. Training data](#1-training-data) | `scripts/data/` | `mot_train_10k_dir` |
| [2. Evaluation sets](#2-evaluation-sets) | `scripts/data/prepare_eval_sets.sh` | `eval_dir` |
| [3. Training](#3-training) | `scripts/train/` | `ckpt_dir/<model>/` |
| [4. Decoding](#4-decoding) | `scripts/decode/` | `decode_dir/<model>/` |
| [5. Scoring](#5-scoring) | `scripts/score/` | `scored_dir/<model>/` |
| [6. Figures](#6-figures) | `plots/` | `figure_dir/` |
| [7. Baselines](#7-baselines) | `baselines/` | `baseline_dir/` |
| [8. Analyses](#8-analyses) | `analysis/` | `analysis_dir/` |

### 1. Training data

Download the `math` split of [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts), apply the Qwen3 chat template, and keep a random subset of 10,000 examples:

```bash
bash scripts/data/download_mot.sh && bash scripts/data/tokenize_mot.sh && bash scripts/data/filter_mot.sh
```

<details>
<summary>Details</summary>

<br>

The download (~700 MB, revision `e55fa28`) is checked against its sha256 sums. The chat template uses no system prompt. The filter keeps examples of 300–8,192 tokens with exactly one `<think>…</think>` block and at least 100 reasoning tokens (93,733 → 70,770 examples), shuffles them with seed 42, and keeps the first 10,000.

</details>

### 2. Evaluation sets

Build the six evaluation sets from the Hugging Face Hub at pinned revisions, with prompts rendered in the Qwen3 chat format (zero-shot, no system prompt). The script checks the files against [`data/eval/SHA256SUMS`](data/eval/SHA256SUMS), the evaluation sets used in the paper:

```bash
bash scripts/data/prepare_eval_sets.sh
```

| Benchmark | Examples | Step limit | Source | License |
|---|---:|---:|---|---|
| GSM8K | 1,319 | 2,048 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), prompt from lm-evaluation-harness | MIT |
| SVAMP | 300 | 2,048 | [ChilleD/SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) | MIT |
| MultiArith | 180 | 2,048 | [ChilleD/MultiArith](https://huggingface.co/datasets/ChilleD/MultiArith) | not specified |
| MATH500 | 500 | 4,096 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), Minerva prompt from lm-evaluation-harness | not specified (from MATH and PRM800K, both MIT) |
| AMC23 | 40 | 4,096 | [math-ai/amc23](https://huggingface.co/datasets/math-ai/amc23) | not specified |
| OlympiadBench | 674 | 4,096 | [Hothan/OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench), `OE_TO_maths_en_COMP` | Apache-2.0 |

> [!NOTE]
> The evaluation sets are not redistributed in this repository. The Hugging Face pages of MultiArith, MATH500 and AMC23 state no license; check the terms of the original sources before using them.

<details>
<summary>How the files are created</summary>

<br>

The GSM8K and MATH500 prompts are the zero-shot prompts of lm-evaluation-harness ([`lm_eval_tasks/`](lm_eval_tasks)), and the MATH500 targets are the boxed answers of the reference solutions, normalized as in the harness. OlympiadBench has only a `train` split on the Hub. [`scripts/data/export_lm_eval_sets.sh`](scripts/data/export_lm_eval_sets.sh) produces the same GSM8K and MATH500 files with lm-evaluation-harness at commit [`03c44ad`](https://github.com/EleutherAI/lm-evaluation-harness/tree/03c44adc0586f88bb343a74da1a1c602103536dd), as originally done for the paper; see the script for its requirements.

</details>

### 3. Training

Train the CoT-SFT baseline and GLR for both models (3–7 hours each on one H100):

```bash
bash scripts/train/sft_qwen3_0.6B.sh && bash scripts/train/glr_qwen3_0.6B.sh
bash scripts/train/sft_qwen3_1.7B.sh && bash scripts/train/glr_qwen3_1.7B.sh
```

Each run writes the final model and `train.log` to `ckpt_dir/{cot_sft,glr}_qwen3-{0.6B,1.7B}/`.

<details>
<summary>Training settings</summary>

<br>

| | |
|---|---|
| Base models | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (`c1899de`), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (`70d244c`) |
| Optimization | 5 epochs, AdamW (β = 0.9, 0.95), weight decay 1e-4 |
| Learning rate | 4e-5, cosine schedule, 5% warmup |
| Batch | effective 16 (micro-batch 1 × 16 accumulation steps), max length 8,192 tokens |
| Precision | bf16, gradient checkpointing |
| Embeddings | input embeddings frozen (tied to the output projection) |
| GLR | transition head: one linear layer (~1M parameters for 0.6B, ~4M for 1.7B); MSE transition objective, discount 0.999, no cross-entropy at latent positions |

The model code is in [`glr_qwen3/`](glr_qwen3). The CoT-SFT baseline with trainable input embeddings is trained with `FREEZE_INPUT_EMBEDDINGS=False bash scripts/train/sft_qwen3_1.7B.sh` (writes `cot_sft_qwen3-1.7B_embed_unfreeze/`). Extra arguments go to the trainer, e.g. `--max_steps=20` for a quick test.

</details>

### 4. Decoding

Decode greedily on all six benchmarks, within the step limits above (for GLR, the *K* latent steps count toward the limit):

```bash
bash scripts/decode/cot_sft_qwen3_0.6B.sh && bash scripts/decode/glr_qwen3_0.6B.sh
bash scripts/decode/cot_sft_qwen3_1.7B.sh && bash scripts/decode/glr_qwen3_1.7B.sh
bash scripts/decode/glr_qwen3_1.7B_k0_ablation.sh
```

GLR-0.6B is decoded with *K* ∈ {5, 10, 20, 50} on GSM8K and MATH500 and *K* = 20 on the other benchmarks; GLR-1.7B with *K* ∈ {5, 10, 20, 50, 80, 100} and *K* = 10. The last script decodes GLR-1.7B with *K* = 0 for the ablation. Pass benchmark names to split the work over GPUs, e.g. `glr_qwen3_1.7B.sh svamp amc23`. The CoT-SFT model with trainable embeddings is decoded with `MODEL_NAME=cot_sft_qwen3-1.7B_embed_unfreeze bash scripts/decode/cot_sft_qwen3_1.7B.sh`.

The GLR decoder used in the paper runs one example at a time in Hugging Face and is slow. Each GLR script has a **vLLM counterpart** with a `_vllm` suffix (e.g. `glr_qwen3_1.7B_vllm.sh`), with the same arguments, settings and output files; decoding both *K* values of GSM8K for GLR-0.6B takes about 2.5 minutes on one H100.

<details>
<summary>How the vLLM decoder works</summary>

<br>

After `<think>`, the *K* latent steps are computed with the GLR model in Hugging Face, batched over prompts ([`vllm_glr_qwen3/generate.py`](vllm_glr_qwen3/generate.py)). The prompt and the latent states are then passed to vLLM as prompt embeddings, and vLLM decodes the text; it loads the checkpoint as a plain Qwen3 model without the transition head ([`vllm_glr_qwen3/model.py`](vllm_glr_qwen3/model.py)).

The outputs are not token-identical to the Hugging Face decoder, since bf16 kernels differ and greedy paths diverge after near-ties; with `--batch_size 1` the latent steps themselves are identical. The paper's results use the Hugging Face decoder.

</details>

### 5. Scoring

Score every decoded model (or pass model names to score selected ones):

```bash
bash scripts/score/score_decodes.sh
```

The scorers in [`src/metric/`](src/metric) follow lm-evaluation-harness: GSM8K uses exact match on the final number; the other benchmarks report exact match and [Math-Verify](https://github.com/huggingface/Math-Verify). Each run prints the number of predictions against the size of the test set.

### 6. Figures

Draw all figures and tables of the paper for both models (or a single one with `python plots/plot_benchmark.py --model 1.7B --benchmark gsm8k_zeroshot`):

```bash
bash plots/run_all_plots.sh
```

An answer counts as correct at budget *B* only if it is correct and finished within *B* generated steps, latent steps included.

<details>
<summary>Figures in the paper and the files that produce them</summary>

<br>

| Paper figure | File in `figure_dir` |
|---|---|
| GSM8K and MATH500 accuracy vs. length (main text) | `{gsm8k,math500}_qwen3-{0.6B,1.7B}.pdf` |
| SVAMP (main text) | `svamp_qwen3-{0.6B,1.7B}.pdf` |
| *K* = 0 ablation (main text) | `qwen3-1.7B_glr0_5_10_gsm8k_boxplot_correct.pdf` |
| AMC23, OlympiadBench, MultiArith (appendix) | `{amc23,olympiadbench,multiarith}_qwen3-{0.6B,1.7B}.pdf` |
| Generated steps over all answers (appendix) | `*_all_answers.pdf` |

Each figure shows accuracy as a function of the budget and the generated steps of correct answers; `*_budget_table.csv` holds the numbers. GSM8K uses exact match, the other benchmarks Math-Verify, except the SVAMP CoT-SFT baseline (exact match).

</details>

### 7. Baselines

Two continuous-reasoning baselines on GSM8K, SVAMP and MATH500, with the same prompts and step limits: **Soft Thinking** ([Zhang et al., 2025](https://arxiv.org/abs/2505.15778)) on the CoT-SFT checkpoints, and **Coconut-style hidden-state feedback** ([Hao et al., 2024](https://arxiv.org/abs/2412.06769)) on Qwen3-0.6B. After running them, compute the budgeted accuracy and median length of every run:

```bash
python baselines/score_baseline_results.py
```

<details>
<summary>Running the baselines</summary>

<br>

- **Soft Thinking**: `bash baselines/soft_thinking/run_soft_thinking.sh` (both models; or pass `0.6B` or `1.7B`). A fixed number of soft-token steps (*K* = 20 for Qwen3-0.6B, *K* = 10 for Qwen3-1.7B), or soft steps until `</think>` (at most 256), with τ = 0.6, top-k 30, top-p 0.95, min-p 0.001 and the top-10 tokens mixed.
- **Coconut-style hidden-state feedback**: `bash baselines/coconut_hsr/train_hsr_qwen3_0.6B.sh`, then `bash baselines/coconut_hsr/decode_hsr_qwen3_0.6B.sh`. A two-pass, teacher-forced variant in which the reasoning-token embeddings are replaced by the preceding last-layer hidden states ([`modeling_hsr.py`](baselines/coconut_hsr/modeling_hsr.py)), trained with the same data and settings as GLR; decoding feeds back hidden states for *K* = 10 or 100 steps, or until `</think>`.

Both decoders resume where they stopped (`OVERWRITE=1` starts over, `LIMIT=2` runs a quick test). The scores are written to `baseline_dir/results/baseline_budget_results.{json,csv}`.

</details>

### 8. Analyses

Three analyses of GLR-10 on Qwen3-1.7B (GSM8K, MATH500, SVAMP), using the checkpoints and scored generations from steps 3–5. Each needs one GPU and resumes where it stopped:

```bash
bash analysis/mechanistic/run_mechanistic.sh
bash analysis/latency/run_latency.sh
bash analysis/robustness/run_robustness.sh
```

<details>
<summary>Details</summary>

<br>

- **Latent dynamics** ([`analysis/mechanistic/`](analysis/mechanistic)): the CoT-SFT trace is teacher-forced through the GLR checkpoint, and each latent state is monotonically aligned to it in the final-layer hidden-state space. Reports alignment progress, distance to the aligned CoT token embedding, FutureHit@10 and a logistic regression of correctness. Causal controls on 100 fixed examples per benchmark replace the learned transitions with a zero transition, the repeated `<think>` embedding, a norm-matched random head or a noisy transition.
- **Latency and compute** ([`analysis/latency/`](analysis/latency)): CoT-SFT and GLR-10 with the same Hugging Face decoder (bf16, FlashAttention-2, KV cache, greedy, batch size 1) on MATH500 and SVAMP, on one H100 80GB; `LATENCY_BENCHMARKS=gsm8k,math500,svamp` adds GSM8K. Timing covers prefill through the final step; FLOPs are counted from the instrumented model calls.
- **Statistical robustness** ([`analysis/robustness/`](analysis/robustness)): paired bootstrap confidence intervals (10,000 resamples) and exact McNemar tests with Holm correction at the constrained budgets (256 steps for GSM8K and SVAMP, 512 for MATH500), and three sampled trajectories per problem (temperature 0.6, top-p 0.95) for sampled pass@1, majority@3 and pass@3.

</details>

## Repository layout

```
glr_qwen3/        GLR model: Qwen3 with the transition head and latent decoding
vllm_glr_qwen3/   fast GLR decoding with vLLM
src/              data preparation, training, decoding and scoring
scripts/          one script per step: data, train, decode, score
data/eval/        checksums of the six evaluation sets (built by scripts/data/prepare_eval_sets.sh)
plots/            figures and tables
baselines/        Soft Thinking and Coconut-style baselines
analysis/         latent-dynamics, latency and robustness analyses
lm_eval_tasks/    lm-evaluation-harness configs for the GSM8K and MATH500 prompts
```

## Citation

```bibtex
@inproceedings{kumar2026geometric,
      title={Geometric Latent Reasoning Induces Shorter Generations in LLMs},
      author={Shashi Kumar and Yacouba Kaloga and Petr Motlicek and Ina Kodrasi and Andrea Cavallaro},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2026},
      eprint={2606.02248},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2606.02248},
}
```

## Contact

Questions and issues are welcome on GitHub, or by email to Shashi Kumar (shashi.kumar@idiap.ch, shashi.kumar@epfl.ch) and Yacouba Kaloga (yacouba.kaloga@idiap.ch).

## License

MIT, see [`LICENSES/MIT.txt`](LICENSES/MIT.txt). The model code in [`glr_qwen3/`](glr_qwen3) is adapted from the Qwen3 implementation in Hugging Face Transformers and keeps its Apache-2.0 license ([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)). Third-party code, models and datasets are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).

---

*Built with ❤️ at the [Idiap Research Institute](https://www.idiap.ch/).*
