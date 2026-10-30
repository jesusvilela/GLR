<!--
SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch

SPDX-License-Identifier: MIT
-->

# THIRD PARTY CODE

| Name of 3rd party code | License | Web page location |
|------------------------|---------|-------------------|
| antlr4-python3-runtime | BSD-3-Clause | https://github.com/antlr/antlr4/blob/dev/LICENSE.txt |
| datasets | Apache-2.0 | https://github.com/huggingface/datasets/blob/main/LICENSE |
| flash-attn | BSD-3-Clause | https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE |
| huggingface_hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub/blob/main/LICENSE |
| lm-evaluation-harness | MIT | https://github.com/EleutherAI/lm-evaluation-harness/blob/main/LICENSE.md |
| Math-Verify | Apache-2.0 | https://github.com/huggingface/Math-Verify/blob/main/LICENSE |
| matplotlib | Matplotlib License (PSF-based) | https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE |
| numpy | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| pandas | BSD-3-Clause | https://github.com/pandas-dev/pandas/blob/main/LICENSE |
| pyarrow | Apache-2.0 | https://github.com/apache/arrow/blob/main/LICENSE.txt |
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| safetensors | Apache-2.0 | https://github.com/huggingface/safetensors/blob/main/LICENSE |
| scikit-learn | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| scipy | BSD-3-Clause | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| seaborn | BSD-3-Clause | https://github.com/mwaskom/seaborn/blob/master/LICENSE.md |
| SymPy | BSD-3-Clause | https://github.com/sympy/sympy/blob/master/LICENSE |
| tqdm | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm/blob/master/LICENCE |
| transformers | Apache-2.0 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| TRL | Apache-2.0 | https://github.com/huggingface/trl/blob/main/LICENSE |
| vLLM | Apache-2.0 | https://github.com/vllm-project/vllm/blob/main/LICENSE |

The model code in `glr_qwen3/` is adapted from the Qwen3 implementation in transformers (Apache-2.0) and keeps its original copyright notice. The answer normalization in `src/metric/minerva_math500_score.py` and `src/metric/olympiadbench_oe_to_maths_en_comp_score.py` and the task configurations in `lm_eval_tasks/` are adapted from lm-evaluation-harness (MIT).

# THIRD PARTY MODELS

| Name of 3rd party model | License | Web page location |
|-------------------------|---------|-------------------|
| Qwen3-0.6B | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-0.6B |
| Qwen3-1.7B | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-1.7B |

# THIRD PARTY DATASET

None of these datasets is redistributed in this repository; the scripts download them from the Hugging Face Hub.

| Name of 3rd party dataset | License | Web page location |
|---------------------------|---------|-------------------|
| Mixture-of-Thoughts (math; sourced from OpenR1-Math-220k, Apache-2.0) | Not specified | https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts |
| GSM8K | MIT | https://huggingface.co/datasets/openai/gsm8k |
| MATH-500 (from PRM800K and MATH, both MIT) | Not specified on the dataset card; sources MIT | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 |
| SVAMP | MIT | https://huggingface.co/datasets/ChilleD/SVAMP |
| MultiArith | Not specified | https://huggingface.co/datasets/ChilleD/MultiArith |
| AMC23 | Not specified | https://huggingface.co/datasets/math-ai/amc23 |
| OlympiadBench | Apache-2.0 | https://huggingface.co/datasets/Hothan/OlympiadBench |
