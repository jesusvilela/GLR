# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config
from glr_qwen3.modeling_qwen3_latent import (
    CustomQwen3ForCausalLM,
)
from transformers import AutoTokenizer
import torch
import json
from tqdm import tqdm
import argparse
import os
from glr_paths import get

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the model")
parser.add_argument("--out_dir", type=str, help="Directory for the generations")
parser.add_argument(
    "--max_gen_tokens",
    type=int,
    help="Maximum number of tokens to generate",
    default=8192,
)
parser.add_argument(
    "--sampling_temp",
    type=float,
    help="Sampling temperature for generation",
    default=0.0,
)

parser.add_argument(
    "--latent_token_budget",
    nargs="+",
    type=int,
    default=[100],
    help="Latent token budget for reasoning",
)
parser.add_argument(
    "--all_test_set_paths",
    nargs="+",
    type=str,
    default=[],
    help="Paths to all test sets",
)
args = parser.parse_args()
print(args)

# attn impl benchmark
# sdpa, 10 trials, max_new_tokens=128, 3.42 s ± 8.99 ms per loop (mean ± std. dev. of 10 runs, 1 loop each)
# eager, same params, 3.79 s ± 3.24 ms per loop (mean ± std. dev. of 10 runs, 1 loop each)
# default, same params, 3.59 s ± 6.26 ms per loop (mean ± std. dev. of 10 runs, 1 loop each)

cache = None  # Hugging Face cache: HF_HOME (set from glr_paths.py by scripts/common.sh)

test_set_dir = get("eval_dir")

all_test_set_paths = args.all_test_set_paths

print(f"Model path: {args.model_path}")
print(f"All test set paths: {all_test_set_paths}")
for latent_token_budget in args.latent_token_budget:
    latent_reason_token_budget = latent_token_budget
    print(
        f"==============> Running inference with latent token budget: {latent_reason_token_budget}"
    )

    max_gen_tokens = args.max_gen_tokens
    print(
        f"Set max_gen_tokens to {max_gen_tokens} based on latent_reason_token_budget: {latent_reason_token_budget}"
    )

    custom_config = {"latent_reason_token_budget": latent_reason_token_budget}

    model_path = args.model_path

    tok = AutoTokenizer.from_pretrained(model_path, cache_dir=cache)
    config = CustomQwen3Config.from_pretrained(
        model_path, cache_dir=cache, **custom_config
    )
    # print(config)
    model = CustomQwen3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        cache_dir=cache,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()

    decoding_kwargs = {
        "max_new_tokens": max_gen_tokens,
        "logits_to_keep": 1,
    }

    if args.sampling_temp > 0.0:
        decoding_kwargs["temperature"] = args.sampling_temp
        decoding_kwargs["do_sample"] = True
    else:
        decoding_kwargs["temperature"] = 0.0
        decoding_kwargs["do_sample"] = False

    print(f"Decoding kwargs: {decoding_kwargs}")

    for test_set_jsonl in all_test_set_paths:

        test_set_path = f"{test_set_dir}/{test_set_jsonl}"
        test_set = [json.loads(line) for line in open(test_set_path) if line.strip()]
        test_set_name = test_set_jsonl.split(".")[0]
        print(
            f"----> Test Set: {test_set_path}; Name: {test_set_name}; total {len(test_set)} samples"
        )
        os.makedirs(args.out_dir, exist_ok=True)
        save_path = f"{args.out_dir}/{test_set_name}_latent{latent_reason_token_budget}_gen{max_gen_tokens}_temp{args.sampling_temp}.json"

        with open(save_path, "w") as f_out:
            for inp in tqdm(test_set, desc="generating"):
                with torch.inference_mode():
                    inputs = tok(inp["q"], return_tensors="pt").to(model.device)
                    outputs = model.generate(
                        **inputs,
                        **decoding_kwargs,
                    )
                out_text = tok.decode(outputs[0], skip_special_tokens=False)
                to_dump = {
                    "question": inp["q"],
                    "answer": inp["a"],
                    "predicted": out_text,
                }
                f_out.write(json.dumps(to_dump) + "\n")
                f_out.flush()

        print(f"Saved predictions to {save_path}")

    del model
