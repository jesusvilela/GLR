# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""GLR-K decoding with vLLM: same decoding as src/inference/generate_glr.py, much faster.

GLR decoding (glr_qwen3, `CustomQwen3ForCausalLM.generate`) has two parts:

1. Latent prefix. After the model emits `<think>` at position T, the inputs at positions
   T+1, ..., T+K are not token embeddings but latent states, chained with the transition head:
       e_1 = emb(<think>) + delta(h_T),    e_{j+1} = e_j + delta(h_{T+j}),
   where h is the final hidden state. The tokens sampled at these positions stay in the
   output (placeholders) and count as generated steps. A `</think>` or EOS ends the prefix early.
2. Text decoding: ordinary Qwen3 decoding from position T+K+1 on.

Part 1 needs the transition head and only K+1 forward steps, so it runs here in Hugging Face,
batched over prompts. Part 2 is the long part and runs in vLLM, which receives the prompt and
the latent prefix as prompt embeddings. Decoding is greedy, as in the paper.
Outputs have the same file names and fields as generate_glr.py.
"""

import os

# Run the vLLM engine in this process, so that the model registration below applies to it.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from vllm import LLM, SamplingParams

from glr_paths import get
from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config
from glr_qwen3.modeling_qwen3_latent import CustomQwen3ForCausalLM
from vllm_glr_qwen3.model import register


@torch.inference_mode()
def latent_prefix(model, prompt_ids, K, max_gen_tokens, eos_ids, batch_size):
    """Greedy-decode each prompt through its latent prefix (part 1), batched.

    Returns, per prompt, the generated token ids so far and either the input embeddings
    of the whole sequence (to continue in vLLM) or None (generation already finished).
    Also returns the number of generations that did not start with `<think>`.
    """
    think = model.config.think_tokens_start[0]
    end_think = model.config.think_tokens_end[0]
    embed = model.model.embed_tokens
    device = embed.weight.device
    results = [None] * len(prompt_ids)
    no_think = 0

    order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]))
    for start in tqdm(range(0, len(order), batch_size), desc=f"latent prefix (K={K})"):
        rows = order[start : start + batch_size]
        B, L = len(rows), max(len(prompt_ids[i]) for i in rows)

        # left padding
        ids = torch.zeros(B, L, dtype=torch.long, device=device)
        mask = torch.zeros(B, L, dtype=torch.long, device=device)
        for b, i in enumerate(rows):
            n = len(prompt_ids[i])
            ids[b, L - n :] = torch.tensor(prompt_ids[i], device=device)
            mask[b, L - n :] = 1
        position = (mask.cumsum(-1) - 1).clamp(min=0)
        cache = DynamicCache(config=model.config)

        out = model.model(
            inputs_embeds=embed(ids),
            attention_mask=mask,
            position_ids=position,
            past_key_values=cache,
            use_cache=True,
        )
        position = position[:, -1]
        generated = [[] for _ in rows]
        fed = [[] for _ in rows]  # input embeddings of the generated positions
        latent = [None] * B
        running = [True] * B

        while any(running):
            next_tokens = model.lm_head(out.last_hidden_state[:, -1]).argmax(-1).tolist()
            delta = out.direction_delta[:, -1]
            inputs = embed(torch.zeros(B, dtype=torch.long, device=device))

            for b, i in enumerate(rows):
                if not running[b]:
                    continue
                t = next_tokens[b]
                generated[b].append(t)
                s = len(generated[b]) - 1  # the new token is at position T + s, with T = first generated position

                if t in eos_ids or len(generated[b]) >= max_gen_tokens:
                    results[i] = (generated[b], None)
                    running[b] = False
                    continue
                if s == 0 and t == think:
                    e = embed.weight[t]
                elif 1 <= s <= K and t != end_think:
                    e = (embed.weight[generated[b][0]] if s == 1 else latent[b]) + delta[b]
                    latent[b] = e
                else:
                    # prefix done (s = K + 1), ended by </think>, or no <think>: continue in vLLM
                    if s == 0:
                        no_think += 1
                    fed[b].append(embed.weight[t])
                    sequence = torch.cat([embed.weight[prompt_ids[i]], torch.stack(fed[b])])
                    results[i] = (generated[b], sequence.cpu())
                    running[b] = False
                    continue
                fed[b].append(e)
                inputs[b] = e

            if not any(running):
                break
            step_mask = torch.tensor(running, dtype=torch.long, device=device)[:, None]
            mask = torch.cat([mask, step_mask], dim=-1)
            position = position + 1
            out = model.model(
                inputs_embeds=inputs[:, None],
                attention_mask=mask,
                position_ids=position[:, None],
                past_key_values=cache,
                use_cache=True,
            )

    return results, no_think


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", type=str, help="Path to the GLR checkpoint")
    parser.add_argument("--out_dir", type=str, help="Directory for the generations")
    parser.add_argument("--max_gen_tokens", type=int, default=8192, help="Maximum number of generated steps (latent steps included)")
    parser.add_argument("--sampling_temp", type=float, default=0.0, help="Only greedy decoding (0.0) is supported")
    parser.add_argument("--latent_token_budget", nargs="+", type=int, default=[100], help="Latent steps K")
    parser.add_argument("--all_test_set_paths", nargs="+", type=str, default=[], help="Test set files in eval_dir")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of the latent prefix")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="GPU memory fraction for vLLM")
    args = parser.parse_args()
    print(args)
    if args.sampling_temp != 0.0:
        parser.error("only greedy decoding (--sampling_temp 0.0) is supported")

    register()
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        enable_prompt_embeds=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tok = AutoTokenizer.from_pretrained(args.model_path)
    config = CustomQwen3Config.from_pretrained(args.model_path)
    model = CustomQwen3ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    eos_ids = set(model.generation_config.eos_token_id)

    os.makedirs(args.out_dir, exist_ok=True)
    for K in args.latent_token_budget:
        print(f"==============> Running inference with latent token budget: {K}")
        for test_set_jsonl in args.all_test_set_paths:
            test_set_path = f"{get('eval_dir')}/{test_set_jsonl}"
            test_set = [json.loads(line) for line in open(test_set_path) if line.strip()]
            test_set_name = test_set_jsonl.split(".")[0]
            print(f"----> Test Set: {test_set_path}; Name: {test_set_name}; total {len(test_set)} samples")

            prompt_ids = [tok(inp["q"])["input_ids"] for inp in test_set]
            prefix, no_think = latent_prefix(model, prompt_ids, K, args.max_gen_tokens, eos_ids, args.batch_size)

            to_vllm = [i for i, (_, sequence) in enumerate(prefix) if sequence is not None]
            print(
                f"{len(test_set)} samples: {len(to_vllm)} continue in vLLM, "
                f"{len(test_set) - len(to_vllm)} finished within the latent prefix, "
                f"{no_think} did not start with <think> (decoded without latent steps)"
            )
            requests = [{"prompt_embeds": prefix[i][1]} for i in to_vllm]
            sampling_params = [
                SamplingParams(
                    max_tokens=args.max_gen_tokens - len(prefix[i][0]),
                    temperature=0.0,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
                for i in to_vllm
            ]
            continuation = {i: [] for i in range(len(test_set))}
            for i, request_output in zip(to_vllm, llm.generate(requests, sampling_params)):
                continuation[i] = list(request_output.outputs[0].token_ids)

            save_path = f"{args.out_dir}/{test_set_name}_latent{K}_gen{args.max_gen_tokens}_temp{args.sampling_temp}.json"
            with open(save_path, "w") as f_out:
                for i, inp in enumerate(test_set):
                    full_ids = prompt_ids[i] + prefix[i][0] + continuation[i]
                    to_dump = {
                        "question": inp["q"],
                        "answer": inp["a"],
                        "predicted": tok.decode(full_ids, skip_special_tokens=False),
                    }
                    f_out.write(json.dumps(to_dump) + "\n")
            print(f"Saved predictions to {save_path}")


if __name__ == "__main__":
    main()
