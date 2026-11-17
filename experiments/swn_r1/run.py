"""Full GLR checkpoint evaluation: 1-pass and Teloi-governed 2-pass arms.

Use a TRAINED GLR checkpoint, never the vanilla base model: the direction
head is essential. Ground truth is stored for post-hoc scoring only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import torch
from safetensors import safe_open

from experiments.swn_r1.adapter import LatentDeltaBridge, RuntimeTelos


ARMS = ("glr", "swn", "random", "glr_2pass", "swn_2pass", "random_2pass")


def verify_trained_glr_checkpoint(path: Path) -> None:
    """Fail closed rather than silently evaluating a randomly initialized head."""
    cfg = path / "config.json"
    if not cfg.is_file():
        raise FileNotFoundError("trained GLR checkpoint must be a local directory with config.json")
    data = json.loads(cfg.read_text(encoding="utf-8"))
    if data.get("enable_latent_reason") is not True:
        raise ValueError("checkpoint is not configured for GLR latent reasoning")
    if data.get("latent_reason_pred_delta_embed") is not True:
        raise ValueError("checkpoint is not in the paper's additive delta mode")
    keys = set()
    index = path / "model.safetensors.index.json"
    if index.exists():
        keys.update(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).keys())
    else:
        for fn in path.glob("*.safetensors"):
            with safe_open(str(fn), framework="pt", device="cpu") as obj:
                keys.update(obj.keys())
    if not ("model.direction_delta_predictor.weight" in keys
            and "model.direction_delta_predictor.bias" in keys):
        raise ValueError("checkpoint lacks saved trained GLR transition-head tensors; do not use base Qwen3")


def question_text(record: dict) -> str:
    if isinstance(record.get("problem"), str) and record["problem"].strip():
        return record["problem"].strip()
    prompt = record["q"]
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, flags=re.S)
    if match:
        return match.group(1).strip()
    raise ValueError("two-pass requires the original problem or a parseable Qwen3 user turn")


def refinement_prompt(tokenizer, raw_question: str, first_answer: str) -> str:
    # New causal prompt, not backward attention and not a weight update.
    # Draft is treated as fallible data, not an instruction or an oracle.
    after_think = first_answer.rsplit("</think>", 1)[-1]
    draft = after_think[:2500].strip() or first_answer[:2500]
    messages = [
        {"role": "user", "content": raw_question},
        {"role": "assistant", "content": draft},
        {"role": "user", "content":
         "Independently check the previous solution. Correct errors if found. "
         "Give the verified final answer explicitly. Do not assume the draft is correct."},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


@torch.inference_mode()
def generate_one(model, tok, prompt: str, *, arm: str, strength: float,
                 K: int, max_new_tokens: int, telos: RuntimeTelos, seed: int):
    mode = arm.replace("_2pass", "")
    model.config.latent_reason_token_budget = K
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()
    with LatentDeltaBridge(model, mode=mode, strength=strength,
                           random_seed=seed, telos=telos) as bridge:
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True, logits_to_keep=1,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - tic
    tokens = out[0, inputs["input_ids"].size(-1):]
    return {
        "text": tok.decode(tokens, skip_special_tokens=False),
        "generated_steps": int(tokens.numel()),
        "wall_seconds": elapsed,
        "telos_audit": bridge.audit.as_dict(),
    }


def run_one_record(model, tok, rec: dict, *, arm: str, strength: float,
                   K: int, budget: int, telos: RuntimeTelos, seed: int):
    first = generate_one(model, tok, rec["q"], arm=arm, strength=strength,
                         K=K, max_new_tokens=budget, telos=telos, seed=seed)
    if arm.endswith("_2pass"):
        prompt = refinement_prompt(tok, question_text(rec), first["text"])
        second = generate_one(model, tok, prompt, arm=arm, strength=strength,
                              K=K, max_new_tokens=budget, telos=telos, seed=seed)
    else:
        second = None
    phases = [first] + ([second] if second is not None else [])
    return {
        "predicted": phases[-1]["text"],
        "first_predicted": first["text"] if second is not None else None,
        "generated_steps": sum(p["generated_steps"] for p in phases),
        "wall_seconds": sum(p["wall_seconds"] for p in phases),
        "fallbacks": sum(p["telos_audit"]["reverted"] for p in phases),
        "transition_proposals": sum(p["telos_audit"]["proposed"] for p in phases),
        "max_observed_ratio": max(p["telos_audit"]["max_observed_ratio"] for p in phases),
        "passes": len(phases),
        "step_breakdown": [p["generated_steps"] for p in phases],
        "telos_audits": [p["telos_audit"] for p in phases],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.02)
    parser.add_argument("--max-relative-perturbation", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if args.max_examples < 1 or args.K < 0 or args.max_new_tokens < 1:
        parser.error("max-examples and max-new-tokens must be positive, K must be nonnegative")
    if len(set(args.arms)) != len(args.arms):
        parser.error("each arm may appear only once")
    verify_trained_glr_checkpoint(args.checkpoint)

    # Late imports allow CPU-only unit tests of the Teloi and adapter.
    from transformers import AutoTokenizer
    from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config
    from glr_qwen3.modeling_qwen3_latent import CustomQwen3ForCausalLM

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    cfg = CustomQwen3Config.from_pretrained(args.checkpoint, latent_reason_token_budget=args.K)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = dict(config=cfg, dtype=dtype, attn_implementation="sdpa")
    model = CustomQwen3ForCausalLM.from_pretrained(args.checkpoint, **kwargs).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    telos = RuntimeTelos(args.max_relative_perturbation)

    records = []
    with args.eval_file.open(encoding="utf-8") as inp:
        for line in inp:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= args.max_examples:
                break
    if not records:
        raise ValueError("evaluation file is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint), "eval_file": str(args.eval_file),
        "arms": args.arms, "K": args.K, "strength": args.strength,
        "seed": args.seed, "max_new_tokens_per_pass": args.max_new_tokens,
        "telos_relative_bound": args.max_relative_perturbation,
        "note": "Gold is never passed to the model. A second pass spends more compute.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            for arm in args.arms:
                result = run_one_record(
                    model, tok, rec, arm=arm, strength=args.strength,
                    K=args.K, budget=args.max_new_tokens,
                    telos=telos, seed=args.seed,
                )
                row = {
                    "id": rec.get("id", idx), "arm": arm,
                    "question": rec["q"], "answer": rec.get("a", ""),
                    **result,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"id={row['id']} arm={arm} steps={result['generated_steps']} "
                      f"fallbacks={result['fallbacks']}", flush=True)


if __name__ == "__main__":
    main()
