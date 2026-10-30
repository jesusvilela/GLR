# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
]


def load_jsonl(
    input_path: str, prompt_field: str
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    prompts: list[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if isinstance(obj, str):
                prompt = obj
                row = {prompt_field: prompt}
            elif isinstance(obj, dict):
                if prompt_field not in obj:
                    raise KeyError(
                        f"Line {line_idx} missing field '{prompt_field}'. "
                        f"Available fields: {list(obj.keys())}"
                    )
                prompt = obj[prompt_field]
                row = obj
            else:
                raise ValueError(
                    f"Line {line_idx} must be a JSON object or JSON string."
                )

            if not isinstance(prompt, str):
                raise ValueError(
                    f"Line {line_idx} field '{prompt_field}' must be a string."
                )

            rows.append(row)
            prompts.append(prompt)

    return rows, prompts


def save_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sampling_params(max_new_tokens: int) -> SamplingParams:
    return SamplingParams(
        # Equivalent to HF max_new_tokens
        max_tokens=max_new_tokens,
        # Equivalent to HF:
        # do_sample=False
        # temperature=0.0
        temperature=0.0,
        # Explicitly disable sampling filters
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        # Neutral penalties
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        # Stop when EOS is generated, just like normal HF generate()
        ignore_eos=False,
        # Keep special tokens in vLLM's own decoded completion text.
        # We still manually decode token IDs below for exact HF-style full output.
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )


def reconstruct_hf_style_output(
    tokenizer: AutoTokenizer,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    finish_reason: str | None,
    stop_reason: str | int | None,
) -> tuple[str, list[int]]:
    """
    Reconstruct equivalent of:

        tok.decode(outputs[0], skip_special_tokens=False)

    where HF outputs[0] contains prompt tokens + generated tokens.

    vLLM returns prompt token IDs and generated token IDs separately.
    """

    eos_token_id = tokenizer.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_ids = [eos_token_id]
    elif isinstance(eos_token_id, list):
        eos_token_ids = eos_token_id
    else:
        eos_token_ids = []

    generated_token_ids = list(generated_token_ids)

    # In most vLLM versions EOS is already present in generated_token_ids.
    # This guard helps mimic HF if EOS caused stopping but was omitted.
    # It avoids double-appending if EOS is already there.
    if (
        finish_reason == "stop"
        and stop_reason is None
        and eos_token_ids
        and (not generated_token_ids or generated_token_ids[-1] not in eos_token_ids)
    ):
        generated_token_ids.append(eos_token_ids[0])

    full_token_ids = list(prompt_token_ids) + generated_token_ids

    decoded_text = tokenizer.decode(
        full_token_ids,
        skip_special_tokens=False,
    )

    return decoded_text, full_token_ids


def run_one_model(
    model_name_or_path: str,
    rows: list[dict[str, Any]],
    prompts: list[str],
    args: argparse.Namespace,
) -> None:
    print(f"\nLoading tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading vLLM model: {model_name_or_path}")

    llm_kwargs: dict[str, Any] = {
        "model": model_name_or_path,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.cache_dir is not None:
        llm_kwargs["download_dir"] = args.cache_dir

    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    sampling_params = build_sampling_params(args.max_new_tokens)

    print(f"Generating {len(prompts)} prompts with vLLM auto-batching...")

    # vLLM handles batching/scheduling internally.
    outputs = llm.generate(prompts, sampling_params)

    output_rows: list[dict[str, Any]] = []

    for row, request_output in zip(rows, outputs):
        completion = request_output.outputs[0]

        prompt_token_ids = list(request_output.prompt_token_ids)
        generated_token_ids = list(completion.token_ids)

        decoded_output, full_token_ids = reconstruct_hf_style_output(
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            finish_reason=completion.finish_reason,
            stop_reason=completion.stop_reason,
        )

        new_row = dict(row)
        new_row["model"] = model_name_or_path

        # This is equivalent to:
        # tok.decode(outputs[0], skip_special_tokens=False)
        new_row[args.output_field] = decoded_output

        if args.save_completion_only:
            new_row["completion_only"] = tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=False,
            )

        if args.save_token_ids:
            new_row["prompt_token_ids"] = prompt_token_ids
            new_row["generated_token_ids"] = generated_token_ids
            new_row["full_token_ids"] = full_token_ids

        if args.save_finish_reason:
            new_row["finish_reason"] = completion.finish_reason
            new_row["stop_reason"] = completion.stop_reason

        output_rows.append(new_row)

    safe_model_name = model_name_or_path.rstrip("/").replace("/", "__")
    output_path = Path(args.out_dir) / f"{args.save_file}_{safe_model_name}.jsonl"

    save_jsonl(output_path, output_rows)

    print(f"Saved: {output_path}")

    del llm
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HF-style decoded Qwen3 outputs from already chat-templated JSONL prompts using vLLM."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        default="generations",
        help="Output directory.",
    )
    parser.add_argument(
        "--save-file",
        required=True,
        help="output JSONL file.",
    )
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="JSON field containing already chat-templated prompt text.",
    )
    parser.add_argument(
        "--output-field",
        default="decoded_output",
        help="Output field for full decoded prompt + generation.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names or local model paths.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Equivalent to HF max_new_tokens.",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Use >1 to shard a model across multiple GPUs.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max context length override.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )

    parser.add_argument(
        "--save-completion-only",
        action="store_true",
        help="Also save generated completion text only.",
    )
    parser.add_argument(
        "--save-token-ids",
        action="store_true",
        help="Also save prompt/generated/full token IDs.",
    )
    parser.add_argument(
        "--save-finish-reason",
        action="store_true",
        help="Also save vLLM finish_reason and stop_reason.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows, prompts = load_jsonl(
        input_path=args.input,
        prompt_field=args.prompt_field,
    )

    if not prompts:
        raise ValueError("No prompts found in input file.")

    print(f"Loaded {len(prompts)} prompts from {args.input}")

    for model_name_or_path in args.models:
        run_one_model(
            model_name_or_path=model_name_or_path,
            rows=rows,
            prompts=prompts,
            args=args,
        )


if __name__ == "__main__":
    main()
