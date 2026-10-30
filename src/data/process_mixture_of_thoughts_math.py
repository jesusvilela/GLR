# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from dataclasses import MISSING, asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@dataclass
class RunConfig:
    # REQUIRED
    out_dataset: str = field(
        default=MISSING,
        metadata={"help": "Output dataset name/path for save_to_disk()."},
    )
    tokenizer_path: str = field(
        default=MISSING, metadata={"help": "Tokenizer directory or HF model id."}
    )

    # DATASET SOURCE
    dataset_inp_dir: str = field(
        default=MISSING,
        metadata={"help": "HF dataset name/path."},
    )
    split: str = field(
        default="train",
        metadata={"help": "Dataset split to load."},
    )
    num_proc: int = field(
        default=1, metadata={"help": "Number of processes for multiprocessing."}
    )

    # FIELD NAMES / OUTPUT OPTIONS
    messages_field: str = field(
        default="messages", metadata={"help": "Conversation/messages column name."}
    )
    tokenized_dataset_field: str = field(
        default="text", metadata={"help": "Output text column name."}
    )
    include_system_prompt: bool = field(
        default=False, metadata={"help": "Prepend a system prompt if true."}
    )
    system_prompt: str = field(
        default=SYSTEM_PROMPT, metadata={"help": "System prompt content."}
    )
    enable_thinking: bool = field(
        default=True,
        metadata={"help": "Pass enable_thinking=True to chat template when supported."},
    )


def apply_chat_template_compat(
    tokenizer, messages: List[Dict[str, Any]], enable_thinking: bool
) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=False)

    # for math at least, <think> and </think> already there in assistant
    return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)

    # if enable_thinking:
    #     try:
    #         return tokenizer.apply_chat_template(
    #             messages, enable_thinking=True, **kwargs
    #         )
    #     except TypeError:
    #         logging.warning(
    #             "Tokenizer does not support enable_thinking=True. Falling back."
    #         )

    # return tokenizer.apply_chat_template(messages, **kwargs)


def normalize_messages(raw_messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        raise ValueError(
            f"Expected a list of at least 2 messages, got: {type(raw_messages)}"
        )

    normalized = []
    n = len(raw_messages)

    for i, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Each message must be a dict, got: {type(msg)}")

        role = msg.get("role")
        if role is None:
            # Fallback for datasets where role is omitted.
            if i == 0:
                role = "user"
            elif i == n - 1:
                role = "assistant"
            else:
                raise ValueError(
                    "Message role missing in a non-terminal position; cannot infer safely."
                )

        content = msg.get("content")
        if content is None:
            content = ""

        new_msg = {"role": role, "content": content}

        # Preserve reasoning_content if it already exists in the dataset.
        if "reasoning_content" in msg and msg["reasoning_content"] not in (None, ""):
            new_msg["reasoning_content"] = msg["reasoning_content"]

        normalized.append(new_msg)

    if normalized[-1]["role"] != "assistant":
        raise ValueError(
            f"Last message must be assistant, got: {normalized[-1]['role']}"
        )

    if not any(m["role"] == "user" for m in normalized[:-1]):
        raise ValueError(
            "Expected at least one user message before the final assistant message."
        )

    return normalized


def process_messages_example(
    tokenizer, config: RunConfig, ex: Dict[str, Any]
) -> Dict[str, str]:
    messages = normalize_messages(ex[config.messages_field])

    if config.include_system_prompt:
        has_system = len(messages) > 0 and messages[0]["role"] == "system"
        if not has_system:
            messages = [{"role": "system", "content": config.system_prompt}] + messages

    text = apply_chat_template_compat(
        tokenizer, messages, enable_thinking=config.enable_thinking
    )
    prompt = apply_chat_template_compat(
        tokenizer, messages[:-1], enable_thinking=config.enable_thinking
    )

    # Safer than rendering assistant alone for Qwen-style thinking traces.
    completion = text[len(prompt) :]

    num_tokens = len(tokenizer.encode_plus(text).input_ids)

    return {
        config.tokenized_dataset_field: text,
        "prompt": prompt,
        "completion": completion,
        "num_tokens": num_tokens,
    }


def tokenize_dataset(args: RunConfig, tokenizer, ds):
    process_fn = partial(process_messages_example, tokenizer, args)
    tokenized_dataset = ds.map(
        process_fn,
        batched=False,
        num_proc=args.num_proc,
        desc="Formatting SFT data from messages",
    )
    return tokenized_dataset


def assert_columns(ds, args: RunConfig):
    cols = set(ds.column_names)
    needed = {args.messages_field}
    missing = needed - cols
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Available columns: {sorted(cols)}"
        )


def main():
    parser = HfArgumentParser((RunConfig,))
    (args,) = parser.parse_args_into_dataclasses()

    logging.info(f"Run config: {asdict(args)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logging.info(f"Loaded tokenizer from {args.tokenizer_path}")

    ds = load_dataset(
        "parquet",
        data_files=f"{args.dataset_inp_dir}/*.parquet",
    )[args.split]
    assert_columns(ds, args)
    logging.info(
        f"Loaded dataset {args.dataset_inp_dir} / {args.split}"
    )

    tokenized_dataset = tokenize_dataset(args, tokenizer, ds)
    logging.info("Formatting complete.")

    tokenized_dataset.save_to_disk(args.out_dataset)
    logging.info(f"Saved tokenized dataset to {args.out_dataset}")


if __name__ == "__main__":
    main()
