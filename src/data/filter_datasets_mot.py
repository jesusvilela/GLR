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

from datasets import load_from_disk
import transformers
from functools import partial
from glr_paths import get


def filter_dataset(tokenizer, max_len, min_len, ex):
    # instance level filtering
    num_tokens = ex.get("num_tokens", None)
    if num_tokens is None:
        num_tokens = len(tokenizer.encode_plus(ex["text"]).input_ids)
    return min_len <= num_tokens <= max_len


def filter_dataset_malformed(ex):
    # instance level filtering
    only_one_think = (
        ex["text"].count("<think>") == 1 and ex["text"].count("</think>") == 1
    )
    return only_one_think


def filter_small_cot(tokenizer, min_think_token, ex):
    # instance level filtering
    text = ex["completion"]
    cot_forward = text.split("<think>")[1].split("</think>")[0]
    cot_reverse = text.split("</think>")[0].split("<think>")[1]
    num_tokens_cot_forward = len(tokenizer.encode_plus(cot_forward).input_ids)
    num_tokens_cot_reverse = len(tokenizer.encode_plus(cot_reverse).input_ids)
    return (
        num_tokens_cot_forward >= min_think_token
        and num_tokens_cot_reverse >= min_think_token
    )


max_len_dataset = 8192
min_len_dataset = 300
min_len_cot = 100
max_select = 10000
tokenizer = transformers.AutoTokenizer.from_pretrained(get("qwen3_1.7b"))
dataset = load_from_disk(get("mot_tokenized_dir"))

logging.info(f"Train dataset loaded: {dataset}")

# filtering dataset
dataset = dataset.filter(
    partial(filter_dataset, tokenizer, max_len_dataset, min_len_dataset),
    desc=f"Filtering more than {max_len_dataset} tokens",
    num_proc=6,
)
logging.info(f"After Filtering Train dataset loaded: {dataset}")

# filtering malformed
dataset = dataset.filter(
    filter_dataset_malformed,
    desc="Filtering malformed samples with not exactly one <think> and </think>",
    num_proc=6,
)
logging.info(f"After Filtering malformed Train dataset loaded: {dataset}")

# filtering small cot
dataset = dataset.filter(
    partial(filter_small_cot, tokenizer, min_len_cot),
    desc=f"Filtering <={min_len_cot} cot",
    num_proc=6,
)
logging.info(f"After Filtering small cot Train dataset loaded: {dataset}")

dataset = dataset.shuffle(seed=42)

# save only max_select
dataset = dataset.select(range(max_select))
dataset.save_to_disk(get("mot_train_10k_dir"))
