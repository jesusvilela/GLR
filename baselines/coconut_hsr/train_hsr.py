#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Train the Qwen3 parallel hidden-state replacement baseline."""

import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import transformers
import trl
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.coconut_hsr.modeling_hsr import (  # noqa: E402
    HSRForCausalLM,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-0.6B")
    block_size: int = field(default=8192)
    wandb_project: Optional[str] = field(default="qwen3-hsr")
    train_file_path: Optional[str] = field(default=None)
    freeze_input_embeddings: bool = field(default=True)

    def __post_init__(self) -> None:
        os.environ["WANDB_PROJECT"] = str(self.wandb_project)


def _param_global_numel(parameter: torch.nn.Parameter) -> int:
    sharded_numel = getattr(parameter, "ds_numel", None)
    if isinstance(sharded_numel, int) and sharded_numel > 0:
        return sharded_numel
    try:
        numel = parameter.numel()
    except Exception:
        numel = 0
    if numel == 0 and hasattr(parameter, "shape"):
        numel = 1
        for dimension in parameter.shape:
            numel *= int(dimension)
    return int(numel)


def freeze_input_embeddings(model: HSRForCausalLM) -> None:
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError("Model has no input embedding layer.")
    for parameter in embeddings.parameters():
        parameter.requires_grad_(False)
    total = sum(_param_global_numel(p) for p in model.parameters())
    trainable = sum(
        _param_global_numel(p) for p in model.parameters() if p.requires_grad
    )
    logging.info("Input embeddings frozen: True")
    logging.info(
        "Trainable params (global): %d/%d (%.2f%%)",
        trainable,
        total,
        100.0 * trainable / total,
    )


def log_model_and_parameters(model: HSRForCausalLM) -> None:
    """Dump the architecture and effective parameter state for auditability."""
    parameters = list(model.named_parameters())
    total = sum(_param_global_numel(parameter) for _, parameter in parameters)
    trainable = sum(
        _param_global_numel(parameter)
        for _, parameter in parameters
        if parameter.requires_grad
    )
    frozen = total - trainable
    logging.info("Model architecture:\n%s", model)
    logging.info("Model config:\n%s", model.config)
    logging.info(
        "Model parameter summary: total=%d, trainable=%d, frozen=%d, "
        "trainable_pct=%.4f",
        total,
        trainable,
        frozen,
        100.0 * trainable / total,
    )
    for name, parameter in parameters:
        logging.info(
            "PARAM name=%s shape=%s dtype=%s requires_grad=%s numel=%d",
            name,
            tuple(parameter.shape),
            parameter.dtype,
            parameter.requires_grad,
            _param_global_numel(parameter),
        )


def filter_dataset(tokenizer, max_len: int, min_len: int, example: dict) -> bool:
    num_tokens = example.get("num_tokens")
    if num_tokens is None:
        num_tokens = len(tokenizer.encode_plus(example["text"]).input_ids)
    return min_len <= num_tokens <= max_len


def marker_ids(tokenizer, marker: str) -> list[int]:
    ids = tokenizer.encode(marker, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected {marker!r} to be one token for Qwen3, found {ids}.")
    return [int(ids[0])]


def train() -> None:
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()

    if getattr(args, "gradient_checkpointing", False):
        checkpointing_kwargs = (
            getattr(args, "gradient_checkpointing_kwargs", None) or {}
        )
        checkpointing_kwargs["use_reentrant"] = False
        args.gradient_checkpointing_kwargs = checkpointing_kwargs
        logging.info(
            "Gradient checkpointing enabled with kwargs=%s",
            args.gradient_checkpointing_kwargs,
        )

    # Keep the same TRL preprocessing and completion collator as the GLR run.
    args.dataset_text_field = "text"
    args.max_length = config.block_size
    args.max_seq_length = config.block_size
    args.completion_only_loss = True
    logging.info(
        "Training config: %s",
        {**asdict(config), **asdict(args)},
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_config = AutoConfig.from_pretrained(config.model_name)
    model_config.hsr_think_start_ids = marker_ids(tokenizer, "<think>")
    model_config.hsr_think_end_ids = marker_ids(tokenizer, "</think>")
    model_config.hsr_detach_first_pass = False
    model_config.hsr_training_method = (
        "two_pass_teacher_forced_hidden_state_replacement"
    )
    model = HSRForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    logging.info("HSR config: %s", model.config)
    logging.info(
        "First-pass hidden-state replacements are detached: %s",
        model.config.hsr_detach_first_pass,
    )

    if config.freeze_input_embeddings:
        freeze_input_embeddings(model)
        if getattr(args, "gradient_checkpointing", False):
            model.enable_input_require_grads()
            logging.info("Enabled input require grads for gradient checkpointing.")
    log_model_and_parameters(model)

    dataset = load_from_disk(config.train_file_path)
    logging.info("Train dataset loaded: %s", dataset)
    dataset = dataset.filter(
        partial(filter_dataset, tokenizer, config.block_size, 0),
        desc=f"Filtering more than {config.block_size} tokens",
    )
    dataset = dataset.remove_columns(
        [
            column
            for column in dataset.column_names
            if column not in ["prompt", "completion"]
        ]
    )
    logging.info("Filtered train dataset: %s", dataset)

    pad_token = tokenizer.pad_token or tokenizer.eos_token
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    padding_free = args.padding_free or (
        args.packing and args.packing_strategy == "bfd"
    )
    collator = DataCollatorForLanguageModeling(
        pad_token_id=pad_token_id,
        completion_only_loss=True,
        padding_free=padding_free,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=args,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()
    logging.info("Final model and tokenizer saved to %s", args.output_dir)


if __name__ == "__main__":
    train()
