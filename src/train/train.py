# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from datasets import load_from_disk
import transformers
import trl
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from enum import Enum
import torch
from functools import partial


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-0.6B")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="qwen3-latent")
    train_file_path: Optional[str] = field(default=None)
    freeze_input_embeddings: bool = field(default=False)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project
        # os.environ["WANDB_ENTITY"] = self.wandb_entity


class LatentReasonType(str, Enum):
    DIRECTIONAL = "directional"
    LANGEVIN = "langevin"
    OTHERS = "others"


class CriteriaLatentReasonType(str, Enum):
    MSE = "mse"
    SIMILARITY = "similarity"


@dataclass
class LatentReasoningConfig:
    enable_latent_reason: bool = field(default=False)
    latent_reason_type: LatentReasonType = field(default=LatentReasonType.DIRECTIONAL)
    latent_reason_pred_delta_embed: bool = field(default=False)
    latent_reason_discount_factor: float = field(default=0.999)
    latent_reason_teacher_forcing: bool = field(default=True)
    latent_reason_criteria: CriteriaLatentReasonType = field(
        default=CriteriaLatentReasonType.MSE
    )
    ce_latent_tokens: bool = field(default=True)


class AuxLossLoggerCallback(transformers.TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        # Save references that aren't provided to every hook
        self.trainer = kwargs.get("trainer", None)
        self.model = kwargs.get("model", None)

    def _maybe_log(self):
        # print(f"maybe log, {self.model is None}")

        logs = {
            k: (v.item() if isinstance(v, torch.Tensor) else float(v))
            for k, v in self.model._losses_for_log.items()
        }
        print(logs)

        if not getattr(self, "trainer", None) or not getattr(self, "model", None):
            return
        if not self.trainer.is_world_process_zero():
            return
        if hasattr(self.model, "_losses_for_log") and self.model._losses_for_log:
            logs = {
                k: (v.item() if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.model._losses_for_log.items()
            }
            print(logs)
            # send to HF Trainer logging (console + W&B if enabled)
            self.trainer.log(logs)

    # Fires each *batch* (even with grad accumulation)
    def on_substep_end(self, args, state, control, **kwargs):
        self._maybe_log()

    # Alternatively, if you only want once per optimizer step:
    # def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
    #     self._maybe_log()


def _param_global_numel(p: torch.nn.Parameter) -> int:
    # DeepSpeed ZeRO-3 partitioned parameters often expose the original size here
    n = getattr(p, "ds_numel", None)
    if isinstance(n, int) and n > 0:
        return n

    # Normal case
    try:
        n = p.numel()
    except Exception:
        n = 0

    # Fallback: product of shape (covers some sharded/meta edge cases)
    if n == 0 and hasattr(p, "shape"):
        n = 1
        for d in p.shape:
            n *= int(d)
    return int(n)


def freeze_input_embeddings(model):
    emb = model.get_input_embeddings()
    if emb is None:
        raise ValueError(
            "Model has no input embeddings (get_input_embeddings() returned None)."
        )

    # Freeze
    for p in emb.parameters():
        p.requires_grad_(False)

    # Sanity: embedding params should now be frozen
    emb_trainable = any(p.requires_grad for p in emb.parameters())
    logging.info(f"Input embeddings frozen: {not emb_trainable}")

    # Robust parameter counting under ZeRO-3
    total = 0
    trainable = 0
    for p in model.parameters():
        n = _param_global_numel(p)
        total += n
        if p.requires_grad:
            trainable += n

    if total > 0:
        pct = 100.0 * trainable / total
        logging.info(f"Trainable params (global): {trainable}/{total} ({pct:.2f}%)")
    else:
        logging.info(
            "Trainable params (global): total=0 (unable to compute under current sharding/offload state)."
        )


def filter_dataset(tokenizer, max_len, min_len, ex):
    # instance level filtering
    num_tokens = ex.get("num_tokens", None)
    if num_tokens is None:
        num_tokens = len(tokenizer.encode_plus(ex["text"]).input_ids)
    return min_len <= num_tokens <= max_len


def train():
    # parsing input
    parser = transformers.HfArgumentParser(
        (LatentReasoningConfig, TrainingConfig, trl.SFTConfig)
    )
    latent_config, config, args = parser.parse_args_into_dataclasses()

    if getattr(args, "gradient_checkpointing", False):
        gckw = getattr(args, "gradient_checkpointing_kwargs", None) or {}
        gckw["use_reentrant"] = False
        args.gradient_checkpointing_kwargs = gckw
        logging.info(
            f"Gradient checkpointing enabled with kwargs={args.gradient_checkpointing_kwargs}"
        )

    ## fixing other variables
    args.dataset_text_field = "text"
    args.max_length = config.block_size
    args.max_seq_length = config.block_size
    args.completion_only_loss = True

    log_config = {**asdict(latent_config), **asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    aux_loggers = []
    kwargs = {}
    if latent_config.enable_latent_reason:
        if latent_config.latent_reason_type == LatentReasonType.DIRECTIONAL:
            from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config
            from glr_qwen3.modeling_qwen3_latent import (
                CustomQwen3ForCausalLM,
            )

            config_kwargs = {
                "enable_latent_reason": latent_config.enable_latent_reason,
                "latent_reason_type": LatentReasonType.DIRECTIONAL.value,
                "latent_reason_pred_delta_embed": latent_config.latent_reason_pred_delta_embed,
                "latent_reason_discount_factor": latent_config.latent_reason_discount_factor,
                "latent_max_sequence_length": config.block_size,
                "latent_reason_teacher_forcing": latent_config.latent_reason_teacher_forcing,
                "latent_reason_criteria": latent_config.latent_reason_criteria,
                "ce_latent_tokens": latent_config.ce_latent_tokens,
            }
            model_kwargs = {
                # "device_map": "auto",
                "torch_dtype": "auto",
                "attn_implementation": "flash_attention_2",
                # "use_cache": False,
            }
            custom_config = CustomQwen3Config.from_pretrained(
                config.model_name, **config_kwargs
            )
            logging.info(f"Custom model config: {custom_config}")
            model = CustomQwen3ForCausalLM.from_pretrained(
                config.model_name, config=custom_config, **model_kwargs
            )
            aux_loggers.append(AuxLossLoggerCallback())
        else:
            raise NotImplementedError
    else:
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name, **kwargs
        )

    logging.info(f"{model}")
    logging.info(f"{model.config}")

    # fixing input embeddings
    if config.freeze_input_embeddings:
        logging.info("Freezing input embeddings...")
        freeze_input_embeddings(model)

        if getattr(args, "gradient_checkpointing", False):
            model.enable_input_require_grads()
            logging.info("Enabled input require grads for gradient checkpointing.")

    # loading dataset from disk
    dataset = load_from_disk(config.train_file_path)
    # dataset = load_dataset(
    #     "parquet", data_files=f"{config.train_file_path}/data/*.parquet"
    # )["train"]
    # dataset = dataset.select(range(32))  # For testing purpose only
    dataset_test = None
    logging.info(f"Train dataset loaded: {dataset}")
    logging.info(f"Test dataset loaded: {dataset_test}")

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)

    # filtering dataset, remove if matched
    dataset = dataset.filter(
        partial(filter_dataset, tokenizer, config.block_size, 0),
        desc=f"Filtering more than {config.block_size} tokens",
    )
    logging.info(f"After Filtering Train dataset loaded: {dataset}")
    logging.info(f"After Filtering Test dataset loaded: {dataset_test}")

    # removing unused columns
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ["prompt", "completion"]]
    )

    # Only compute loss over assistant responses
    # loading data collator
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
        eval_dataset=dataset_test if dataset_test is not None else dataset,
        args=args,
        data_collator=collator,
        callbacks=aux_loggers,
    )

    # training
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
