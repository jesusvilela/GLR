#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Greedy fixed-K and until-close decoding for a trained Qwen3 HSR model.

Latent positions are advanced directly through the KV cache with the preceding
final-layer hidden state as ``inputs_embeds``.  No token is sampled or appended
at a latent position.  Output files are JSONL with the same three record fields
as ``src/inference/generate_glr.py``.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from glr_paths import get  # noqa: E402

TEST_SET_DIR = Path(get("eval_dir"))


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    total_step_budget: int

    @property
    def path(self) -> Path:
        return TEST_SET_DIR / self.filename


DATASETS = {
    "gsm8k": DatasetSpec("gsm8k_zeroshot.jsonl", 2048),
    "svamp": DatasetSpec("svamp.jsonl", 2048),
    "math500": DatasetSpec("minerva_math500_zeroshot.jsonl", 4096),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--latent-steps",
        nargs="+",
        type=int,
        default=[10, 100],
        help="Fixed K values.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("fixed", "until_close"),
        default=["fixed", "until_close"],
        help=(
            "fixed uses exactly K latent steps; until_close keeps taking "
            "latent steps until the greedy next token is </think>."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to the checkpoint directory, like the GLR decoder.",
    )
    parser.add_argument(
        "--until-close-max-latent-steps",
        type=int,
        help=(
            "Safety cap for until-close. By default the benchmark's complete "
            "total-step budget is available to the unended latent phase."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        help="Diagnostic only: decode the first N examples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files rather than resume valid rows.",
    )
    args = parser.parse_args()
    if any(value < 0 for value in args.latent_steps):
        parser.error("--latent-steps values must be non-negative")
    if (
        args.until_close_max_latent_steps is not None
        and args.until_close_max_latent_steps < 1
    ):
        parser.error("--until-close-max-latent-steps must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected an object")
            yield row


def suffix_matches(values: list[int], suffix: list[int]) -> bool:
    return len(values) >= len(suffix) and values[-len(suffix) :] == suffix


class HSRDecoder:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.embedding = model.get_input_embeddings()
        self.device = self.embedding.weight.device
        self.think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
        self.think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if len(self.think_start_ids) != 1 or len(self.think_end_ids) != 1:
            raise ValueError(
                "The HSR Qwen3 decoder requires single-token think markers."
            )
        eos_ids = model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self.stop_ids = {int(value) for value in (eos_ids or [])}

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden)[:, -1, :]

    def _token_embedding(self, token_id: int) -> torch.Tensor:
        ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        return self.embedding(ids)

    def _prefill(self, input_ids: torch.LongTensor):
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, -1:, :]
        return self._logits(hidden), hidden, outputs.past_key_values

    def _advance(self, inputs_embeds: torch.FloatTensor, past_key_values):
        old_length = past_key_values.get_seq_length()
        query_length = inputs_embeds.shape[1]
        cache_position = torch.arange(
            old_length,
            old_length + query_length,
            device=self.device,
        )
        attention_mask = torch.ones(
            (1, old_length + query_length),
            dtype=torch.long,
            device=self.device,
        )
        outputs = self.model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, -1:, :]
        return self._logits(hidden), hidden, outputs.past_key_values

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        mode: str,
        fixed_k: int,
        total_step_budget: int,
        until_close_cap: int,
    ) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(self.device)
        prompt_token_ids = [int(value) for value in prompt_ids[0].tolist()]
        logits, hidden, past = self._prefill(prompt_ids)

        generated_ids: list[int] = []
        latent_steps = 0
        thinking_started = suffix_matches(prompt_token_ids, self.think_start_ids)
        latent_phase_done = False

        while latent_steps + len(generated_ids) < total_step_budget:
            if thinking_started and not latent_phase_done:
                if mode == "fixed":
                    if latent_steps < fixed_k:
                        logits, hidden, past = self._advance(hidden, past)
                        latent_steps += 1
                        continue
                    latent_phase_done = True
                elif mode == "until_close":
                    next_token = int(torch.argmax(logits, dim=-1).item())
                    if next_token == self.think_end_ids[0]:
                        latent_phase_done = True
                    elif latent_steps < until_close_cap:
                        logits, hidden, past = self._advance(hidden, past)
                        latent_steps += 1
                        continue
                    else:
                        latent_phase_done = True
                else:
                    raise ValueError(mode)

            next_token = int(torch.argmax(logits, dim=-1).item())
            generated_ids.append(next_token)
            if next_token in self.stop_ids:
                break
            if latent_steps + len(generated_ids) >= total_step_budget:
                break

            logits, hidden, past = self._advance(
                self._token_embedding(next_token),
                past,
            )
            if not thinking_started and suffix_matches(
                prompt_token_ids + generated_ids,
                self.think_start_ids,
            ):
                thinking_started = True
                if mode == "fixed" and fixed_k == 0:
                    latent_phase_done = True

        return self.tokenizer.decode(
            prompt_token_ids + generated_ids,
            skip_special_tokens=False,
        )


def output_path(
    output_dir: Path,
    spec: DatasetSpec,
    mode: str,
    fixed_k: int,
    until_close_cap: int,
) -> Path:
    test_set_name = Path(spec.filename).stem
    if mode == "fixed":
        latent_name = str(fixed_k)
    else:
        latent_name = f"until_close_kmax{until_close_cap}"
    return output_dir / (
        f"{test_set_name}_latent{latent_name}_"
        f"gen{spec.total_step_budget}_temp0.0.jsonl"
    )


def completed_questions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["question"])
        for row in read_jsonl(path)
        if {"question", "answer", "predicted"} <= row.keys()
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.model_path
    output_dir.mkdir(parents=True, exist_ok=True)
    load_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "attn_implementation": args.attn_implementation,
    }
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **load_kwargs,
    ).eval()
    decoder = HSRDecoder(model, tokenizer)

    tasks: list[tuple[str, int]] = []
    if "fixed" in args.modes:
        tasks.extend(("fixed", value) for value in args.latent_steps)
    if "until_close" in args.modes:
        tasks.append(("until_close", 0))

    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        rows = list(read_jsonl(spec.path))
        if args.limit is not None:
            rows = rows[: args.limit]
        until_close_cap = args.until_close_max_latent_steps or spec.total_step_budget
        for mode, fixed_k in tasks:
            destination = output_path(
                output_dir,
                spec,
                mode,
                fixed_k,
                until_close_cap,
            )
            if args.overwrite and destination.exists():
                destination.unlink()
            done = completed_questions(destination)
            pending = [row for row in rows if str(row["q"]) not in done]
            print(
                f"{dataset_name} {mode} K={fixed_k}: "
                f"{len(done)} complete, {len(pending)} pending -> "
                f"{destination}",
                flush=True,
            )
            with destination.open("a", encoding="utf-8") as output:
                for row in tqdm(
                    pending,
                    desc=f"{dataset_name}:{mode}:{fixed_k}",
                ):
                    predicted = decoder.generate(
                        str(row["q"]),
                        mode=mode,
                        fixed_k=fixed_k,
                        total_step_budget=spec.total_step_budget,
                        until_close_cap=until_close_cap,
                    )
                    output.write(
                        json.dumps(
                            {
                                "question": row["q"],
                                "answer": row["a"],
                                "predicted": predicted,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    output.flush()


if __name__ == "__main__":
    main()
