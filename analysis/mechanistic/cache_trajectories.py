# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Worker that caches GLR reference, learned, and control trajectories.

The worker wraps the GLR model at runtime instead of editing it.  Normal
generation is still delegated to ``GenerationMixin.generate``; the wrapper
records the exact embeddings/states used by the latent loop and substitutes a
control transition only when requested.
"""

import argparse
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
for path in (REPO_ROOT, ANALYSIS_DIR):
    sys.path.insert(0, str(path))

from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config
from glr_qwen3.modeling_qwen3_latent import CustomQwen3ForCausalLM
from mechanistic_common import (
    ABS_CHECKPOINT_DEFAULT,
    DATASETS,
    DEFAULT_CACHE_ROOT,
    GLR_CHECKPOINT,
    K_STAR,
    K_VALUES,
    cache_run_dir,
    load_dataset,
    score_generation,
    sha256_file,
    sha256_text,
    write_json,
)

METHODS = (
    "reference",
    "learned",
    "zero_transition",
    "repeated_think",
    "shuffled_transition",
    "matched_norm_random_direction",
    "random_head_norm_matched",
    "noisy_transition",
    "absolute_prediction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-sigma", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--tensor-storage-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Deterministic random subset size (preferred to --limit for controls).",
    )
    parser.add_argument("--subset-seed", type=int, default=2026)
    parser.add_argument("--forced-continuation", action="store_true")
    parser.add_argument("--ban-close-token-for", type=int, default=16)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.method == "reference" and args.k != 0:
        parser.error("reference requires --k 0")
    if args.method != "reference" and args.k <= 0:
        parser.error("latent methods require --k > 0")
    if args.method == "learned" and args.k not in K_VALUES:
        parser.error(f"learned K must be one of {K_VALUES}")
    if args.method == "noisy_transition" and args.noise_sigma is None:
        parser.error("noisy_transition requires --noise-sigma")
    if args.method != "noisy_transition" and args.noise_sigma is not None:
        parser.error("--noise-sigma is only valid for noisy_transition")
    if args.forced_continuation and args.method in {"reference", "shuffled_transition"}:
        parser.error("forced continuation is not defined for this method")
    if args.limit is not None and args.subset_size is not None:
        parser.error("--limit and --subset-size are mutually exclusive")
    if args.subset_size is not None and args.subset_size <= 0:
        parser.error("--subset-size must be positive")
    return args


def _storage_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _cache_len(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    return int(past_key_values.get_seq_length())


def _special_span(
    token_ids: list[int], start: list[int], end: list[int]
) -> tuple[int, int, bool]:
    def find(sequence: list[int], pattern: list[int], begin: int = 0) -> int:
        for index in range(begin, len(sequence) - len(pattern) + 1):
            if sequence[index : index + len(pattern)] == pattern:
                return index
        return -1

    start_at = find(token_ids, start)
    if start_at < 0:
        return 0, 0, False
    content_start = start_at + len(start)
    end_at = find(token_ids, end, content_start)
    if end_at < 0:
        return content_start, len(token_ids), True
    return content_start, end_at, False


class LearnedDonorIndex:
    """Lazy index over learned K*=10 shards for shuffled-transition controls."""

    def __init__(
        self, cache_root: Path, dataset: str, seed: int, target_ids: list[str]
    ):
        run_dir = cache_run_dir(cache_root, dataset, "learned", K_STAR, seed=0)
        if not (run_dir / "_COMPLETE").exists():
            raise FileNotFoundError(
                f"Complete learned K={K_STAR} cache required before shuffle: {run_dir}"
            )
        self.entries: dict[str, tuple[Path, str]] = {}
        for metadata_path in sorted(run_dir.glob("shard-*.metadata.jsonl")):
            shard_path = metadata_path.with_name(
                metadata_path.name.replace(".metadata.jsonl", ".safetensors")
            )
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.entries[str(row["example_id"])] = (
                    shard_path,
                    row["tensor_prefix"],
                )
        rng = random.Random(seed)
        offset = rng.randrange(1, len(target_ids))
        self.donor = {
            example_id: target_ids[(index + offset) % len(target_ids)]
            for index, example_id in enumerate(target_ids)
        }
        self._loaded: dict[Path, dict[str, torch.Tensor]] = {}

    def transition(
        self, example_id: str, step: int, device: torch.device
    ) -> torch.Tensor:
        donor_id = self.donor[example_id]
        shard, prefix = self.entries[donor_id]
        if shard not in self._loaded:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                self._loaded[shard] = {
                    key: handle.get_tensor(key) for key in handle.keys()
                }
        delta = self._loaded[shard][f"{prefix}_delta"]
        if step >= delta.shape[0]:
            raise IndexError(f"donor {donor_id} has only {delta.shape[0]} transitions")
        return delta[step : step + 1].to(device=device)


class CaptureController:
    def __init__(
        self,
        model: CustomQwen3ForCausalLM,
        method: str,
        k: int,
        seed: int,
        example_id: str,
        donor_index: LearnedDonorIndex | None,
        noise_sigma: float | None,
        storage_dtype: torch.dtype,
    ):
        self.model = model
        self.method = method
        self.k = k
        self.seed = seed
        self.example_id = example_id
        self.donor_index = donor_index
        self.noise_sigma = noise_sigma
        self.storage_dtype = storage_dtype
        self.prompt_len = 0
        self.z: list[torch.Tensor] = []
        self.h: list[torch.Tensor] = []
        self.delta: list[torch.Tensor] = []
        self.learned_transition_norms: list[float] = []
        self.applied_transition_norms: list[float] = []
        self.position_ids: list[int] = []
        self.cache_lengths: list[int] = []
        self.reference_hidden: list[torch.Tensor] = []
        self.resume_hidden: list[torch.Tensor] = []
        self.resume_generated_index: int | None = None
        self._pending_latent = False
        self._head_input: torch.Tensor | None = None
        self._generator: torch.Generator | None = None
        self._random_head: torch.nn.Linear | None = None
        self._applied_norm_before_storage: float | None = None
        self._prepare_original = model.prepare_inputs_for_generation
        self._handles: list[Any] = []

    def install(self) -> None:
        seed_material = int(sha256_text(f"{self.seed}:{self.example_id}")[:15], 16)
        self._generator = torch.Generator(device=self.model.device)
        self._generator.manual_seed(seed_material)
        if self.method == "random_head_norm_matched":
            # A random-head ablation is one fixed random function for the
            # entire run, not a different function for every example.
            cache = getattr(self.model, "_analysis_random_heads", None)
            if cache is None:
                cache = {}
                self.model._analysis_random_heads = cache
            if self.seed not in cache:
                size = int(self.model.config.hidden_size)
                random_head = torch.nn.Linear(size, size, bias=True).to(
                    device=self.model.device, dtype=self.model.dtype
                )
                head_generator = torch.Generator(device=self.model.device)
                head_seed = int(sha256_text(f"random-head:{self.seed}")[:15], 16)
                head_generator.manual_seed(head_seed)
                with torch.no_grad():
                    random_head.weight.normal_(
                        mean=0.0,
                        std=float(self.model.config.initializer_range),
                        generator=head_generator,
                    )
                    random_head.bias.zero_()
                random_head.eval()
                cache[self.seed] = random_head
            self._random_head = cache[self.seed]
            self._random_head.eval()

        def head_hook(_module, inputs, _output):
            self._head_input = inputs[0][:, -1, :].detach()

        def model_hook(_module, _inputs, output):
            hidden = output.last_hidden_state[:, -1, :].detach()
            if self.method == "reference":
                self.reference_hidden.append(hidden.to("cpu", dtype=self.storage_dtype))
            elif self._pending_latent:
                self.h.append(hidden.to("cpu", dtype=self.storage_dtype))
                self._pending_latent = False
            elif (
                self.resume_generated_index is not None and len(self.resume_hidden) < 16
            ):
                self.resume_hidden.append(hidden.to("cpu", dtype=self.storage_dtype))

        self._handles.append(
            self.model.model.direction_delta_predictor.register_forward_hook(head_hook)
        )
        self._handles.append(self.model.model.register_forward_hook(model_hook))
        self.model.prepare_inputs_for_generation = types.MethodType(
            self._prepare_wrapper, self.model
        )

    def close(self) -> None:
        self.model.prepare_inputs_for_generation = self._prepare_original
        for handle in self._handles:
            handle.remove()

    def _randn_unit(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        value = torch.randn(
            shape, generator=self._generator, device=device, dtype=torch.float32
        )
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _control_delta(self, learned: torch.Tensor) -> torch.Tensor:
        step = len(self.z)
        learned_f = learned.float()
        norm = learned_f.norm(dim=-1, keepdim=True)
        if self.method in {"learned", "absolute_prediction"}:
            self._applied_norm_before_storage = float(norm.item())
            return learned
        if self.method in {"zero_transition", "repeated_think"}:
            self._applied_norm_before_storage = 0.0
            return torch.zeros_like(learned)
        if self.method == "shuffled_transition":
            assert self.donor_index is not None
            value = self.donor_index.transition(
                self.example_id, step, learned.device
            ).to(dtype=learned.dtype)
            self._applied_norm_before_storage = float(value.float().norm().item())
            return value
        if self.method == "matched_norm_random_direction":
            value = norm * self._randn_unit(learned.shape, learned.device)
            self._applied_norm_before_storage = float(value.norm().item())
            return value.to(dtype=learned.dtype)
        if self.method == "random_head_norm_matched":
            assert self._random_head is not None and self._head_input is not None
            raw = self._random_head(self._head_input).float()
            raw = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            value = norm * raw
            self._applied_norm_before_storage = float(value.norm().item())
            return value.to(dtype=learned.dtype)
        if self.method == "noisy_transition":
            unit_learned = learned_f / norm.clamp_min(1e-12)
            mixed = unit_learned + float(self.noise_sigma) * self._randn_unit(
                learned.shape, learned.device
            )
            mixed = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            value = norm * mixed
            self._applied_norm_before_storage = float(value.norm().item())
            return value.to(dtype=learned.dtype)
        raise KeyError(self.method)

    def _prepare_wrapper(
        controller_self,
        _bound_model,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        # This predicate is copied exactly from the GLR decoder's active
        # prepare_inputs_for_generation implementation.
        start, end = controller_self.model.model.get_start_end_think_idx(
            input_ids, train=False
        )
        is_latent = (
            start >= 1
            and end >= input_ids.size(-1)
            and controller_self.model.config.latent_reason_token_budget is not None
            and end - start + 1
            <= controller_self.model.config.latent_reason_token_budget
            and input_ids.size(-1) - start >= 0
        )
        learned = controller_self.model.model._last_direction_delta
        if is_latent:
            if learned is None:
                raise RuntimeError(
                    "latent step requested before a transition was computed"
                )
            applied = controller_self._control_delta(learned)
            controller_self.learned_transition_norms.append(
                float(learned.float().norm(dim=-1).item())
            )
            controller_self.applied_transition_norms.append(
                float(controller_self._applied_norm_before_storage)
            )
            controller_self.model.model._last_direction_delta = applied

        result = controller_self._prepare_original(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        if (
            not is_latent
            and controller_self.z
            and controller_self.resume_generated_index is None
        ):
            # K is a maximum budget. If the model emits </think> early (or
            # leaves the budget condition), the newest generated token is the
            # first discrete token after the realized latent trajectory.
            controller_self.resume_generated_index = max(
                0,
                int(input_ids.shape[-1]) - controller_self.prompt_len - 1,
            )
        if is_latent:
            applied = controller_self.model.model._last_direction_delta
            if controller_self.method == "repeated_think":
                think_id = int(controller_self.model.config.think_tokens_start[0])
                think = torch.tensor(
                    [[think_id]], device=input_ids.device, dtype=torch.long
                )
                result["inputs_embeds"] = controller_self.model.model.embed_tokens(
                    think
                )
                controller_self.model.last_input = result["inputs_embeds"].detach()
            z = result["inputs_embeds"][:, -1, :].detach()
            controller_self.z.append(z.to("cpu", dtype=controller_self.storage_dtype))
            controller_self.delta.append(
                applied.detach().to("cpu", dtype=controller_self.storage_dtype)
            )
            position = result.get("position_ids")
            if position is None:
                position = result.get("cache_position")
            controller_self.position_ids.append(
                int(position.reshape(-1)[-1])
                if position is not None
                else _cache_len(past_key_values)
            )
            controller_self.cache_lengths.append(_cache_len(past_key_values))
            controller_self._pending_latent = True
            if len(controller_self.z) == controller_self.k:
                # Candidate emitted by z_K is the first token whose embedding
                # will be consumed normally on the next decode step.
                controller_self.resume_generated_index = (
                    int(input_ids.shape[-1]) - controller_self.prompt_len
                )
        return result


class BanCloseAfterLatent(LogitsProcessor):
    def __init__(self, controller: CaptureController, close_ids: list[int], count: int):
        self.controller = controller
        self.close_ids = close_ids
        self.count = count

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        start = self.controller.resume_generated_index
        if start is None:
            return scores
        candidate_index = int(input_ids.shape[-1]) - self.controller.prompt_len
        if start <= candidate_index < start + self.count:
            scores[:, self.close_ids] = -torch.inf
        return scores


def stack(values: list[torch.Tensor], width: int) -> torch.Tensor:
    if not values:
        return torch.empty((0, width), dtype=torch.bfloat16)
    return torch.cat(values, dim=0).contiguous()


def completed_ids(run_dir: Path) -> set[str]:
    found: set[str] = set()
    for path in run_dir.glob("shard-*.metadata.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            found.add(str(json.loads(line)["example_id"]))
    return found


def verify_partial_cache(run_dir: Path) -> None:
    seen: set[str] = set()
    for metadata_path in sorted(run_dir.glob("shard-*.metadata.jsonl")):
        tensor_path = metadata_path.with_name(
            metadata_path.name.replace(".metadata.jsonl", ".safetensors")
        )
        if not tensor_path.exists():
            raise FileNotFoundError(tensor_path)
        rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
        ]
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        for row in rows:
            example_id = str(row["example_id"])
            if example_id in seen:
                raise ValueError(f"{run_dir}: duplicate cached ID {example_id}")
            seen.add(example_id)
            prefix = row["tensor_prefix"]
            required = {f"{prefix}_prompt_ids", f"{prefix}_generated_ids"}
            if not required.issubset(keys):
                raise KeyError(f"{tensor_path}: missing keys {required - keys}")


def flush_shard(
    run_dir: Path,
    shard_index: int,
    records: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
) -> None:
    tensor_path = run_dir / f"shard-{shard_index:05d}.safetensors"
    metadata_path = run_dir / f"shard-{shard_index:05d}.metadata.jsonl"
    tensor_tmp = tensor_path.with_suffix(f".safetensors.tmp.{os.getpid()}")
    metadata_tmp = metadata_path.with_suffix(f".jsonl.tmp.{os.getpid()}")
    save_file(tensors, tensor_tmp)
    metadata_tmp.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    tensor_tmp.replace(tensor_path)
    metadata_tmp.replace(metadata_path)


def load_model(args: argparse.Namespace):
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = (
            ABS_CHECKPOINT_DEFAULT
            if args.method == "absolute_prediction"
            else GLR_CHECKPOINT
        )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    config = CustomQwen3Config.from_pretrained(
        checkpoint, cache_dir=args.cache_dir, latent_reason_token_budget=args.k
    )
    if args.method == "absolute_prediction" and bool(
        config.latent_reason_pred_delta_embed
    ):
        raise ValueError("absolute checkpoint has latent_reason_pred_delta_embed=True")
    if args.method != "absolute_prediction" and not bool(
        config.latent_reason_pred_delta_embed
    ):
        raise ValueError("GLR checkpoint is not displacement-prediction")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=args.cache_dir)
    model = CustomQwen3ForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        cache_dir=args.cache_dir,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval()
    model.debug_directional = False
    return checkpoint, tokenizer, model


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = load_dataset(args.dataset, args.limit)
    if args.subset_size is not None:
        if args.subset_size > len(rows):
            raise ValueError(
                f"subset size {args.subset_size} exceeds {len(rows)} examples"
            )
        subset_rng = random.Random(args.subset_seed)
        selected = sorted(subset_rng.sample(range(len(rows)), args.subset_size))
        rows = [rows[index] for index in selected]
    run_dir = cache_run_dir(
        args.cache_root,
        args.dataset,
        args.method,
        args.k,
        args.seed,
        args.noise_sigma,
        args.forced_continuation,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_existing:
        verify_partial_cache(run_dir)
    done = completed_ids(run_dir) if args.resume_existing else set()
    if done and not args.resume_existing:
        raise FileExistsError(
            f"{run_dir} contains partial output; pass --resume-existing"
        )
    remaining = [row for i, row in enumerate(rows) if str(row.get("id", i)) not in done]
    if not remaining and len(done) == len(rows) and (run_dir / "_COMPLETE").exists():
        print(f"Already complete: {run_dir} ({len(done)} examples)")
        return
    checkpoint, tokenizer, model = load_model(args)
    width = int(model.config.hidden_size)
    storage_dtype = _storage_dtype(args.tensor_storage_dtype)
    donor_index = None
    if args.method == "shuffled_transition":
        donor_index = LearnedDonorIndex(
            args.cache_root,
            args.dataset,
            args.seed,
            [str(row.get("id", i)) for i, row in enumerate(rows)],
        )
    existing_shards = sorted(run_dir.glob("shard-*.safetensors"))
    shard_index = len(existing_shards)
    shard_records: list[dict[str, Any]] = []
    shard_tensors: dict[str, torch.Tensor] = {}

    for row_index, row in enumerate(remaining):
        example_id = str(row.get("id", row_index))
        inputs = tokenizer(str(row["q"]), return_tensors="pt").to(model.device)
        controller = CaptureController(
            model,
            args.method,
            args.k,
            args.seed,
            example_id,
            donor_index,
            args.noise_sigma,
            storage_dtype,
        )
        controller.prompt_len = int(inputs["input_ids"].shape[-1])
        controller.install()
        processors = LogitsProcessorList()
        if args.forced_continuation:
            processors.append(
                BanCloseAfterLatent(
                    controller,
                    [int(value) for value in model.config.think_tokens_end],
                    args.ban_close_token_for,
                )
            )
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=DATASETS[args.dataset].cap,
                    do_sample=False,
                    num_beams=1,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    logits_to_keep=1,
                    use_cache=True,
                    logits_processor=processors,
                )
        finally:
            controller.close()

        prompt_ids = [int(x) for x in inputs["input_ids"][0].tolist()]
        generated_ids = [int(x) for x in output[0, len(prompt_ids) :].tolist()]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
        full_text = tokenizer.decode(output[0], skip_special_tokens=False)
        reasoning_start, reasoning_end, thought_truncated = _special_span(
            generated_ids,
            [int(x) for x in model.config.think_tokens_start],
            [int(x) for x in model.config.think_tokens_end],
        )
        parsed, valid, correct = score_generation(
            args.dataset, completion, str(row["a"])
        )
        eos = model.generation_config.eos_token_id
        eos_ids = {int(eos)} if isinstance(eos, int) else {int(x) for x in (eos or [])}
        stopped_on_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
        truncated = (
            len(generated_ids) == DATASETS[args.dataset].cap and not stopped_on_eos
        )
        prefix = f"e{shard_index:05d}_{len(shard_records):04d}"
        metadata: dict[str, Any] = {
            "example_id": example_id,
            "dataset": args.dataset,
            "method": args.method,
            "K": args.k,
            "seed": args.seed,
            "noise_sigma": args.noise_sigma,
            "forced_continuation": args.forced_continuation,
            "tensor_prefix": prefix,
            "prompt_hash": sha256_text(str(row["q"])),
            "gold_answer": str(row["a"]),
            "generated_text": completion,
            "full_generated_text": full_text,
            "parsed_answer": parsed,
            "valid_answer": valid,
            "correct": correct,
            "control_correct": correct,
            "control_correctness_key": "repository_exact_equivalence_recomputed",
            "truncated": truncated,
            "thought_truncated": thought_truncated,
            "stopped_on_eos": stopped_on_eos,
            "reasoning_start": reasoning_start,
            "reasoning_end": reasoning_end,
            "resume_generated_index": controller.resume_generated_index,
            "position_ids": controller.position_ids,
            "kv_cache_lengths": controller.cache_lengths,
            "kv_cache_length_before_resume": (
                controller.cache_lengths[-1] + 1 if controller.cache_lengths else None
            ),
            "learned_transition_norms_float32": controller.learned_transition_norms,
            "applied_transition_norms_float32": controller.applied_transition_norms,
            "checkpoint": str(checkpoint),
            "prompt_token_count": len(prompt_ids),
            "generated_token_count": len(generated_ids),
            "actual_latent_steps": len(controller.z),
        }
        shard_tensors[f"{prefix}_prompt_ids"] = torch.tensor(
            prompt_ids, dtype=torch.int64
        )
        shard_tensors[f"{prefix}_generated_ids"] = torch.tensor(
            generated_ids, dtype=torch.int64
        )

        if args.method == "reference":
            # Forward call 0 is prompt prefill. Calls 1..N-1 contextualize
            # generated tokens 0..N-2. Keep only the explicit thought span.
            contextual = controller.reference_hidden[1:]
            available_end = min(reasoning_end, len(contextual))
            thought_ids = generated_ids[reasoning_start:available_end]
            thought_h = contextual[reasoning_start:available_end]
            shard_tensors[f"{prefix}_reasoning_ids"] = torch.tensor(
                thought_ids, dtype=torch.int64
            )
            shard_tensors[f"{prefix}_h_discrete"] = stack(thought_h, width)
        else:
            if (
                not controller.z
                or len(controller.z) > args.k
                or len(controller.h) != len(controller.z)
            ):
                raise RuntimeError(
                    f"{args.dataset}/{example_id}: captured z={len(controller.z)}, "
                    f"h={len(controller.h)}, budget K={args.k}"
                )
            resume_at = controller.resume_generated_index
            resume_ids = (
                generated_ids[resume_at : resume_at + 16]
                if resume_at is not None
                else []
            )
            shard_tensors[f"{prefix}_z"] = stack(controller.z, width)
            shard_tensors[f"{prefix}_h_latent"] = stack(controller.h, width)
            shard_tensors[f"{prefix}_delta"] = stack(controller.delta, width)
            shard_tensors[f"{prefix}_resume_ids"] = torch.tensor(
                resume_ids, dtype=torch.int64
            )
            shard_tensors[f"{prefix}_resume_h"] = stack(
                controller.resume_hidden[:16], width
            )
        shard_records.append(metadata)

        if len(shard_records) == args.shard_size:
            flush_shard(run_dir, shard_index, shard_records, shard_tensors)
            shard_index += 1
            shard_records, shard_tensors = [], {}

    if shard_records:
        flush_shard(run_dir, shard_index, shard_records, shard_tensors)

    all_metadata: list[str] = []
    for path in sorted(run_dir.glob("shard-*.metadata.jsonl")):
        all_metadata.extend(path.read_text(encoding="utf-8").splitlines())
    (run_dir / "metadata.jsonl").write_text(
        "\n".join(all_metadata) + ("\n" if all_metadata else ""), encoding="utf-8"
    )
    expected = len(rows)
    actual = len(completed_ids(run_dir))
    if actual != expected:
        raise RuntimeError(f"{run_dir}: expected {expected} examples, found {actual}")
    manifest = {
        "dataset": args.dataset,
        "method": args.method,
        "K": args.k,
        "seed": args.seed,
        "noise_sigma": args.noise_sigma,
        "forced_continuation": args.forced_continuation,
        "num_examples": actual,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "test_set_sha256": sha256_file(DATASETS[args.dataset].path),
        "decoding": {
            "greedy": True,
            "max_new_tokens": DATASETS[args.dataset].cap,
            "top_p": 1.0,
            "top_k": 0,
        },
        "tensor_storage_dtype": args.tensor_storage_dtype,
        "selection": {
            "kind": (
                "deterministic_random_subset"
                if args.subset_size is not None
                else ("prefix_limit" if args.limit is not None else "all")
            ),
            "subset_size": args.subset_size,
            "subset_seed": args.subset_seed if args.subset_size is not None else None,
            "limit": args.limit,
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Complete: {run_dir} ({actual} examples)")


if __name__ == "__main__":
    main()
