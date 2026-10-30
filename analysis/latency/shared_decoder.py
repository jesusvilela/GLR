# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""One explicit autoregressive decoder shared by CoT-SFT and GLR-10.

This module intentionally bypasses ``GenerationMixin`` for both methods.  Both
checkpoints call the same backbone and vocabulary head code.  GLR differs only
by replacing ten would-be text-token positions after ``<think>`` with the
checkpoint's learned continuous transition.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import torch


EXPECTED_ARCHITECTURE = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}


@dataclass
class CallInstrumentation:
    backbone_trace: list[dict[str, Any]] = field(default_factory=list)
    head_trace: list[dict[str, Any]] = field(default_factory=list)
    backbone_hook_calls: int = 0
    vocab_hook_calls: int = 0
    transition_hook_calls: int = 0
    vocab_hook_positions: int = 0
    transition_hook_positions: int = 0
    _phase: str = "idle"

    def reset(self) -> None:
        self.backbone_trace.clear()
        self.head_trace.clear()
        self.backbone_hook_calls = 0
        self.vocab_hook_calls = 0
        self.transition_hook_calls = 0
        self.vocab_hook_positions = 0
        self.transition_hook_positions = 0
        self._phase = "idle"


class HookBundle:
    """Independent forward hooks used to validate the explicit trace."""

    def __init__(self, model: Any, instrumentation: CallInstrumentation):
        self.instrumentation = instrumentation
        self.handles = [
            model.model.register_forward_hook(self._backbone_hook),
            model.lm_head.register_forward_pre_hook(self._vocab_hook),
        ]
        transition = getattr(model.model, "direction_delta_predictor", None)
        if transition is not None:
            self.handles.append(
                transition.register_forward_pre_hook(self._transition_hook)
            )

    def _backbone_hook(self, _module: Any, _inputs: Any, _output: Any) -> None:
        self.instrumentation.backbone_hook_calls += 1

    def _vocab_hook(self, _module: Any, inputs: Any) -> None:
        hidden = inputs[0]
        self.instrumentation.vocab_hook_calls += 1
        self.instrumentation.vocab_hook_positions += int(hidden.shape[-2])

    def _transition_hook(self, _module: Any, inputs: Any) -> None:
        hidden = inputs[0]
        self.instrumentation.transition_hook_calls += 1
        positions = int(hidden.shape[-2]) if hidden.ndim == 3 else 1
        self.instrumentation.transition_hook_positions += positions

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def architecture_from_config(config: Any) -> dict[str, int]:
    return {
        key: int(getattr(config, key))
        for key in EXPECTED_ARCHITECTURE
    }


def validate_architecture(config: Any) -> dict[str, int]:
    actual = architecture_from_config(config)
    if actual != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"Unexpected Qwen3-1.7B architecture: {actual}; "
            f"expected {EXPECTED_ARCHITECTURE}"
        )
    return actual


def approximate_flops(
    config: Any,
    backbone_trace: list[dict[str, Any]],
    vocabulary_positions: int,
    transition_positions: int,
) -> dict[str, int]:
    architecture = validate_architecture(config)
    d = architecture["hidden_size"]
    f = architecture["intermediate_size"]
    layers = architecture["num_hidden_layers"]
    d_kv = architecture["num_key_value_heads"] * architecture["head_dim"]
    vocabulary = architecture["vocab_size"]
    backbone = 0
    for call in backbone_trace:
        q, s = int(call["query_length"]), int(call["context_length"])
        layer_flops = q * (4 * d * d + 4 * d * d_kv + 6 * d * f) + 4 * d * q * s
        backbone += layers * layer_flops
    vocab = int(vocabulary_positions) * 2 * d * vocabulary
    transition = int(transition_positions) * 2 * d * d
    return {
        "backbone": int(backbone),
        "vocabulary_head": int(vocab),
        "transition_head": int(transition),
        "total": int(backbone + vocab + transition),
    }


class SharedGreedyDecoder:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        method: str,
        k: int,
        max_total_steps: int,
    ):
        if method not in ("cot_sft", "glr_k10"):
            raise ValueError(method)
        if (method == "cot_sft" and k != 0) or (
            method == "glr_k10" and k != 10
        ):
            raise ValueError(f"Invalid method/K pair: {method}/{k}")
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.k = k
        self.max_total_steps = max_total_steps
        self.instrumentation = CallInstrumentation()
        self.hooks = HookBundle(model, self.instrumentation)
        self.architecture = validate_architecture(model.config)
        self.think_start_ids = [
            int(value)
            for value in getattr(model.config, "think_tokens_start", [151667])
        ]
        if len(self.think_start_ids) != 1:
            raise ValueError(
                "The matched decoder currently expects one Qwen <think> token"
            )
        generation_eos = model.generation_config.eos_token_id
        if generation_eos is None:
            generation_eos = tokenizer.eos_token_id
        self.eos_ids = (
            {int(generation_eos)}
            if isinstance(generation_eos, int)
            else {int(value) for value in generation_eos}
        )
        transition = getattr(model.model, "direction_delta_predictor", None)
        if method == "glr_k10" and transition is None:
            raise ValueError("GLR checkpoint has no direction_delta_predictor")
        if method == "cot_sft" and transition is not None:
            raise ValueError("CoT-SFT unexpectedly contains a transition head")
        # The explicit decoder invokes the learned transition only at the one
        # hidden position needed for each latent update.  Disabling the
        # backbone's convenience branch prevents projection of every prompt
        # position through that head.
        self.model.model.enable_latent_reason = False

    def close(self) -> None:
        self.hooks.remove()

    def _past_length(self, past_key_values: Any) -> int:
        return 0 if past_key_values is None else int(past_key_values.get_seq_length())

    def _backbone(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any = None,
        phase: str,
    ) -> tuple[torch.Tensor, Any]:
        query_length = int(
            input_ids.shape[-1]
            if input_ids is not None
            else inputs_embeds.shape[-2]
        )
        context_length = self._past_length(past_key_values) + query_length
        self.instrumentation._phase = phase
        self.instrumentation.backbone_trace.append(
            {
                "phase": phase,
                "query_length": query_length,
                "context_length": context_length,
            }
        )
        outputs = self.model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, -1:, :], outputs.past_key_values

    def _vocabulary(self, hidden: torch.Tensor, phase: str) -> int:
        self.instrumentation._phase = phase
        logits = self.model.lm_head(hidden)
        self.instrumentation.head_trace.append(
            {"head": "vocabulary", "phase": phase, "hidden_positions": 1}
        )
        return int(torch.argmax(logits[:, -1, :], dim=-1).item())

    def _transition(
        self, hidden: torch.Tensor, base_embedding: torch.Tensor, phase: str
    ) -> torch.Tensor:
        self.instrumentation._phase = phase
        delta = self.model.model.direction_delta_predictor(hidden[:, -1, :])
        self.instrumentation.head_trace.append(
            {"head": "transition", "phase": phase, "hidden_positions": 1}
        )
        return base_embedding + delta.unsqueeze(1)

    def _token_embedding(self, token_id: int) -> torch.Tensor:
        token = torch.tensor(
            [[token_id]], dtype=torch.long, device=self.model.device
        )
        return self.model.model.embed_tokens(token)

    def _assert_instrumentation(self, latent_steps: int) -> None:
        trace_backbones = len(self.instrumentation.backbone_trace)
        trace_vocab = sum(
            item["head"] == "vocabulary"
            for item in self.instrumentation.head_trace
        )
        trace_transition = sum(
            item["head"] == "transition"
            for item in self.instrumentation.head_trace
        )
        if self.instrumentation.backbone_hook_calls != trace_backbones:
            raise AssertionError("Backbone hook/trace count mismatch")
        if self.instrumentation.vocab_hook_calls != trace_vocab:
            raise AssertionError("Vocabulary hook/trace count mismatch")
        if self.instrumentation.transition_hook_calls != trace_transition:
            raise AssertionError("Transition hook/trace count mismatch")
        latent_vocab = [
            item
            for item in self.instrumentation.head_trace
            if item["head"] == "vocabulary"
            and item["phase"].startswith("latent_update")
        ]
        if latent_vocab:
            raise AssertionError("Vocabulary head called during GLR latent update")
        if self.method == "glr_k10":
            if latent_steps != self.k or trace_transition != self.k:
                raise AssertionError(
                    f"GLR expected K={self.k}, got latent_steps={latent_steps}, "
                    f"transition_calls={trace_transition}"
                )
        elif latent_steps != 0 or trace_transition != 0:
            raise AssertionError("CoT-SFT executed a latent transition")

    @torch.inference_mode()
    def generate(
        self, prompt_ids: torch.Tensor, *, synchronize: bool = True
    ) -> dict[str, Any]:
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("Primary latency path requires batch size one")
        self.instrumentation.reset()
        generated: list[int] = []
        latent_steps = 0
        stop_reason = "max_total_steps"
        first_text_event = torch.cuda.Event(enable_timing=True)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(self.model.device)
        if synchronize:
            torch.cuda.synchronize(self.model.device)
        started_ns = time.perf_counter_ns()
        start_event.record()

        hidden, past = self._backbone(
            input_ids=prompt_ids, phase="prefill"
        )
        next_token = self._vocabulary(hidden, "text_decode")
        generated.append(next_token)
        first_text_event.record()

        entered_latent = False
        while len(generated) + latent_steps < self.max_total_steps:
            if generated[-1] in self.eos_ids:
                stop_reason = "eos_or_end_of_turn"
                break

            if (
                self.method == "glr_k10"
                and not entered_latent
                and generated[-1] == self.think_start_ids[0]
            ):
                if len(generated) + self.k > self.max_total_steps:
                    stop_reason = "max_total_steps"
                    break
                # Process the emitted <think> token once.  Its hidden state
                # produces continuous latent input 1, exactly matching the
                # trained delta-embedding rule.
                think_embedding = self._token_embedding(generated[-1])
                hidden, past = self._backbone(
                    inputs_embeds=think_embedding,
                    past_key_values=past,
                    phase="text_context_before_latent",
                )
                latent_input = self._transition(
                    hidden, think_embedding, "latent_transition_1"
                )
                entered_latent = True
                for latent_index in range(1, self.k + 1):
                    hidden, past = self._backbone(
                        inputs_embeds=latent_input,
                        past_key_values=past,
                        phase=f"latent_update_{latent_index}",
                    )
                    latent_steps += 1
                    if latent_index < self.k:
                        latent_input = self._transition(
                            hidden,
                            latent_input,
                            f"latent_transition_{latent_index + 1}",
                        )
                    elif len(generated) + latent_steps < self.max_total_steps:
                        # This projection consumes the final latent hidden state
                        # only after all K updates have completed.  No placeholder
                        # vocabulary logits are materialized during updates 1..9.
                        generated.append(
                            self._vocabulary(hidden, "post_latent_text_decode")
                        )
                continue

            token_embedding = self._token_embedding(generated[-1])
            hidden, past = self._backbone(
                inputs_embeds=token_embedding,
                past_key_values=past,
                phase="text_decode",
            )
            generated.append(self._vocabulary(hidden, "text_decode"))

        if generated and generated[-1] in self.eos_ids:
            stop_reason = "eos_or_end_of_turn"
        elif len(generated) + latent_steps >= self.max_total_steps:
            stop_reason = "max_total_steps"

        end_event.record()
        if synchronize:
            torch.cuda.synchronize(self.model.device)
        ended_ns = time.perf_counter_ns()
        self._assert_instrumentation(latent_steps)
        flop_counts = approximate_flops(
            self.model.config,
            self.instrumentation.backbone_trace,
            self.instrumentation.vocab_hook_positions,
            self.instrumentation.transition_hook_positions,
        )
        return {
            "generated_token_ids": generated,
            "latent_steps": latent_steps,
            "text_token_count": len(generated),
            "total_post_prompt_steps": len(generated) + latent_steps,
            "stop_reason": stop_reason,
            "max_step_truncation": stop_reason == "max_total_steps",
            "generation_latency_ms": (ended_ns - started_ns) / 1_000_000.0,
            "cuda_event_latency_ms": float(start_event.elapsed_time(end_event)),
            "time_to_first_text_token_ms": float(
                start_event.elapsed_time(first_text_event)
            ),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self.model.device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(self.model.device)
            ),
            "backbone_trace": list(self.instrumentation.backbone_trace),
            "head_trace": list(self.instrumentation.head_trace),
            "backbone_call_count": self.instrumentation.backbone_hook_calls,
            "vocabulary_head_call_count": self.instrumentation.vocab_hook_calls,
            "transition_head_call_count": self.instrumentation.transition_hook_calls,
            "vocabulary_head_positions": self.instrumentation.vocab_hook_positions,
            "transition_head_positions": self.instrumentation.transition_hook_positions,
            "approximate_flops": flop_counts,
        }
