# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Qwen3 with two-pass, teacher-forced hidden-state replacement (HSR).

The first pass is an ordinary discrete teacher-forced pass.  The second pass
replaces every token strictly between ``<think>`` and ``</think>`` with the
final-normalized first-pass hidden state at the preceding position.  The
replacement is not detached, matching the autograd path in the comparison GLR
run.  Only the second-pass answer loss is optimized.

This is the parallel HSR baseline described in
``coconut_style_parallel_hidden_state_replacement.md``.  It is not recursive
Coconut training and intentionally has no transition head or auxiliary loss.
"""

from typing import Optional, Union

import torch
from transformers import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class HSRForCausalLM(Qwen3ForCausalLM):
    """Qwen3 CausalLM whose labeled forward implements parallel HSR training."""

    def _marker_ids(self) -> tuple[int, int]:
        start_ids = getattr(self.config, "hsr_think_start_ids", None)
        end_ids = getattr(self.config, "hsr_think_end_ids", None)
        if not start_ids or not end_ids:
            raise ValueError(
                "The config must contain hsr_think_start_ids and "
                "hsr_think_end_ids. Use train_hsr.py to construct the model."
            )
        if len(start_ids) != 1 or len(end_ids) != 1:
            raise ValueError(
                "This Qwen3 baseline expects single-token <think> and "
                "</think> markers."
            )
        return int(start_ids[0]), int(end_ids[0])

    def _thought_spans(self, input_ids: torch.LongTensor) -> list[tuple[int, int]]:
        """Return half-open spans containing tokens strictly inside markers."""
        start_token_id, end_token_id = self._marker_ids()
        spans: list[tuple[int, int]] = []
        for row_index, row in enumerate(input_ids):
            starts = (row == start_token_id).nonzero(as_tuple=False).flatten()
            ends = (row == end_token_id).nonzero(as_tuple=False).flatten()
            if starts.numel() != 1 or ends.numel() != 1:
                raise ValueError(
                    f"HSR requires exactly one <think> and </think> per "
                    f"training row; row {row_index} has {starts.numel()} "
                    f"start marker(s) and {ends.numel()} end marker(s)."
                )
            start = int(starts.item()) + 1
            end = int(ends.item())
            if start >= end:
                raise ValueError(
                    f"Row {row_index} has no token strictly between its "
                    "<think> and </think> markers."
                )
            spans.append((start, end))
        return spans

    @staticmethod
    def _replace_thought_embeddings(
        inputs_embeds: torch.FloatTensor,
        first_pass_hidden: torch.FloatTensor,
        spans: list[tuple[int, int]],
    ) -> torch.FloatTensor:
        """Apply all first-pass replacements simultaneously."""
        replaced = inputs_embeds.clone()
        for row_index, (start, end) in enumerate(spans):
            replaced[row_index, start:end, :] = first_pass_hidden[
                row_index, start - 1 : end - 1, :
            ]
        return replaced

    @staticmethod
    def _answer_only_labels(
        labels: torch.LongTensor,
        spans: list[tuple[int, int]],
    ) -> torch.LongTensor:
        """Keep supervision only after ``</think>``.

        ``end`` is the position of ``</think>``.  Masking through that
        position removes prompt, reasoning, and marker supervision while
        preserving the answer and the normal assistant end-of-turn token.
        """
        answer_labels = labels.clone()
        for row_index, (_, end) in enumerate(spans):
            answer_labels[row_index, : end + 1] = -100
        if not torch.any(answer_labels != -100):
            raise ValueError("HSR batch has no supervised answer tokens.")
        return answer_labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Ordinary inference remains the stock Qwen3 path.  HSR inference is
        # driven explicitly by decode_hsr.py because latent cache positions
        # must not create placeholder token IDs.
        if labels is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if input_ids is None:
            raise ValueError("HSR training requires input_ids to locate markers.")
        if inputs_embeds is not None:
            raise ValueError(
                "HSR training constructs inputs_embeds internally; pass input_ids."
            )
        if past_key_values is not None:
            raise ValueError("HSR training does not accept past_key_values.")
        if isinstance(logits_to_keep, int) and logits_to_keep != 0:
            raise ValueError("HSR loss requires full-sequence logits_to_keep=0.")

        spans = self._thought_spans(input_ids)
        discrete_embeds = self.get_input_embeddings()(input_ids)

        # GLR's slice replacement preserves its first-pass autograd path.
        # HSR deliberately does the same for the controlled comparison.
        discrete_outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=discrete_embeds,
            use_cache=False,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        replaced_embeds = self._replace_thought_embeddings(
            discrete_embeds,
            discrete_outputs.last_hidden_state,
            spans,
        )
        replaced_outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=replaced_embeds,
            use_cache=False,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(replaced_outputs.last_hidden_state)
        answer_labels = self._answer_only_labels(labels, spans)
        loss = self.loss_function(
            logits=logits,
            labels=answer_labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=replaced_outputs.hidden_states,
            attentions=replaced_outputs.attentions,
        )
