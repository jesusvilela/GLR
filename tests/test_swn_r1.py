"""SWN/GLR R1: structural and optional real-small-Qwen3 integration tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from experiments.swn_r1.adapter import LatentDeltaBridge, RuntimeTelos
from experiments.swn_r1.teloi import EVALUATION_TELOI
from experiments.swn_r1.run import question_text, verify_trained_glr_checkpoint


class FakeCore:
    def __init__(self):
        self._last_direction_delta = torch.tensor([[0.1, 0.5, 0.7, -0.2,
                                                      0.2, -0.3, 0.4, 0.1]])
    def get_start_end_think_idx(self, ids, train=False):
        return 2, ids.size(-1)


class FakeModel:
    def __init__(self):
        self.model = FakeCore()
        self.config = SimpleNamespace(enable_latent_reason=True,
                                      latent_reason_pred_delta_embed=True,
                                      latent_reason_token_budget=10)
        self.received = []

    def prepare_inputs_for_generation(self, ids, **kw):
        delta = self.model._last_direction_delta.clone()
        self.received.append(delta)
        return delta


class TestBridge(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        self.old = self.model.model._last_direction_delta.clone()

    def test_baseline_bitwise_identity(self):
        with LatentDeltaBridge(self.model, mode="glr") as bridge:
            new = self.model.prepare_inputs_for_generation(self.ids)
        self.assertTrue(torch.equal(new, self.old))
        self.assertEqual(bridge.audit.proposed, 0)
        self.assertTrue(torch.equal(self.model.model._last_direction_delta, self.old))

    def test_spin_woven_changes_latent_delta_and_restores(self):
        original_prepare = self.model.prepare_inputs_for_generation
        with LatentDeltaBridge(self.model, mode="swn", strength=0.05) as bridge:
            changed = self.model.prepare_inputs_for_generation(self.ids)
        self.assertGreater((changed - self.old).abs().max().item(), 1e-7)
        self.assertTrue(torch.equal(self.model.model._last_direction_delta, self.old))
        self.assertEqual(bridge.audit.accepted, 1)
        self.assertEqual(bridge.audit.reverted, 0)
        self.assertEqual(self.model.prepare_inputs_for_generation, original_prepare)

    def test_strict_telos_reverts_and_reports_remainder(self):
        with LatentDeltaBridge(self.model, mode="swn", strength=0.05,
                               telos=RuntimeTelos(1e-7)) as bridge:
            got = self.model.prepare_inputs_for_generation(self.ids)
        self.assertTrue(torch.equal(got, self.old))
        self.assertEqual(bridge.audit.reverted, 1)
        self.assertEqual(bridge.audit.reasons["telos_norm_envelope"], 1)

    def test_random_control_has_matching_perturbation_norm(self):
        with LatentDeltaBridge(self.model, mode="swn", strength=0.05) as b:
            swn = self.model.prepare_inputs_for_generation(self.ids)
        with LatentDeltaBridge(self.model, mode="random", strength=0.05) as b:
            rnd = self.model.prepare_inputs_for_generation(self.ids)
        s = torch.linalg.vector_norm((swn - self.old).reshape(-1, 4), dim=-1)
        r = torch.linalg.vector_norm((rnd - self.old).reshape(-1, 4), dim=-1)
        self.assertTrue(torch.allclose(s, r, atol=1e-6, rtol=1e-5))
        self.assertFalse(torch.equal(swn, rnd))

    def test_inactive_thinking_does_not_transform(self):
        self.model.model.get_start_end_think_idx = lambda ids, train=False: (-1, -1)
        with LatentDeltaBridge(self.model, mode="swn", strength=0.1) as b:
            got = self.model.prepare_inputs_for_generation(self.ids)
        self.assertTrue(torch.equal(got, self.old))
        self.assertEqual(b.audit.proposed, 0)

    def test_restoration_on_exception(self):
        def bad(*args, **kw):
            raise RuntimeError("decoder stopped")
        self.model.prepare_inputs_for_generation = bad
        with self.assertRaisesRegex(RuntimeError, "decoder stopped"):
            with LatentDeltaBridge(self.model, mode="swn", strength=0.05):
                self.model.prepare_inputs_for_generation(self.ids)
        self.assertTrue(torch.equal(self.model.model._last_direction_delta, self.old))
        self.assertEqual(self.model.prepare_inputs_for_generation, bad)


class TestTeloi(unittest.TestCase):
    def test_pareto_and_no_scalar_collapsing(self):
        a = dict(accuracy=.8, mean_steps=140., mean_seconds=3., fallbacks_per_example=0.)
        b = dict(accuracy=.75, mean_steps=160., mean_seconds=4., fallbacks_per_example=0.)
        self.assertEqual(EVALUATION_TELOI.compare(a, b), "proposed_pareto_dominates")
        self.assertEqual(EVALUATION_TELOI.compare(b, a), "reference_pareto_dominates")
        self.assertEqual(EVALUATION_TELOI.compare(a, a), "equal")
        c = dict(b, accuracy=.9)
        self.assertEqual(EVALUATION_TELOI.compare(a, c), "incomparable")
        with self.assertRaises(ValueError):
            EVALUATION_TELOI.compare(a, dict(b, accuracy=float("nan")))


class TestCheckpointGate(unittest.TestCase):
    def test_rejects_base_model_and_missing_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(json.dumps({"enable_latent_reason": False}))
            with self.assertRaises(ValueError):
                verify_trained_glr_checkpoint(path)
            (path / "config.json").write_text(json.dumps({
                "enable_latent_reason": True, "latent_reason_pred_delta_embed": True}))
            with self.assertRaises(ValueError):
                verify_trained_glr_checkpoint(path)

    def test_parse_original_question_without_gold(self):
        rec = {"q": "<|im_start|>user\nQuestion: What is 2 + 2?\nAnswer:"
                    "<|im_end|>\n<|im_start|>assistant\n", "a": "#### 4"}
        self.assertIn("What is 2 + 2?", question_text(rec))


class TestRealTinyQwen3(unittest.TestCase):
    def test_actual_model_forward_and_delta_bridge(self):
        try:
            from glr_qwen3.configuration_qwen3_latent import CustomQwen3Config
            from glr_qwen3.modeling_qwen3_latent import CustomQwen3ForCausalLM
        except (ImportError, ModuleNotFoundError):
            self.skipTest("requires GLR's pinned Transformers dependency")
        torch.manual_seed(19)
        cfg = CustomQwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=16,
            enable_latent_reason=True, latent_reason_pred_delta_embed=True,
            think_tokens_start=[13], think_tokens_end=[14],
            latent_reason_token_budget=10, use_cache=False,
        )
        model = CustomQwen3ForCausalLM(cfg).eval()
        ids = torch.tensor([[5, 13, 7, 8]], dtype=torch.long)
        with torch.inference_mode():
            model(input_ids=ids)
            assert model.model._last_direction_delta is not None
            native = model.prepare_inputs_for_generation(ids)["inputs_embeds"]
            with LatentDeltaBridge(model, mode="swn", strength=0.05) as bridge:
                woven = model.prepare_inputs_for_generation(ids)["inputs_embeds"]
        self.assertEqual(native.shape, woven.shape)
        self.assertGreater((native - woven).abs().max().item(), 1e-8)
        self.assertEqual(bridge.audit.accepted, 1)


if __name__ == "__main__":
    unittest.main()
