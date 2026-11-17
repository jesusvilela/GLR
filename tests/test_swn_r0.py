"""CPU-only pilot. No model weights or CUDA dependency."""
import unittest
import torch
from experiments.swn_r0.transport import S, T, I, spinor_data, spin_woven_delta


class TestSpinWovenPilot(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.randn(5, 8, dtype=torch.float64)

    def test_group_relations(self):
        self.assertTrue(torch.equal(S @ S, I))
        self.assertTrue(torch.equal(T @ T @ T, I))
        self.assertTrue(torch.equal(S @ T @ S, T @ T))

    def test_metric_and_sector_invariance(self):
        s, t, _, g = spinor_data()
        self.assertGreater(torch.linalg.eigvalsh(g).min().item(), 0)
        for a in [s, t]:
            self.assertTrue(torch.allclose(a.T @ g @ a, g, atol=1e-13))

    def test_exact_identity_and_nontriviality(self):
        self.assertTrue(torch.equal(spin_woven_delta(self.x, 0), self.x))
        self.assertGreater((spin_woven_delta(self.x, 0.1) - self.x).norm().item(), 1e-8)

    def test_change_of_frame_covariance(self):
        _, _, c, g = spinor_data()
        b = torch.tensor([[1., 0.1, 0., 0.], [0., 1., 0.3, 0.],
                          [0., 0., 1., 0.2], [0., 0., 0., 1.]], dtype=torch.float64)
        inv = torch.linalg.inv(b)
        new_c = b @ c @ inv
        new_g = inv.T @ g @ inv
        direct = spin_woven_delta(self.x, 0.07)
        y = (self.x.reshape(-1, 4) @ b.T).reshape_as(self.x)
        moved = spin_woven_delta(y, 0.07, commutator=new_c, metric=new_g)
        expected = (direct.reshape(-1, 4) @ b.T).reshape_as(direct)
        self.assertTrue(torch.allclose(moved, expected, atol=1e-12, rtol=1e-12))

    def test_gradients(self):
        x = self.x.detach().requires_grad_(True)
        out = spin_woven_delta(x, 0.1).square().mean()
        out.backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(x.grad.abs().max().item(), 0)

    def test_nonlinearity_and_shape_guards(self):
        x, y = self.x, self.x * 0.33
        a = spin_woven_delta(x+y, 0.1)
        b = spin_woven_delta(x, 0.1) + spin_woven_delta(y, 0.1)
        self.assertGreater((a - b).norm().item(), 1e-9)
        with self.assertRaises(ValueError):
            spin_woven_delta(torch.randn(2, 7), 0.1)


if __name__ == '__main__':
    unittest.main()
