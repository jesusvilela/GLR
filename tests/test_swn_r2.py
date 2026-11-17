"""R2 exact finite identities, reversible transport, and model bridge smoke."""
import unittest
from types import SimpleNamespace
import torch
from experiments.swn_r2.cayley import (
    S, T, G, C, A, I, cayley_delta, quadratic_energy, random_orientations,
)
from experiments.swn_r2.bridge import CayleyDeltaBridge
from experiments.swn_r1.adapter import RuntimeTelos


class FakeCore:
    def __init__(self):
        self._last_direction_delta=torch.tensor([[0.1,0.5,0.7,-0.2,0.2,-0.3,0.4,0.1]])
    def get_start_end_think_idx(self,ids,train=False):
        return 2,ids.size(-1)


class FakeModel:
    def __init__(self):
        self.model=FakeCore()
        self.config=SimpleNamespace(enable_latent_reason=True,
                                    latent_reason_pred_delta_embed=True,
                                    latent_reason_token_budget=10)
    def prepare_inputs_for_generation(self,ids,**kwargs):
        return self.model._last_direction_delta.clone()


class TestCayley(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.x=torch.randn(8,1024,dtype=torch.float64)/3

    def test_finite_identities(self):
        self.assertTrue(torch.equal(S@S,I))
        self.assertTrue(torch.equal(T@T@T,I))
        self.assertTrue(torch.equal(S@T@S,T@T))
        self.assertTrue(torch.allclose(C.T@G,G@C))
        self.assertTrue(torch.allclose(A.T@G+G@A,torch.zeros_like(G)))
        self.assertTrue(torch.equal(A@A@A,-12*A))
        self.assertGreater(A.abs().sum().item(),0)

    def test_zero_identity_and_nontriviality(self):
        self.assertTrue(torch.equal(cayley_delta(self.x,0),self.x))
        self.assertGreater((cayley_delta(self.x,.1)-self.x).abs().max().item(),1e-4)

    def test_isometry_inverse(self):
        y=cayley_delta(self.x,.2)
        self.assertTrue(torch.allclose(quadratic_energy(y),quadratic_energy(self.x),rtol=1e-11,atol=1e-11))
        restored=cayley_delta(y,.2,direction=-1)
        self.assertTrue(torch.allclose(restored,self.x,rtol=1e-11,atol=1e-11))

    def test_orientation_control_equal_per_block_distance(self):
        signs=random_orientations(self.x)
        y=cayley_delta(self.x,.2)
        control=cayley_delta(self.x,.2,sign_blocks=signs)
        self.assertTrue(torch.allclose(quadratic_energy(control-self.x),
                                       quadratic_energy(y-self.x),rtol=1e-11,atol=1e-11))
        self.assertGreater((control-y).abs().max().item(),1e-4)

    def test_covariant_moving_chart(self):
        b=torch.tensor([[1.,.2,0.,0.],[0.,1.,0.,0.],
                        [0.,0.,1.,.4],[0.,0.,0.,1.]],dtype=torch.float64)
        inv=torch.linalg.inv(b)
        y=cayley_delta(self.x,.2)
        transformed=(self.x.reshape(-1,4)@b.T).reshape_as(self.x)
        in_new_chart=cayley_delta(transformed,.2,generator=b@A@inv,metric=inv.T@G@inv)
        target=(y.reshape(-1,4)@b.T).reshape_as(y)
        self.assertTrue(torch.allclose(in_new_chart,target,rtol=1e-11,atol=1e-11))

    def test_float32_gradient(self):
        x=self.x.float().detach().requires_grad_(True)
        y=cayley_delta(x,.1)
        self.assertTrue(torch.allclose(quadratic_energy(y),quadratic_energy(x),rtol=1e-5,atol=1e-5))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_invalid_generator_and_shape(self):
        with self.assertRaises(ValueError):
            cayley_delta(self.x,.1,generator=C)
        with self.assertRaises(ValueError):
            cayley_delta(torch.ones(5),.1)

    def test_glr_bridge_restores_original_delta(self):
        model=FakeModel()
        original=model.model._last_direction_delta.clone()
        ids=torch.tensor([[5,6,7,8]])
        with CayleyDeltaBridge(model,mode="swn_cayley",strength=.1) as bridge:
            changed=model.prepare_inputs_for_generation(ids)
        self.assertEqual(bridge.audit.accepted,1)
        self.assertGreater((changed-original).abs().max().item(),1e-7)
        self.assertTrue(torch.equal(model.model._last_direction_delta,original))
        with CayleyDeltaBridge(model,mode="orientation_cayley",strength=.1,
                               telos=RuntimeTelos(max_relative_perturbation=1e-7)) as reject:
            unchanged=model.prepare_inputs_for_generation(ids)
        self.assertEqual(reject.audit.reverted,1)
        self.assertTrue(torch.equal(unchanged,original))


if __name__=="__main__":
    unittest.main()
