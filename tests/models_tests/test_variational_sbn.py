import math
from unittest import TestCase

from chainer import cuda
import chainer.links as L
import cupy as cp
import numpy as np

from sbn.models import VariationalSBN
from sbn import SigmoidBernoulliVariable


class TestVariationalSBN(TestCase):

    def setUp(self):
        self.infer_layers = [
            L.Linear(5, 4),
            L.Linear(4, 3),
            L.Linear(3, 2),
        ]
        self.gen_layers = [
            L.Linear(4, 5),
            L.Linear(3, 4),
            L.Linear(2, 3)
        ]
        self.mean = np.random.randn(5).astype('f')
        self.model = VariationalSBN(self.gen_layers, self.infer_layers, 2, self.mean)

        self.x = (np.random.rand(3, 5) > 0.5).astype('f')
        self.xp = np

    def test_n_stochastic_layers(self):
        self.assertEqual(self.model.n_stochastic_layers, len(self.infer_layers))

    def test_infer(self):
        xp = self.xp
        zs = self.model.infer(self.x)
        self.assertEqual(len(zs), len(self.infer_layers))
        for i in range(len(zs)):
            self.assertIsInstance(zs[i], SigmoidBernoulliVariable)
            x = self.x - self.mean if i == 0 else zs[i-1].sample.data
            xp.testing.assert_allclose(zs[i].logit.data, self.infer_layers[i](x).data)

    def test_compute_generative_factors(self):
        xp = self.xp
        zs = self.model.infer(self.x)
        ps = self.model.compute_generative_factors(self.x, zs)
        self.assertEqual(len(ps), len(self.gen_layers) + 1)

        prior = xp.broadcast_to(self.model.generative_net.prior.data, zs[-1].sample.shape)
        xp.testing.assert_allclose(ps[-1].logit.data, prior)

        for i in range(len(ps) - 1):
            self.assertIsInstance(ps[i], SigmoidBernoulliVariable)
            z = ps[i + 1].sample.data
            xp.testing.assert_allclose(ps[i].logit.data, self.gen_layers[i](z).data)

    def test_compute_variational_bound(self):
        xp = self.xp
        zs = self.model.infer(self.x)
        ps = self.model.compute_generative_factors(self.x, zs)
        vb = self.model.compute_variational_bound(zs, ps)

        self.assertEqual(vb.shape, (len(self.x),))
        self.assertTrue(bool((vb < 0).all()))  # LL < 0

        ll = sum(p.log_prob.data for p in ps) - sum(z.log_prob.data for z in zs)
        xp.testing.assert_allclose(vb, ll)

    def test_compute_monte_carlo_bound(self):
        xp = self.xp
        B = len(self.x)
        K = 5
        x = self.x.reshape(1, B, -1).repeat(K, axis=0).reshape(B * K, -1)
        zs = self.model.infer(x)
        ps = self.model.compute_generative_factors(x, zs)
        vb = self.model.compute_variational_bound(zs, ps)
        mcb = self.model.compute_monte_carlo_bound(vb, K)

        self.assertEqual(mcb.shape, (B,))
        self.assertTrue(bool((mcb < 0).all()))  # LL < 0

        ll = sum(p.log_prob.data for p in ps) - sum(z.log_prob.data for z in zs)
        ll = ll.reshape(B, K)
        ll_max = ll.max(axis=1)
        ll_mcb = xp.log(xp.exp(ll - ll_max[:, None]).sum(axis=1)) + ll_max - math.log(K)
        xp.testing.assert_allclose(mcb, ll_mcb)

    def test_compute_local_signals(self):
        xp = self.xp
        zs = self.model.infer(self.x)
        ps = self.model.compute_generative_factors(self.x, zs)
        signals = self.model.compute_local_signals(zs, ps)
        self.assertEqual(len(signals), len(self.infer_layers))

        for i in range(len(self.infer_layers)):
            z_terms = sum(z.log_prob.data for z in zs[i:])
            p_terms = sum(p.log_prob.data for p in ps[i:])
            signal_ll = p_terms - z_terms
            xp.testing.assert_allclose(signals[i], signal_ll, rtol=1e-6)


class TestVariationalSBNOnGPU(TestVariationalSBN):

    def setUp(self):
        super().setUp()
        self.model.to_gpu()
        self.mean = self.model.mean
        self.x = cuda.to_gpu(self.x)
        self.xp = cp
