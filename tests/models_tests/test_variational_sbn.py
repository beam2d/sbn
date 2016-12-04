import copy
from unittest import TestCase

from chainer import cuda, functions as F, links as L, Variable
import cupy as cp
import numpy as np

from sbn.models import VariationalSBN
from sbn import SigmoidBernoulliVariable


_rtol = 1e-5


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

        self.x = Variable((np.random.rand(3, 5) > 0.5).astype('f'))
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

        self.assertEqual(vb.shape, (len(self.x.data),))

        ll = sum(p.log_prob.data for p in ps) - sum(z.log_prob.data for z in zs)
        xp.testing.assert_allclose(vb, ll, rtol=_rtol)

    def test_compute_monte_carlo_bound(self):
        xp = self.xp
        B = len(self.x.data)
        K = 5
        x = Variable(self.x.data.reshape(1, B, -1).repeat(K, axis=0).reshape(B * K, -1))
        zs = self.model.infer(x)
        ps = self.model.compute_generative_factors(x, zs)
        vb = self.model.compute_variational_bound(zs, ps)
        mcb = self.model.compute_monte_carlo_bound(vb, K)

        self.assertEqual(mcb.shape, (B,))
        self.assertTrue(bool((mcb < 0).all()))  # LL < 0

        ll = sum(p.log_prob.data for p in ps) - sum(z.log_prob.data for z in zs)
        ll = ll.reshape(B, K)
        ll_max = ll.max(axis=1)
        ll_mcb = xp.log(xp.exp(ll - ll_max[:, None]).mean(axis=1)) + ll_max
        xp.testing.assert_allclose(mcb, ll_mcb, rtol=_rtol)

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
            xp.testing.assert_allclose(signals[i], signal_ll, rtol=_rtol)

    def test_compute_local_marginal_signals(self):
        xp = self.xp
        zs = self.model.infer(self.x)
        ps = self.model.compute_generative_factors(self.x, zs)
        signals = self.model.compute_local_marginal_signals(zs, ps)
        self.assertEqual(len(signals), len(self.infer_layers))

        for i in range(len(self.infer_layers)):
            z_terms = sum(z.entropy.data for z in zs[i + 1:])
            p_terms = sum(p.log_prob.data for p in ps[i:])
            signal_ll = p_terms + z_terms
            xp.testing.assert_allclose(signals[i], signal_ll, rtol=_rtol)

    def test_compute_local_expectation(self):
        zs = self.model.infer(self.x)
        ps = self.model.compute_generative_factors(self.x, zs)
        vb = self.model.compute_variational_bound(zs, ps)
        log_q = sum(z.log_prob.data for z in zs)

        for l in range(len(zs)):
            le = self.model.compute_local_expectation(self.x, zs, ps, l).data
            expects = self.xp.empty_like(le)

            for j in range(zs[l].sample.shape[1]):
                zl = Variable(zs[l].sample.data.copy())
                zl.data[:, j] = 1 - zl.data[:, j]

                # Make flipped version of zs
                zs_flipped = list(zs)
                zs_flipped[l] = SigmoidBernoulliVariable(zs[l].logit, zl)
                if l + 1 < len(zs):
                    logit = self.model.inference_net[l + 1](zl)
                    zs_flipped[l + 1] = SigmoidBernoulliVariable(logit, zs[l + 1].sample)

                # Make flipped version of ps
                ps_flipped = list(ps)
                logit = self.model.generative_net[l](zl)
                ps_flipped[l] = SigmoidBernoulliVariable(logit, ps[l].sample)
                ps_flipped[l + 1] = SigmoidBernoulliVariable(ps[l + 1].logit, zl)

                # Evaluate f(z') = log p(x, z') - log q(z' | x), where z' is the flipped version of z
                vb_flipped = self.model.compute_variational_bound(zs_flipped, ps_flipped)

                # Compute q(z_i | mb_i) and q(z'_i | mb_i)
                log_q_flipped = sum(z.log_prob.data for z in zs_flipped)
                coeff = F.sigmoid(log_q - log_q_flipped).data
                coeff_flipped = 1 - coeff

                # Compute the local expectation * log q(z_i | pa_i)
                orig_term = vb * coeff * zs[l].elementwise_log_prob[:, j]
                flipped_term = vb_flipped * coeff_flipped * zs_flipped[l].elementwise_log_prob[:, j]
                expect = orig_term + flipped_term
                expects[:, j] = expect.data

            self.xp.testing.assert_allclose(le, expects, rtol=_rtol)

    def test_compute_reparameterized_local_expectation(self):
        zs = self.model.infer(self.x)
        ps = self.model.compute_generative_factors(self.x, zs)
        vb = self.model.compute_variational_bound(zs, ps)

        for l in range(len(zs)):
            le = self.model.compute_reparameterized_local_expectation(self.x, zs, ps, l).data
            expects = self.xp.empty_like(le)

            for j in range(zs[l].sample.shape[1]):
                zl = Variable(zs[l].sample.data.copy())
                zl.data[:, j] = 1 - zl.data[:, j]

                # Make flipped version of zs
                zs_flipped = list(zs[:l])
                zs_flipped.append(SigmoidBernoulliVariable(zs[l].logit, zl))
                for k in range(l + 1, len(zs)):
                    z = zs_flipped[-1].sample
                    enc = self.model.inference_net[k]
                    logit = enc(z)
                    zs_flipped.append(SigmoidBernoulliVariable(logit, noise=zs[k].noise))

                # Make flipped version of ps
                ps_flipped = list(ps[:l])
                for k in range(l, len(zs)):
                    dec = self.model.generative_net[k]
                    logit = dec(zs_flipped[k].sample)
                    sample = self.x if k == 0 else zs_flipped[k - 1].sample
                    ps_flipped.append(SigmoidBernoulliVariable(logit, sample))
                ps_flipped.append(SigmoidBernoulliVariable(ps[-1].logit, zs_flipped[-1].sample))

                # Evaluate f(z') = log p(x, z') - log q(z' | x), where z' is the flipped version of z
                vb_flipped = self.model.compute_variational_bound(zs_flipped, ps_flipped)

                q_z = zs[l].elementwise_prob[:, j].data
                expects[:, j] = q_z * vb + (1 - q_z) * vb_flipped

            self.xp.testing.assert_allclose(le, expects, rtol=_rtol)


class TestVariationalSBNOnGPU(TestVariationalSBN):

    def setUp(self):
        super().setUp()
        self.model.to_gpu()
        self.mean = self.model.mean
        self.x = Variable(cuda.to_gpu(self.x.data))
        self.xp = cp
