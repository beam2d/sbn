from unittest import TestCase

from chainer import Variable
import cupy as cp
import numpy as np

from sbn import RandomVariable, SigmoidBernoulliVariable


class TestRandomVariable(TestCase):

    def setUp(self):
        self.rv = RandomVariable()

    def test_sample(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.sample

    def test_prob(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.prob

    def test_elementwise_prob(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.elementwise_prob

    def test_log_prob(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.log_prob

    def test_elementwise_log_prob(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.elementwise_log_prob

    def test_mean(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.mean

    def test_entropy(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.entropy

    def test_getitem(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv[0]

    def test_make_flips(self):
        with self.assertRaises(NotImplementedError):
            self.rv.make_flips()


class TestSigmoidBernoulliVariable(TestCase):

    xp = np

    def setUp(self):
        self.B = 5
        self.H = 4
        self.logit = self.xp.random.randn(self.B, self.H).astype('f')
        self.rv = SigmoidBernoulliVariable(Variable(self.logit))

    def test_sample(self):
        z = self.rv.sample.data
        self.xp.testing.assert_array_equal(self.rv.sample.data, z)  # cached
        self.assertTrue(((z == 0) | (z == 1)).all())

    def test_prob(self):
        xp = self.xp
        z = self.rv.sample.data
        mu = self.rv.mean.data
        p = z * mu + (1 - z) * (1 - mu)
        xp.testing.assert_allclose(self.rv.prob.data, p.prod(axis=1), rtol=1e-5)

    def test_elementwise_prob(self):
        xp = self.xp
        z = self.rv.sample.data
        mu = self.rv.mean.data
        p = z * mu + (1 - z) * (1 - mu)
        xp.testing.assert_allclose(self.rv.elementwise_prob.data, p, rtol=1e-5)

    def test_log_prob(self):
        xp = self.xp
        z = self.rv.sample.data
        mu = self.rv.mean.data
        p = z * mu + (1 - z) * (1 - mu)
        xp.testing.assert_allclose(self.rv.log_prob.data, xp.log(p).sum(axis=1), rtol=1e-5)

    def test_elementwise_log_prob(self):
        xp = self.xp
        z = self.rv.sample.data
        mu = self.rv.mean.data
        p = z * mu + (1 - z) * (1 - mu)
        xp.testing.assert_allclose(self.rv.elementwise_log_prob.data, xp.log(p), rtol=1e-5)

    def test_mean(self):
        xp = self.xp
        logit = self.logit
        mu = 1 / (1 + xp.exp(-logit))  # sigmoid
        xp.testing.assert_allclose(self.rv.mean.data, mu, rtol=1e-5)

    def test_entropy(self):
        xp = self.xp
        logit = self.logit
        mu = 1 / (1 + xp.exp(-logit))
        entropy = (-mu * xp.log(mu) - (1 - mu) * xp.log(1 - mu)).sum(axis=1)
        xp.testing.assert_allclose(self.rv.entropy.data, entropy, rtol=1e-5)

    def test_getitem(self):
        xp = self.xp
        slc = self.rv[1:3, 2:]
        xp.testing.assert_array_equal(slc.logit.data, self.rv.logit[1:3, 2:].data)
        xp.testing.assert_array_equal(slc.sample.data, self.rv.sample[1:3, 2:].data)
        xp.testing.assert_array_equal(slc.noise, self.rv.noise[1:3, 2:])

    def test_make_flips(self):
        xp = self.xp
        rv = self.rv
        flips = rv.make_flips()
        self.assertTupleEqual(flips.sample.shape, (self.B, self.H, self.H))

        # flips[:, j, :] is a copy of rv[:, :] with rv[:, j] flipped.
        # flips.logit[i, j, k] should be equal to rv.logit[i, k].
        xp.testing.assert_array_equal(flips.logit.data == rv.logit.data[:, None, :], True)
        # flips.sample[i, j, k] should be equal to rv.sample[i, k] if j != k, and should not otherwise.
        xp.testing.assert_array_equal((flips.sample.data != rv.sample.data[:, None, :]) == xp.identity(self.H), True)


class TestSigmoidBernoulliVariableGPU(TestSigmoidBernoulliVariable):

    xp = cp
