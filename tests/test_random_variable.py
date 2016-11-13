from unittest import TestCase

import cupy as cp
import numpy as np

from sbn import RandomVariable, ConstantVariable, SigmoidBernoulliVariable


class TestRandomVariable(TestCase):

    def setUp(self):
        self.rv = RandomVariable()

    def test_sample(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.sample

    def test_log_prob(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.log_prob

    def test_mean(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.mean

    def test_entropy(self):
        with self.assertRaises(NotImplementedError):
            _ = self.rv.entropy


class TestConstantVariable(TestCase):

    def setUp(self):
        self.value = np.array([1., 2., 3.])
        self.cv = ConstantVariable(self.value)

    def test_sample(self):
        np.testing.assert_array_equal(self.cv.sample.data, self.value)

    def test_log_prob(self):
        np.testing.assert_array_equal(self.cv.log_prob.data, 0)

    def test_mean(self):
        np.testing.assert_array_equal(self.cv.mean.data, self.value)

    def test_entropy(self):
        np.testing.assert_array_equal(self.cv.entropy.data, 0)


class TestSigmoidBernoulliVariable(TestCase):

    xp = np

    def setUp(self):
        self.logit = self.xp.random.randn(5, 5).astype('f')
        self.rv = SigmoidBernoulliVariable(self.logit)

    def test_sample(self):
        z = self.rv.sample.data
        self.xp.testing.assert_array_equal(self.rv.sample.data, z)  # cached
        self.assertTrue(((z == 0) | (z == 1)).all())

    def test_log_prob(self):
        xp = self.xp
        z = self.rv.sample.data
        mu = self.rv.mean.data
        p = z * mu + (1 - z) * (1 - mu)
        xp.testing.assert_allclose(self.rv.log_prob.data, xp.log(p).sum(axis=1), rtol=1e-5)

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


class TestSigmoidBernoulliVariableGPU(TestSigmoidBernoulliVariable):

    xp = cp
