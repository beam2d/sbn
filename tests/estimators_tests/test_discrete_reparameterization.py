from unittest import TestCase

from sbn.estimators import DiscreteReparameterizationEstimator
from .helper import GradientEstimationTester


class TestDiscreteReparameterizationEstimator(TestCase):

    def setUp(self):
        self.tester = GradientEstimationTester()
        self.x = self.tester.gen_x()

    def to_gpu(self):
        self.x.to_gpu()
        self.tester.to_gpu()

    def test_estimate_gradient(self):
        estimator = DiscreteReparameterizationEstimator(self.tester.model)
        self.to_gpu()
        estimator.to_gpu()
        self.tester.check_estimator(self.x, estimator, n_sample=100000, trial=100)

    def test_estimate_gradient_with_multiple_samples(self):
        estimator = DiscreteReparameterizationEstimator(self.tester.model, n_samples=10)
        self.to_gpu()
        estimator.to_gpu()
        self.tester.check_estimator(self.x, estimator, n_sample=10000, trial=100, rtol=2e-3)
