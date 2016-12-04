from unittest import TestCase

from sbn.estimators import DiscreteReparameterizationEstimator
from .helper import GradientEstimationTester


class TestLikelihoodRatioEstimator(TestCase):

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
