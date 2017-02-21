from unittest import TestCase

from sbn.estimators import LikelihoodRatioEstimator
from .helper import GradientEstimationTester


class TestLikelihoodRatioEstimator(TestCase):

    def setUp(self):
        self.tester = GradientEstimationTester()
        self.x = self.tester.gen_x()

    def to_gpu(self):
        self.x.to_gpu()
        self.tester.to_gpu()

    def check_gradient_estimation(self, estimator, rtol=1e-3, atol=1e-3):
        self.to_gpu()
        estimator.to_gpu()
        self.tester.dry_run(self.x, estimator, n_sample=1000, trial=100)  # prepare the baseline estimation
        self.tester.check_estimator(self.x, estimator, n_sample=100000, trial=100, rtol=rtol, atol=atol)

    def test_constant_baseline(self):
        self.check_gradient_estimation(LikelihoodRatioEstimator(self.tester.model))

    def test_muprop(self):
        self.check_gradient_estimation(LikelihoodRatioEstimator(self.tester.model, use_muprop=True))
