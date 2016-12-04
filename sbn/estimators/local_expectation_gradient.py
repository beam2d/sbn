from chainer import functions as F, Variable

from sbn.gradient_estimator import GradientEstimator
from sbn.variational_model import VariationalModel


__all__ = ['LocalExpectationGradientEstimator']


class LocalExpectationGradientEstimator(GradientEstimator):

    """Implementation of the local expectation gradient.

    This class implements the local expectation gradient method. The original paper [1] only deals with the case of
    shallow models (i.e., all latent variables are mutually independent given input data). This implementation extends
    it to the case of deep ones. The density factor q(z_i | z_{\i}) is exactly computed.

    """
    def __init__(self, model: VariationalModel) -> None:
        self._model = model

    def estimate_gradient(self, x: Variable) -> None:
        model = self._model
        zs = model.infer(x)
        ps = model.compute_generative_factors(x, zs)
        p_terms = F.sum(F.vstack([p.log_prob for p in ps]))

        local_expectations = [model.compute_local_expectation(x, zs, ps, l) for l in range(len(zs))]
        legrad_signal = sum(F.sum(le) for le in local_expectations)

        model.cleargrads()
        (-p_terms - legrad_signal).backward()
