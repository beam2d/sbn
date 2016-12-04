from chainer import functions as F, Variable

from sbn.gradient_estimator import GradientEstimator
from sbn.variational_model import VariationalModel


__all__ = ['DiscreteReparameterizationEstimator']


class DiscreteReparameterizationEstimator(GradientEstimator):

    """Gradient estimator based on reparameterization for discrete variables.

    This class implements the reparameterization trick for discrete variables.

    """
    def __init__(self, model: VariationalModel) -> None:
        self._model = model

    def estimate_gradient(self, x: Variable) -> None:
        model = self._model
        zs = model.infer(x)
        ps = model.compute_generative_factors(x, zs)
        direct_signal = F.sum(F.vstack([p.log_prob for p in ps]))

        # Compute the reparameterized local expectations
        local_expectations = [model.compute_reparameterized_local_expectation(x, zs, ps, l) for l in range(len(zs))]
        reparam_signal = F.sum(F.vstack([F.sum(le) for le in local_expectations]))

        # Backprop errors from ps and each local expectation.
        model.cleargrads()
        (-direct_signal - reparam_signal).backward()
