from chainer import cuda, Variable

from sbn.gradient_estimator import GradientEstimator
from sbn.util import backprop_from_all
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

        local_signals = model.compute_local_signals(zs, ps)

        # Compute the reparameterized local expectations
        local_expectations = [model.compute_local_expectation(x, zs, signal, l)
                              for l, signal in enumerate(local_signals)]

        # Backprop errors from ps and each local expectation.
        signal = backprop_from_all([p.log_prob for p in ps] + local_expectations, -1)
        model.cleargrads()
        signal.backward()
