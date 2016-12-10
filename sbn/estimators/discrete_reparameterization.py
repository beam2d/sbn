from chainer import functions as F, Variable

from sbn.gradient_estimator import GradientEstimator
from sbn.variational_model import VariationalModel


__all__ = ['DiscreteReparameterizationEstimator']


class DiscreteReparameterizationEstimator(GradientEstimator):

    """Gradient estimator based on reparameterization for discrete variables.

    This class implements the reparameterization trick for discrete variables.

    Args:
        model: Model for which it estimates the gradient.
        n_samples: Number of samples used for Monte Carlo simulations of directly simulated terms.

    """
    def __init__(self, model: VariationalModel, n_samples: int=1) -> None:
        self._model = model
        self._n_samples = n_samples

    def estimate_gradient(self, x: Variable) -> None:
        model = self._model
        x0 = x
        K = self._n_samples
        if K > 1:
            x = Variable(x.data.repeat(K, axis=0), volatile=x.volatile)

        zs = model.infer(x)
        ps = model.compute_generative_factors(x, zs)
        direct_signal = F.sum(F.vstack([p.log_prob for p in ps])) + F.sum(F.vstack([z.entropy for z in zs]))
        if K > 1:
            direct_signal /= K

        # Compute the reparameterized local expectations (only for one of the sample for each data)
        zs0, ps0 = zs, ps
        if K > 1:
            zs0 = tuple(z[::K] for z in zs0)
            ps0 = tuple(p[::K] for p in ps0)
        local_expectations = [model.compute_reparameterized_local_expectation(x0, zs0, ps0, l)
                              for l in range(len(zs0))]
        reparam_signal = F.sum(F.vstack([F.sum(le) for le in local_expectations]))

        # Backprop errors from ps and each local expectation.
        model.cleargrads()
        (-direct_signal - reparam_signal).backward()
