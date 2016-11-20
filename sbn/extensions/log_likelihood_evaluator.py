import copy
from typing import Callable, Tuple

from chainer import no_backprop_mode, report
from chainer.dataset import concat_examples, Iterator
from chainer.training import Extension, make_extension, PRIORITY_WRITER, Trainer

from sbn.util import Array
from sbn.variational_model import VariationalModel


__all__ = ['LogLikelihoodEvaluator']


class LogLikelihoodEvaluator:

    """Evaluator of the bound of the log likelihood.

    This object computes two types of bounds of the log likelihood: variational bound and Monte Carlo bound. See
    ``sbn.VariationalModel`` for more details about these bounds.

    Args:
        iterator: Dataset iterator to iterate the training or validation dataset. It must stop after each epoch.
        target: Variational model to evaluate.
        device: Device to compute all evaluations.
        n_epochs: Number of epochs to evaluate.
        n_samples: Number of samples for computing the Monte Carlo bound.

    """
    trigger = 1, 'epoch'
    priority = PRIORITY_WRITER
    default_name = 'validation'

    def __init__(
            self,
            iterator: Iterator,
            target: VariationalModel,
            device,
            n_epochs: int=1,
            n_samples: int=100
    ) -> None:
        self._iterator = iterator
        self._target = target
        self._device = device
        self._n_epochs = n_epochs
        self._n_samples = n_samples

    def evaluate(self) -> Tuple[Array, Array]:
        model = self._target

        vb_sum = 0
        vb_count = 0
        mcb_sum = 0
        mcb_count = 0

        with no_backprop_mode():
            for epoch in range(self._n_epochs):
                iterator = copy.copy(self._iterator)
                for batch in iterator:
                    x = concat_examples(batch, self._device)
                    B = len(batch)
                    K = self._n_samples
                    D = x.shape[1]
                    x = x.reshape(1, B, D).repeat(K, axis=0).reshape(B * K, D)

                    zs = model.infer(x)
                    ps = model.compute_generative_factors(x, zs)
                    vb = model.compute_variational_bound(zs, ps)
                    mcb = model.compute_monte_carlo_bound(vb, K)

                    vb_sum += vb.sum()
                    vb_count += B * K
                    mcb_sum += mcb.sum()
                    mcb_count += B

        vb_mean = vb_sum / vb_count
        mcb_mean = mcb_sum / mcb_count
        # Negate to make them bounds of the log likelihood (instead of those of the NEGATIVE log likelihood)
        return -vb_mean, -mcb_mean


def evaluate_log_likelihood(
        iterator: Iterator,
        target: VariationalModel,
        device,
        n_epochs: int=1,
        n_samples: int=100
) -> Callable[[Trainer], None]:
    """Returns a trainer extension to evaluate the bound of the log likelihood.

    This function creates a ``LogLikelihoodEvaluator`` object and turns it into a trainer extension.

    """
    evaluator = LogLikelihoodEvaluator(iterator, target, device, n_epochs, n_samples)

    @make_extension(trigger=(1, 'epoch'), default_name='validation', priority=PRIORITY_WRITER)
    def log_likelihood_evaluator(_) -> None:
        vb, mcb = evaluator.evaluate()
        name = get_name()
        report({name + '/vb': vb, name + '/mcb': mcb})

    def get_name() -> str:
        return log_likelihood_evaluator.name

    return log_likelihood_evaluator
