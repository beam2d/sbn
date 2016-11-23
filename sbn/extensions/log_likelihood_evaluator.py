import copy
from typing import Callable, Tuple

from chainer import no_backprop_mode, report
from chainer.dataset import concat_examples, Iterator
from chainer.training import Extension, make_extension, PRIORITY_WRITER, Trainer

from sbn.util import Array, KahanSum
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

        vb_sum = KahanSum()
        mcb_sum = KahanSum()

        K = self._n_samples
        with no_backprop_mode():
            for epoch in range(self._n_epochs):
                iterator = copy.copy(self._iterator)
                for batch in iterator:
                    B = len(batch)

                    x = concat_examples(batch, self._device)
                    x = x.repeat(K, axis=0)

                    zs = model.infer(x)
                    ps = model.compute_generative_factors(x, zs)
                    vb = model.compute_variational_bound(zs, ps)
                    mcb = model.compute_monte_carlo_bound(vb, K)

                    vb_sum.add(vb.sum(), B * K)
                    mcb_sum.add(mcb.sum(), B)

        return vb_sum.mean, mcb_sum.mean


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
