import copy
from typing import Callable, Dict

from chainer import Link, report
from chainer.dataset import concat_examples, Iterator
from chainer.training import make_extension, PRIORITY_WRITER, Trainer

from sbn.grad_estimator import GradientEstimator
from sbn.util import Array


__all__ = ['GradientVarianceEvaluator', 'evaluate_gradient_variance']


class GradientVarianceEvaluator:

    """Evaluator of variance of a gradient estimator.

    This class evaluates the variance of a gradient estimator. The variance comes from two types of stochasticity:
    input data selection and latent variable simulation. This class tries to evaluate the variance coming from both of
    these two sources of deviation.

    Args:
        iterator: Dataset iterator to iterate the training or validation dataset. It must be shuffled and must not stop
            during the iterations.
        target: The target model.
        estimator: Gradient estimator.
        device: Device to compute all evaluations.
        n_iterations: Number of iterations to collect the statistics.

    """
    def __init__(
            self,
            iterator: Iterator,
            target: Link,
            estimator: GradientEstimator,
            device,
            n_iterations: int
    ) -> None:
        self._iterator = iterator
        self._target = target
        self._estimator = estimator
        self._device = device
        self._n_iterations = n_iterations

    def evaluate(self) -> Dict[str, Array]:
        """Evaluates the variance of the gradient estimator.

        Returns:
            Mapping of parameter names to the variances of the gradient estimations of the parameters.

        """
        model = self._target
        estimator = self._estimator

        # Online variance estimation.
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm.
        n = 0
        mean = {k: 0 for k, _ in model.namedparams()}
        var = {k: 0 for k, _ in model.namedparams()}

        iterator = copy.copy(self._iterator)
        for iteration in range(self._n_iterations):
            batch = next(iterator)
            x = concat_examples(batch, self._device)
            model.cleargrads()
            estimator.estimate_gradient(x)

            n += 1
            for name, param in model.namedparams():
                delta = param.grad - mean[name]
                mean[name] += delta / n
                var[name] += delta * (x - mean[name])

        for k in var.keys():
            var[k] /= n - 1

        return var


def evaluate_gradient_variance(
        iterator: Iterator,
        target: Link,
        estimator: GradientEstimator,
        device,
        n_iterations: int
) -> Callable[[Trainer], None]:
    """Returns a trainer extension to evaluate the gradient variance.

    This function creates a ``GradientVarianceEvaluator`` object and turns it into a trainer extension.

    """
    evaluator = GradientVarianceEvaluator(iterator, target, estimator, device, n_iterations)

    @make_extension(trigger=(1, 'epoch'), default_name='gradvar', priority=PRIORITY_WRITER)
    def gradient_variance_estimator(_) -> None:
        var = evaluator.evaluate()
        name = get_name()

        rep = {}
        s = 0
        c = 0
        count = 0
        # Compensated summation algorithm. See https://en.wikipedia.org/wiki/Kahan_summation_algorithm.
        for k, v in var.items():
            vsum = v.sum()
            y = vsum - c
            t = s + y
            c = (t - s) - y
            s = t

            count += v.size
            rep[name + k] = vsum / v.size

        rep[name + '/mean'] = s / count
        report(rep)

    def get_name() -> str:
        return gradient_variance_estimator.name

    return gradient_variance_estimator
