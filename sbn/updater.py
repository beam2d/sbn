from typing import Optional

from chainer import AbstractSerializer, Optimizer, Variable
from chainer.dataset import Iterator
from chainer.training import StandardUpdater

from sbn.gradient_estimator import GradientEstimator


__all__ = ['Updater']


class Updater(StandardUpdater):

    """Updater designed for the variational learning of belief nets with discrete variables.

    """
    def __init__(
            self,
            estimator: GradientEstimator,
            iterator: Iterator,
            gen_optimizer: Optimizer,
            infer_optimizer: Optimizer,
            estimator_optimizer: Optional[Optimizer]=None,
            device=None
    ) -> None:
        optimizers = {'gen': gen_optimizer, 'infer': infer_optimizer}
        self.has_estimator_optimizer = estimator_optimizer is not None
        if self.has_estimator_optimizer:
            optimizers['estimator'] = estimator_optimizer
        super().__init__(iterator, optimizers, device=device)

        self._estimator = estimator

    def update_core(self) -> None:
        batch = self.get_iterator('main').next()
        x = self.converter(batch, self.device)

        self._estimator.estimate_gradient(Variable(x))

        self.get_optimizer('gen').update()
        self.get_optimizer('infer').update()
        if self.has_estimator_optimizer:
            self.get_optimizer('estimator').update()

    def serialize(self, serializer: AbstractSerializer) -> None:
        super().serialize(serializer)

        estimator = self._estimator
        s = getattr(estimator, 'serialize', None)
        if s is not None:
            s(serializer['_estimator'])
