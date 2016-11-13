from typing import Optional

from chainer import AbstractSerializer, Optimizer
from chainer.dataset import Iterator
from chainer.training import StandardUpdater

from sbn.grad_estimator import GradientEstimator


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

        gen_optimizer = self.get_optimizer('gen')
        infer_optimizer = self.get_optimizer('infer')
        estimator_optimizer = self.get_optimizer('estimator') if self.has_estimator_optimizer else None

        gen_optimizer.target.cleargrads()
        infer_optimizer.target.cleargrads()
        self._estimator.estimate_gradient(x)

        gen_optimizer.update()
        infer_optimizer.update()
        if estimator_optimizer is not None:
            estimator_optimizer.update()

    def serialize(self, serializer: AbstractSerializer) -> None:
        super().serialize(serializer)

        estimator = self._estimator
        s = getattr(estimator, 'serialize', None)
        if s is not None:
            s(serializer['_estimator'])
