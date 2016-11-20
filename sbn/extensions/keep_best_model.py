from chainer import AbstractSerializer
from chainer.training import Extension, Trainer

from sbn.variational_model import VariationalModel


__all__ = ['KeepBestModel']


class KeepBestModel(Extension):

    """Trainer extension to keep the model with the best validation performance.

    Args:
        target_model: Training target.
        best_model: Model object to store the parameters of the best model.
        key: Key to observe the score of each iteration.

    """
    def __init__(
            self,
            target_model: VariationalModel,
            best_model: VariationalModel,
            key: str='validation/mcb'
    ) -> None:
        self._key = key
        self._best_score = -1e300
        self._best_iter = 0
        self._best_model = best_model
        self._target_model = target_model

    def __call__(self, trainer: Trainer) -> None:
        score = trainer.observation.get(self._key, None)
        if score is not None and score > self._best_score:
            self._best_score = score
            self._best_iter = trainer.updater.iteration
            self._best_model.copyparams(self._target_model)

    @property
    def best_iteration(self):
        """Iteration counts at the best validation score."""
        return self._best_iter

    def serialize(self, serializer: AbstractSerializer) -> None:
        self._best_score = serializer('_best_score', self._best_score)
        self._best_iter = serializer('_best_iter', self._best_iter)
        self._best_model.serialize(serializer['_best_model'])
