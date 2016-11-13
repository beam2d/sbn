from chainer import AbstractSerializer, Link
from chainer.training import Extension, Trainer


__all__ = ['KeepBestModel']


class KeepBestModel(Extension):

    """Trainer extension to keep the model with the best validation performance.

    Args:
        best_model: Model object to store the parameters of the best model.
        key: Key to observe the score of each iteration.

    """
    def __init__(self, best_model: Link, key: str='validation/mcb') -> None:
        self._key = key
        self._best_score = -1e300
        self._best_iter = 0
        self._best_model = best_model

    def __call__(self, trainer: Trainer) -> None:
        score = trainer.observation.get(self._key, None)
        if score is not None and score > self._best_score:
            self._best_score = score
            self._best_iter = trainer.updater.iteration
            self._best_model.copyparams(trainer.updater.get_optimizer('main').target)

    def serialize(self, serializer: AbstractSerializer) -> None:
        self._best_score = serializer('_best_score', self._best_score)
        self._best_iter = serializer('_best_iter', self._best_iter)
        self._best_model.serialize(serializer['_best_model'])
