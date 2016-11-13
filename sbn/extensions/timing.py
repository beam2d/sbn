from typing import Optional

from chainer import AbstractSerializer, report
from chainer.training import Extension, Trainer


__all__ = ['Timing', 'TrainingTime', 'report_training_time']


class Timing(Extension):

    """Trainer extension to record a timing.

    This extension is used to record when an iteration finished. The recorded timing can be used to measure the
    pure elapsed time of training by ``TrainingTime`` extension.

    """
    priority = 0
    invoke_before_training = True

    def __init__(self):
        self.time = None  # type: Optional[float]

    def __call__(self, trainer: Trainer) -> None:
        self.time = trainer.elapsed_time


class TrainingTime(Extension):

    """Trainer extension to measure the elapsed time of training.

    The elapsed time of Trainer includes times taken for both training and evaluation. This extension measures the
    elapsed time only taken for training (i.e., parameter updates).

    The elapsed time at each iteration is reported by the extension name (which is ``training_time`` by default). The
    value is in seconds.

    Args:
        timing: Timing extension used to measure when the last iteration finished. This value is used to measure how
            long it took for the next training iteration.

    """
    priority = 10000
    default_name = 'training_time'

    def __init__(self, timing: Timing) -> None:
        self._timing = timing
        self._elapsed_time = 0

    def __call__(self, trainer: Trainer) -> None:
        if self._timing.time is None:
            raise RuntimeError('the last timing is not measured')
        self._elapsed_time += trainer.elapsed_time - self._timing.time
        report({self.name: self._elapsed_time})

    def serialize(self, serializer: AbstractSerializer) -> None:
        self._elapsed_time = serializer('_elapsed_time', self._elapsed_time)


def report_training_time(trainer: Trainer, key: str='training_time') -> None:
    """Registers extensions to report training time.

    Args:
        trainer: Trainer object to register extensions.
        key: Name with which the training time is reported.

    """
    timing = Timing()
    trainer.extend(timing)
    trainer.extend(TrainingTime(timing), name=key)
