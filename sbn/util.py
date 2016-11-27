import functools
from typing import Any, Callable, Sequence, Union

from chainer import cuda, Function, Variable
import numpy as np


__all__ = ['Array', 'backprop_from_all', 'cached_property', 'KahanSum']


# Type hinting utilities
if cuda.available:
    import cupy as cp
    Array = Union[np.ndarray, cp.ndarray]
else:
    Array = np.ndarray


# Method result caching
def cached_property(meth: Callable[[Any], Any]):
    """Caches the result of a method without arguments."""
    attr_name = '_' + meth.__name__

    @functools.wraps(meth)
    def method(self):
        ret = getattr(self, attr_name, None)
        if ret is None:
            ret = meth(self)
            setattr(self, attr_name, ret)
        return ret

    return property(method)


# Kahan summation algorithm (see https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
class KahanSum:
    """Kahan summation algorithm.

    This class implements Kahan summation algorithm (https://en.wikipedia.org/wiki/Kahan_summation_algorithm). It
    computes summation of a stream of values with larger precision than each scalar.

    """
    def __init__(self):
        self.sum = 0.
        self.c = 0.
        self.count = 0

    def add(self, value, count: int=1) -> None:
        y = value - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t

        self.count += count

    @property
    def mean(self):
        return self.sum / self.count
