import functools
from typing import Any, Callable, Union

from chainer import cuda
import numpy as np


__all__ = ['Array', 'cached_property']


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
