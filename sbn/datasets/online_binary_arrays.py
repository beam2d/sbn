from typing import Tuple

from chainer import cuda

from sbn.util import Array


__all__ = ['OnlineBinaryArrays']


class OnlineBinaryArrays:

    def __init__(self, base: Array) -> None:
        self.base = base

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.base.shape

    def __getitem__(self, i) -> Array:
        x = self.base[i]
        xp = cuda.get_array_module(x)
        return (xp.random.rand(*x.shape) < x).astype(x.dtype)

    def __len__(self) -> int:
        return len(self.base)

