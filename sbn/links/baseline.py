from chainer import Chain, functions as F, links as L, Variable


__all__ = ['BaselineModel']


class BaselineModel(Chain):

    """The input-dependent baseline model used in [1].

    This is a multi-layer perceptron with one hidden layer for regression. It uses the tanh nonlinearity.

    Reference:
        [1]: A. Mnih and K. Gregor. Neural Variational Inference and Learning in Belief Networks. ICML, 2014.

    Args:
        n_unit: Number of units in the hidden layer.

    """
    def __init__(self, n_unit: int=200) -> None:
        super().__init__(
            l1=L.Linear(None, n_unit),
            l2=L.Linear(None, 1)
        )

    def __call__(self, x: Variable) -> Variable:
        B = len(x.data)
        h = F.tanh(self.l1(x))
        return F.reshape(self.l2(h), (B,))
