from sbn.util import Array

__all__ = ['GradientEstimator']


class GradientEstimator:

    """Abstract gradient estimator for sigmoid belief nets.

    This class provides an interface of general gradient estimators. It can have its own parameters (e.g. baseline
    estimator). In such a case, the implementation should also inherits ``chainer.Link`` or ``chainer.Chain``.

    """
    def estimate_gradient(self, x: Array) -> None:
        """Estimates a gradient w.r.t. a given input array.

        This method computes a gradient of an underlying belief net model with given input ``x``. The gradient is
        directly stored to the ``grad`` attribute of each parameter variable in the model. Note that the gradient is
        not reset before the estimation, and thus users have to appropriately initialize the gradient arrays before
        calling this method.

        Args:
            x: Input array.

        """
        raise NotImplementedError
