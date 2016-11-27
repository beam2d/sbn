from typing import Optional, Tuple

from chainer import cuda, Variable
import chainer.functions as F

from sbn.util import Array, cached_property


__all__ = ['ConstantVariable', 'RandomVariable', 'SigmoidBernoulliVariable']


class RandomVariable:

    """Random variable with a sampled configuration.

    This interface represents p(x|w) where x is a configuration of the variable and w a parameter of the distribution.
    The parameterization and representation is defined by each implementation.

    """
    @property
    def sample(self) -> Variable:
        """Sampled configuration."""
        raise NotImplementedError

    @property
    def log_prob(self) -> Variable:
        """Log probability of the sample.

        This variable is a B-dimensional vector where B is the mini-batch size.

        """
        raise NotImplementedError

    @property
    def mean(self) -> Variable:
        """Mean value of the distribution."""
        raise NotImplementedError

    @property
    def entropy(self) -> Variable:
        """Entropy of the distribution.

        This variable is a B-dimensional vector where B is the mini-batch size.

        """
        raise NotImplementedError

    def make_flips(self):
        """Creates an array of random variables with each element flipped.

        This method returns a new RandomVariable of the shape ``(B, C, D)``, where B is the batch size, C the number of
        flips, and D the dimensionality of each example. The length C depends on the type of the random variable.

        Returns:
            RandomVariable: Sample with flipped values.

        """
        raise NotImplementedError

    def reshape(self, *shape: int):
        """Returns a reshaped random variable.

        Args:
            shape: New shape.

        Returns:
            Random variable of the new shape.

        """
        raise NotImplementedError


class ConstantVariable(RandomVariable):

    """Random variable representing a constant value.

    Args:
        value (array): Sampled value.

    """
    def __init__(self, value: Array) -> None:
        self._sample = Variable(value)
        self._xp = cuda.get_array_module(value)

    @property
    def sample(self) -> Variable:
        return self._sample

    @property
    def log_prob(self) -> Variable:
        return self.zeros

    @property
    def mean(self) -> Variable:
        return self._sample

    @property
    def entropy(self) -> Variable:
        return self.zeros

    @cached_property
    def zeros(self) -> Variable:
        return Variable(self._xp.zeros(len(self._sample.data), dtype=self._sample.dtype))

    def reshape(self, shape: Tuple[int, ...]):
        return ConstantVariable(self._sample.reshape(shape))


class SigmoidBernoulliVariable(RandomVariable):

    """Bernoulli variable with given logit value.

    Given logit value, this class represents a Bernoulli distribution with a mean parameter ``sigmoid(logit)``.

    Args:
        logit: Logit of the Bernoulli variable.
        sample: Sampled configuration.
        noise: Noise used to sample a configuration. It should be a sample from the Uniform distribution over the
            interval [0, 1].

    """
    def __init__(
            self,
            logit: Variable,
            sample: Optional[Variable]=None,
            noise: Optional[Array]=None
    ) -> None:
        self.logit = logit
        self._noise = noise
        self._sample = sample

        if sample is not None and logit.shape != sample.shape:
            raise ValueError('shape mismatched between logit and sample: {} and {}'.format(logit.shape, sample.shape))
        if noise is not None and logit.shape != noise.shape:
            raise ValueError('shape mismatched between logit and noise: {} and {}'.format(logit.shape, noise.shape))

    @cached_property
    def mean(self) -> Variable:
        return F.sigmoid(self.logit)

    @cached_property
    def sample(self) -> Variable:
        mean = self.mean
        return Variable((self.noise < mean.data).astype(mean.dtype))

    @cached_property
    def log_prob(self) -> Variable:
        return F.sum(self.sample * self.logit - self.softplus_logit, axis=-1)

    @cached_property
    def entropy(self) -> Variable:
        # TODO(beam2d): Unify the kernels.
        return F.sum(self.softplus_logit - self.mean * self.logit, axis=-1)

    @cached_property
    def softplus_logit(self) -> Variable:
        return F.softplus(self.logit)

    @cached_property
    def noise(self) -> Array:
        mean = self.mean
        xp = cuda.get_array_module(mean.data)
        return xp.random.rand(*mean.shape).astype(mean.dtype)

    def make_flips(self):
        # Create D copies of a (B, D) binary array, where the (*, i)-th elements are flipped in the i-th copy.
        # The returned variable is of shape (B, D, D).
        z = self.sample.data
        z = z[:, None].repeat(z.shape[1], axis=1)
        z_diag = z.diagonal(axis1=1, axis2=2)
        z_diag.flags.writeable = True
        z_diag[...] = 1 - z_diag  # flip

        xp = cuda.get_array_module(z)
        logit = F.broadcast_to(self.logit[:, None], z.shape)
        noise = xp.broadcast_to(self.noise[:, None], z.shape)
        return SigmoidBernoulliVariable(logit, Variable(z), noise)

    def reshape(self, *shape: int):
        return SigmoidBernoulliVariable(F.reshape(self.logit, shape), F.reshape(self.sample, shape))
