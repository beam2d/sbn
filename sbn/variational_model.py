import math
from typing import Sequence, Tuple

from chainer import AbstractSerializer, Variable
import chainer.functions as F

from sbn.random_variable import RandomVariable
from sbn.util import Array


__all__ = ['VariationalModel']


class VariationalModel:

    """Interface of variational models.

    This class provides an abstract interface of a variational model, a pair of two models: a generative model and a
    corresponding approximate posterior model.

    """
    @property
    def n_stochastic_layers(self) -> int:
        """Number of layers of latent variables in the model.

        This value should be equal to the length of the result of :meth:`infer`.

        """
        raise NotImplementedError

    def cleargrads(self) -> None:
        """Clears all gradients of the parameters."""
        raise NotImplementedError

    def copyparams(self, model) -> None:
        """Copies parameters from another model."""
        raise NotImplementedError

    def to_cpu(self) -> None:
        """Migrates all parameters and arrays in the model to CPU."""
        raise NotImplementedError

    def to_gpu(self, device=None) -> None:
        """Migrates all parameters and arrays in the model to GPU.

        Args:
            device: Device specifier.

        """
        raise NotImplementedError

    def infer(self, x: Variable) -> Tuple[RandomVariable, ...]:
        """Infers latent variables of the approximate posterior model given the input configuration x.

        The returned variables follow the topological order of the directed graphical model.

        Args:
            x: Configuration of input variables.

        Returns:
            Tuple of inferred latent variables.

        """
        raise NotImplementedError

    def compute_generative_factors(
            self,
            x: Variable,
            zs: Sequence[RandomVariable],
    ) -> Tuple[RandomVariable, ...]:
        """Computes all conditionals of the generative model with fully-given configurations.

        Given configurations of all variables, this method computes all conditionals of the generative model.

        Args:
            x: Configuration of input variables.
            zs: Configurations of all latent variables.

        Returns:
            Tuple of conditionals and priors. The conditional factor of the input variable is put to the head of the
            tuple, and those of latent variables follow it in the same order of the inferred latent variables.

        """
        raise NotImplementedError

    def compute_variational_bound(
            self,
            zs: Sequence[RandomVariable],
            ps: Sequence[RandomVariable],
            marginalize_q=False
    ) -> Array:
        """Estimates the variational bound of the log likelihood.

        This method estimates the variational bound of the log likelihood, namely ``log p(x, z) - log q(z | x)``.

        Args:
            zs: Inferred latent variables.
            ps: Factors of generative models.
            marginalize_q: If true, the log q(z|x) terms use the locally-marginalized entropy. Otherwise, it uses the
                sampled configurations of z to compute each term of log q(z|x).

        Returns:
            An estimation of the variational bound of the log likelihood.

        """
        if marginalize_q:
            entropy = sum(z.entropy.data for z in zs)
        else:
            entropy = -sum(z.log_prob.data for z in zs)
        log_p = sum(p.log_prob.data for p in ps)
        return log_p + entropy

    def compute_monte_carlo_bound(
            self,
            variational_bound: Array,
            n_samples: int,
    ) -> Array:
        """Estimates the Monte Carlo bound of the log likelihood.

        The Monte Carlo bound (a.k.a. importance weighting estimate) is originally proposed in [1] and also used for
        discrete models in [2]. This method computes the Monte Carlo objectives using variables simulated multiple
        times.

        Let K be the sample size used for the Monte Carlo bound, and B the mini-batch size. For each input x, this
        method computes ``log (1/K sum_{k=1,...,K} p(x, z^k)/q(z^k|x))`` where ``z^1, ..., z^K`` are independent
        samples of ``q(z|x)``.

        This method accepts the variational bound of already simulated ``z^1, ..., z^K`` and the corresponding
        generative factors. The variational bound should be of length B*K, where K samples for each input form a
        contiguous block, i.e., it can be reshaped to ``(B, K)``.

        Reference:
            [1]: Y. Burda, R. Grosse, and R. Salakhutdinov. Importance Weighted Autoencoders. ICLR, 2016.
            [2]: A. Mnih and D. J. Rezende. Variational inference for Monte Carlo objectives. ICML, 2016.

        Args:
            variational_bound: Variational bound of the log likelihood computed for each Monte Carlo sample.
            n_samples: Sample size for each input example, i.e., ``K`` in the above description.

        Returns:
            The Monte Carlo bound of each example.

        """
        B = len(variational_bound) // n_samples
        lse, = F.LogSumExp(axis=1).forward((variational_bound.reshape(B, n_samples),))
        lse -= math.log(n_samples)
        return lse

    def compute_local_signals(self, zs: Sequence[RandomVariable], ps: Sequence[RandomVariable]) -> Tuple[Array, ...]:
        """Computes the local signal of each latent variable.

        This method computes a local signal for each latent variable. The variational bound is written by a summation
        of many terms, and a local signal of a latent variable is the partial sum of these terms that depend on the
        variable.

        Args:
            zs: Inferred latent variables.
            ps: Factors of generative models.

        Returns:
            Local signals of latent variables. The i-th element represents the local signal of the i-th variable.

        """
        raise NotImplementedError

    def compute_local_expectation(
            self,
            x: Variable,
            zs: Sequence[RandomVariable],
            local_signal: Array,
            layer: int
    ) -> Variable:
        """Computes the local signal of the specified layer marginalized over each variable.

        This method computes local expectation of the variational bound for each variable of ``zs[layer]``. It is done
        by flipping each element of zs[layer] and simulate zs[layer+1:] given flipped configurations. The simulation
        follows the reparameterization trick for discrete variables, i.e., all latent variables of zs[layer+1:] are
        simulated with the fixed noise.

        Args:
            x: Input variable.
            zs: Inferred latent variables.
            local_signal: Local signal of the layer.
            layer: Which layer to compute the local expectations.

        Returns:
            Variable of the same shape as ``zs[layer]``. It can be backprop-ed through the logit of the variable.

        """
        raise NotImplementedError

    def serialize(self, serializer: AbstractSerializer) -> None:
        """Serializes the model."""
        raise NotImplementedError
