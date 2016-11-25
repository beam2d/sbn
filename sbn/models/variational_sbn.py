from typing import Optional, Sequence, Tuple

from chainer import AbstractSerializer, ChainList, cuda, Link, no_backprop_mode, Variable
import chainer.functions as F

from sbn.random_variable import RandomVariable, SigmoidBernoulliVariable
from sbn.util import Array
from sbn.variational_model import VariationalModel


__all__ = ['VariationalSBN']


class VariationalSBN(VariationalModel):

    """Pair of generative and inference Sigmoid Belief Nets (SBNs) for variational learning.

    SBN consists of one or more layers of sigmoid-Bernoulli variables. Variables in each layer are only wired from the
    variables in the previous layer.

    Let x be the input layer, and z_1, ..., z_L the latent layers. Let us denote z_0 = x for simplicity. The generative
    SBN models conditionals p(z_i | z_{i+1}) for each i in [0, ..., L-1]. For i=L, a prior model p(z_L) is given.

    This class provides an approximate posterior of this model defined by another SBN, named q. The approximate
    posterior model infers latent variables in the inverse direction of generative process: it models q(z_{i+1} | z_i)
    for each i in [0, ..., L-1]. Note that it does not model the input variables x.

    Args:
        generative_layers: Sequence of layer connections for the generative network. The first layer represents
            p(x|z_1) and the last layer represents p(z_{L-1}|z_L).
        inference_layers: Sequence of layer connections for the inference network. The first layer represents
            q(z_1|x) and the last layer represents q(z_L|z_{L-1}).
        prior_size: Number of units in the deepest layer.
        mean: Mean array that subtracts the input to the inference network.

    """
    def __init__(
            self,
            generative_layers: Sequence[Link],
            inference_layers: Sequence[Link],
            prior_size: int,
            mean: Optional[Array]=None,
    ) -> None:
        self.generative_net = ChainList(*generative_layers)
        self.inference_net = ChainList(*inference_layers)

        self.generative_net.add_param('prior', (1, prior_size))
        self.generative_net.prior.data.fill(0)
        self.mean = mean

    @property
    def n_stochastic_layers(self) -> int:
        return len(self.inference_net)

    def cleargrads(self) -> None:
        self.generative_net.cleargrads()
        self.inference_net.cleargrads()

    def copyparams(self, model) -> None:
        self.generative_net.copyparams(model.generative_net)
        self.inference_net.copyparams(model.inference_net)

    def to_gpu(self, device=None) -> None:
        self.generative_net.to_gpu(device)
        self.inference_net.to_gpu(device)
        self.mean = cuda.to_gpu(self.mean, device)

    def to_cpu(self) -> None:
        self.generative_net.to_cpu()
        self.inference_net.to_cpu()
        self.mean = cuda.to_cpu(self.mean)

    def infer(self, x: Variable) -> Tuple[SigmoidBernoulliVariable, ...]:
        if self.mean is not None:
            x = x - self.mean
        zs = []
        for layer in self.inference_net:
            logit = layer(x)
            z = SigmoidBernoulliVariable(logit)
            zs.append(z)
            x = z.sample
        return tuple(zs)

    def compute_generative_factors(
            self,
            x: Variable,
            zs: Sequence[RandomVariable]
    ) -> Tuple[SigmoidBernoulliVariable, ...]:
        ps = []
        for layer, z in zip(self.generative_net, zs):
            logit = layer(z.sample)
            p_x = SigmoidBernoulliVariable(logit, x)
            ps.append(p_x)
            x = z.sample
        # prior
        prior = F.broadcast_to(self.generative_net.prior, zs[-1].sample.shape)
        ps.append(SigmoidBernoulliVariable(prior, zs[-1].sample))
        return tuple(ps)

    def compute_local_signals(self, zs: Sequence[RandomVariable], ps: Sequence[RandomVariable]) -> Tuple[Array, ...]:
        signals = []
        current = ps[-1].log_prob.data
        for q, p in zip(reversed(zs), reversed(ps[:-1])):
            current = current + p.log_prob.data - q.log_prob.data
            signals.append(current)
        return tuple(reversed(signals))

    def compute_local_expectation(
            self,
            x: Variable,
            zs: Sequence[RandomVariable],
            local_signal: Array,
            layer: int
    ) -> Array:
        if layer > 0:
            x = zs[layer - 1].sample
        z_flipped = zs[layer].make_flips()
        B = len(x.data)
        H = zs[layer].sample.shape[1]
        xp = cuda.get_array_module(x.data)

        vb_flipped = 0
        with no_backprop_mode():
            for enc, dec, z in zip(self.inference_net[layer:], self.generative_net[layer:], zs[layer:]):
                if z is not zs[layer]:  # skip the first layer
                    logit = F.reshape(enc(z_flipped.sample.data.reshape(B * H, -1)), (B, H, -1))
                    noise = xp.broadcast_to(z.noise[:, None], logit.shape)
                    z_flipped = SigmoidBernoulliVariable(F.reshape(logit, (B, H, -1)), noise=noise)
                # compute the vb term
                x_logit = F.reshape(dec(z_flipped.sample.data.reshape(B * H, -1)), (B, H, -1))
                x_broadcast = F.broadcast_to(F.reshape(x, (B, 1, -1)), x_logit.shape)
                p_x = SigmoidBernoulliVariable(x_logit, x_broadcast)
                vb_flipped += p_x.log_prob.data - z_flipped.log_prob.data
                x = z.sample

            # prior
            prior = F.broadcast_to(self.generative_net.prior[None], z_flipped.sample.shape)
            vb_flipped += SigmoidBernoulliVariable(prior, z_flipped.sample).log_prob.data

        z = zs[layer].sample.data
        mu = F.sigmoid(zs[layer].logit)
        s = F.where(z.astype('?'), mu, 1 - mu)
        return s * local_signal[:, None] + (1 - s) * vb_flipped

    def serialize(self, serializer: AbstractSerializer) -> None:
        self.generative_net.serialize(serializer['generative_net'])
        self.inference_net.serialize(serializer['inference_net'])
        if self.mean is not None:
            self.mean = serializer('mean', self.mean)
