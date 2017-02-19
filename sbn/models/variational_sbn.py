from typing import Optional, Sequence, Tuple

from chainer import AbstractSerializer, ChainList, cuda, Link, Variable
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

    def infer(self, x: Variable, mean_field: bool=False) -> Tuple[SigmoidBernoulliVariable, ...]:
        if self.mean is not None:
            x = x - self.mean
        zs = []
        for layer in self.inference_net:
            logit = layer(x)
            z = SigmoidBernoulliVariable(logit)
            if mean_field:
                z._sample = z.mean
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
            current = current + p.log_prob.data - q.log_prob.data  # do not use += here
            signals.append(current)
        return tuple(reversed(signals))

    def compute_local_marginal_signals(
            self,
            zs: Sequence[SigmoidBernoulliVariable],
            ps: Sequence[SigmoidBernoulliVariable]
    ) -> Tuple[Array, ...]:
        signals = []
        current = SigmoidBernoulliVariable(ps[-1].logit, zs[-1].mean).log_prob.data
        for q, p in zip(reversed(zs), reversed(ps[:-1])):
            current += p.log_prob.data
            signals.append(current)
            if q is not zs[0]:  # skip the last redundant addition
                current = current + q.entropy.data  # do not use += here
        return tuple(reversed(signals))

    def compute_local_expectation(
            self,
            x: Variable,
            zs: Sequence[SigmoidBernoulliVariable],
            ps: Sequence[SigmoidBernoulliVariable],
            layer: int
    ) -> Variable:
        xp = cuda.get_array_module(x.data)
        z = zs[layer]
        p = ps[layer]
        p_z = ps[layer + 1]
        B = len(z.sample.data)
        H = z.sample.shape[1]
        vb = xp.broadcast_to(self.compute_variational_bound(zs, ps)[:, None], (B, H))

        # z_l with all bits flipped
        z_all_flipped = SigmoidBernoulliVariable(z.logit, 1 - z.sample)
        # Copies of z_l with one bit flipped (whose shape is (B, H, H))
        z_flipped = z.make_flips()
        log_z_flipped = (z.log_prob.data[:, None] + z_all_flipped.elementwise_log_prob.data
                         - z.elementwise_log_prob.data)

        # Compute p(x, z) with flipped z
        base_score = sum(p.log_prob.data - z.log_prob.data
                         for l, (p, z) in enumerate(zip(ps, zs)) if not (layer <= l <= layer + 1))
        dec = self.generative_net[layer]
        p_flipped = SigmoidBernoulliVariable(
            F.reshape(dec(F.reshape(z_flipped.sample, (B * H, H))), (B, H, -1)),
            F.broadcast_to(p.sample[:, None], (B, H, p.sample.shape[1])))
        p_z_flipped = SigmoidBernoulliVariable(p_z.logit, 1 - p_z.sample)
        log_p_z_flipped = (p_z.log_prob.data[:, None] + p_z_flipped.elementwise_log_prob.data
                           - p_z.elementwise_log_prob.data)
        if isinstance(base_score, xp.ndarray):
            base_score = base_score[:, None]
        vb_flipped = base_score + p_flipped.log_prob.data + log_p_z_flipped - log_z_flipped

        if layer == len(zs) - 1:  # last layer
            coeff = xp.where(z.sample.data, z.mean.data, 1 - z.mean.data)
        else:
            enc = self.inference_net[layer + 1]
            y = zs[layer + 1]
            H_y = y.sample.shape[1]
            y_flipped = SigmoidBernoulliVariable(F.reshape(enc(F.reshape(z_flipped.sample, (B * H, H))), (B, H, -1)),
                                                 F.broadcast_to(y.sample[:, None], (B, H, H_y)))
            vb_flipped += ps[-1].log_prob.data[:, None] - y_flipped.log_prob.data
            coeff = F.sigmoid(z.elementwise_log_prob + F.broadcast_to(y.log_prob[:, None], (B, H))
                              - z_all_flipped.elementwise_log_prob - y_flipped.log_prob).data

        localexp = (vb * coeff * z.elementwise_log_prob +
                    vb_flipped * (1 - coeff) * z_all_flipped.elementwise_log_prob)
        return localexp

    def compute_reparameterized_local_expectation(
            self,
            x: Variable,
            zs: Sequence[RandomVariable],
            ps: Sequence[RandomVariable],
            layer: int
    ) -> Array:
        if layer > 0:
            x = zs[layer - 1].sample
        B = len(x.data)
        H = zs[layer].sample.shape[1]
        xp = cuda.get_array_module(x.data)
        vb = xp.broadcast_to(self.compute_variational_bound(zs, ps, marginalize_q=True)[:, None], (B, H))

        z_flipped = zs[layer].make_flips()
        if layer == 0:
            vb_flipped = xp.zeros((B, H), dtype='f')
        else:
            vb_flipped = sum(p.log_prob.data + z.entropy.data for l, (p, z) in enumerate(zip(ps, zs)) if l < layer)
            vb_flipped = vb_flipped[:, None].repeat(H, axis=1)

        x_current = F.broadcast_to(F.reshape(x, (B, 1, -1)), (B, H, x.shape[1]))
        for enc, dec, z in zip(self.inference_net[layer:], self.generative_net[layer:], zs[layer:]):
            if z is not zs[layer]:  # skip the first layer
                logit_flipped = F.reshape(enc(x_current.data.reshape(B * H, -1)), (B, H, -1))
                noise = xp.broadcast_to(z.noise[:, None], logit_flipped.shape)
                z_flipped = SigmoidBernoulliVariable(F.reshape(logit_flipped, (B, H, -1)), noise=noise)

            x_logit = F.reshape(dec(z_flipped.sample.data.reshape(B * H, -1)), (B, H, -1))
            p_x = SigmoidBernoulliVariable(x_logit, x_current)
            vb_flipped += p_x.log_prob.data
            vb_flipped += z_flipped.entropy.data

            x_current = z_flipped.sample

        # prior
        prior = F.broadcast_to(self.generative_net.prior[None], x_current.shape)
        p_z = SigmoidBernoulliVariable(prior, z_flipped.mean)
        vb_flipped += p_z.log_prob.data

        q_z = zs[layer].elementwise_prob
        return q_z * (vb - vb_flipped) + vb_flipped

    def serialize(self, serializer: AbstractSerializer) -> None:
        self.generative_net.serialize(serializer['generative_net'])
        self.inference_net.serialize(serializer['inference_net'])
        if self.mean is not None:
            self.mean = serializer('mean', self.mean)
