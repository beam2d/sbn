from chainer import cuda, functions as F, links as L, Variable
import numpy as np

from sbn import GradientEstimator, SigmoidBernoulliVariable
from sbn.models import VariationalSBN


__all__ = ['GradientEstimationTester']


class GradientEstimationTester:

    """Helper class for testing gradient estimators.

    This class compares estimated gradients with the true gradient. The true gradient is obtained by analytically
    computing the expectation over q(z | x). In order to realize the analytical solution, this class uses a very small
    model with only 2^3 = 8 latent variables.

    """
    def __init__(self) -> None:
        self.inf_layers = [L.Linear(3, 2), L.Linear(2, 2), L.Linear(2, 2)]
        self.gen_layers = [L.Linear(2, 3), L.Linear(2, 2), L.Linear(2, 2)]
        self.mean = np.random.rand(3).astype('f')
        self.model = VariationalSBN(self.gen_layers, self.inf_layers, 2, self.mean)

    def to_gpu(self) -> None:
        self.model.to_gpu()
        self.mean = self.model.mean

    def gen_x(self, n_sample: int=3) -> Variable:
        return Variable(np.random.binomial(1, 0.5, (n_sample, 3)).astype('f'))

    def estimate_variational_bound_gradient(self, x: Variable) -> None:
        xp = cuda.get_array_module(x.data)
        signal = 0
        x_in = Variable(x.data - self.mean)
        q_z = xp.array([1] * len(x.data), dtype='f')
        for l, (enc, dec) in enumerate(zip(self.inf_layers, self.gen_layers)):
            z_logit = enc(x_in)
            z = _gen_binary_combinations(xp, z_logit.shape[1])
            z_logit, z = F.broadcast(z_logit[:, None], z[None])
            z_var = SigmoidBernoulliVariable(z_logit, z)
            q_z = F.broadcast_to(q_z[:, None], z_var.prob.shape) * z_var.prob
            signal -= F.sum(z_var.log_prob * q_z)

            q_z = F.reshape(q_z, (-1,))

            x = F.reshape(F.broadcast_to(x[:, None], (x.shape[0], z.shape[1], x.shape[1])), (-1, x.shape[1]))
            z = F.reshape(z, (len(q_z.data), -1))
            x_logit = dec(z)
            p = SigmoidBernoulliVariable(x_logit, x)
            signal += F.sum(p.log_prob * q_z)

            x = x_in = z

        prior = F.broadcast_to(self.model.generative_net.prior, z.shape)
        p = SigmoidBernoulliVariable(prior, z)
        signal += F.sum(p.log_prob * q_z)

        self.model.cleargrads()
        (-signal).backward()

    def dry_run(self, x: Variable, estimator: GradientEstimator, n_sample: int, trial: int) -> None:
        x_repeat = Variable(x.data.repeat(n_sample, axis=0))
        for t in range(trial):
            estimator.estimate_gradient(x_repeat)

    def check_estimator(
            self,
            x: Variable,
            estimator: GradientEstimator,
            n_sample: int=10000,
            trial: int=10,
            rtol: float=1e-3,
            atol: float=1e-3,
    ) -> None:
        xp = cuda.get_array_module(x.data)
        self.estimate_variational_bound_gradient(x)
        expect_gen_grads = {k: v.grad.copy() for k, v in self.model.generative_net.namedparams()}
        expect_inf_grads = {k: v.grad.copy() for k, v in self.model.inference_net.namedparams()}

        x_repeat = Variable(x.data.repeat(n_sample, axis=0))
        gen_grads_accum = None
        inf_grads_accum = None
        for t in range(trial):
            estimator.estimate_gradient(x_repeat)
            gen_grads = {k: v.grad / n_sample for k, v in self.model.generative_net.namedparams()}
            inf_grads = {k: v.grad / n_sample for k, v in self.model.inference_net.namedparams()}
            if gen_grads_accum is None:
                gen_grads_accum = gen_grads
                inf_grads_accum = inf_grads
            else:
                for k in gen_grads:
                    gen_grads_accum[k] += gen_grads[k]
                for k in inf_grads:
                    inf_grads_accum[k] += inf_grads[k]

        for k in gen_grads:
            xp.testing.assert_allclose(gen_grads_accum[k] / trial, expect_gen_grads[k], rtol=rtol, atol=atol)

        for k in inf_grads:
            xp.testing.assert_allclose(inf_grads_accum[k] / trial, expect_inf_grads[k], rtol=rtol, atol=atol)


def _gen_binary_combinations(xp, dim: int) -> Variable:
    n_comb = 2 ** dim
    if n_comb > 256:
        raise ValueError('too many combinations')
    comb = np.unpackbits(np.arange(n_comb, dtype='B')).reshape(n_comb, 8)[:, -dim:].astype('f')
    return Variable(xp.asarray(comb))
