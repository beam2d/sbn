from typing import Optional, Sequence

from chainer import Chain, ChainList, cuda, Function, Link, Variable
import chainer.functions as F
import numpy as np

from sbn.gradient_estimator import GradientEstimator
from sbn.util import Array
from sbn.variational_model import VariationalModel


__all__ = ['LikelihoodRatioEstimator']


class LogProbOfMean(Function):

    """Numerically stable log-prob of mean."""
    def __init__(self, logit):
        self.logit = logit

    def forward(self, inputs):
        mean, = inputs
        return mean * self.logit - F.Softplus().forward((self.logit,))[0],

    def backward(self, inputs, grad_outputs):
        mean, = inputs
        gy, = grad_outputs
        return gy * self.logit,


def log_prob_of_mean(logit, mean):
    return LogProbOfMean(logit)(mean)


class LikelihoodRatioEstimator(Chain, GradientEstimator):

    """Likelihood-Ratio estimator.

    This class implements the likelihood-ratio estimator with some variance reduction and normalization techniques.
    It actually implements the method called Neural Variational Inference and Learning (NVIL) [1].

    Args:
        model: Model for which it estimates the gradient.
        baseline_models: Local signal predictors for input-dependent baseline. The i-th predictor takes all ancestral
            variables of the i-th latent layer and outputs a B-vector, where B is the mini-batch size.
        alpha: Moving average coefficient of the accumulated signal values. This is the alpha parameter of NVIL, which
            appears in the appendix of [1]. Passing alpha=1 is equivalent to disable the standard baseline.
        normalize_variance: If true, variance normalization is enabled.
        use_muprop: If true, MuProp baseline [2] is enabled.
        n_samples: Number of samples used for Monte Carlo simulations.

    Reference;
        [1]: A. Mnih and K. Gregor. Neural Variational Inference and Learning in Belief Networks. ICML, 2014.
        [2]: S. Gu, S. Levine, I. Sutskever and A. Mnih. MuProp: Unbiased Backpropagation for Stochastic Neural
        Networks. ICLR, 2016.

    """
    def __init__(
            self,
            model: VariationalModel,
            baseline_models: Optional[Sequence[Link]]=None,
            alpha: float=0.8,
            normalize_variance: bool=False,
            use_muprop: bool=False,
            n_samples: int=1
    ) -> None:
        super().__init__()
        self._model = model

        # standard baseline (moving average of local signals)
        self.add_persistent('c', np.zeros(model.n_stochastic_layers, dtype='f'))
        self._coeff = 1 - alpha

        # variance normalization
        if normalize_variance:
            self.add_persistent('v', np.zeros(model.n_stochastic_layers, dtype='f'))
        else:
            self.v = None  # type: Optional[Array]

        # input-dependent baseline
        if baseline_models is None:
            self.baseline_models = None  # type: Optional[ChainList]
        else:
            self.add_link('baseline_models', ChainList(*baseline_models))

        self._use_muprop = use_muprop
        self._n_samples = n_samples

    def estimate_gradient(self, x: Variable) -> None:
        xp = cuda.get_array_module(x.data)
        K = self._n_samples
        if K > 1:
            x = Variable(x.data.repeat(K, axis=0), volatile=x.volatile)

        zs = self._model.infer(x)
        ps = self._model.compute_generative_factors(x, zs)
        signals = xp.vstack(self._model.compute_local_signals(zs, ps))  # (L, B)-matrix

        # MuProp baseline
        residuals = 0
        if self._use_muprop:
            zs_mf = self._model.infer(x, mean_field=True)
            ps_mf = self._model.compute_generative_factors(x, zs_mf)

            q_mf = F.sum(F.vstack([F.sum(log_prob_of_mean(z.logit.data, z.sample), axis=-1)[None] for z in zs_mf]))
            p_mf = F.sum(F.vstack([p.log_prob[None] for p in ps_mf]))
            (p_mf - q_mf).backward(retain_grad=True)
            zs_mf_grad = [z.sample.grad for z in zs_mf]

            signals -= xp.vstack(self._model.compute_local_signals(zs_mf, ps_mf))
            signals -= xp.vstack([(gf * (z.sample.data - z_mf.sample.data)).sum(axis=1)[None]
                                  for gf, z, z_mf in zip(zs_mf_grad, zs, zs_mf)])

            residuals = F.sum(F.vstack([F.sum(z.mean * gf) for z, gf in zip(zs, zs_mf_grad)]))

        # input-dependent baseline
        if self.baseline_models is not None:
            baselines_list = []
            args = [x]
            mean = getattr(self._model, 'mean', None)
            if mean is not None:
                args[0] = x - mean
            for z, bl_model in zip(zs, self.baseline_models):
                bl = bl_model(*args)
                baselines_list.append(bl)
                args.append(z.sample)
            baselines = F.vstack(baselines_list)
            signals -= baselines.data

        # standard baseline and variance normalization
        if self._coeff > 0:
            self.c += self._coeff * (signals.mean(axis=1) - self.c)  # TODO(beam2d): Unify the kernel
            if self.v is not None:
                self.v += self._coeff * (signals.var(axis=1) - self.v)  # TODO(beam2d): Unify the kernel
                signals -= self.c[:, None]
                signals /= xp.maximum(1, xp.sqrt(self.v))[:, None]
            else:
                signals -= self.c[:, None]

        p_terms = F.sum(F.vstack([p.log_prob for p in ps]))
        q_terms = F.sum(signals * F.vstack([z.log_prob for z in zs]))
        # Note: we have to compute the gradient w.r.t. the bound of the NEGATIVE log likelihood
        self._model.cleargrads()
        ((-p_terms - q_terms - residuals) / K).backward()

        if self.baseline_models is not None:
            bl_terms = -F.sum(signals * baselines) / K
            self.baseline_models.cleargrads()
            bl_terms.backward()

    def get_estimator_model(self) -> Optional[Link]:
        return self.baseline_models
