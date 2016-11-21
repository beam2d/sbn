from typing import Optional, Sequence

from chainer import Chain, ChainList, cuda, Link, Variable
import chainer.functions as F
import numpy as np

from sbn.gradient_estimator import GradientEstimator
from sbn.util import Array
from sbn.variational_model import VariationalModel


__all__ = ['LikelihoodRatioEstimator']


class LikelihoodRatioEstimator(Chain, GradientEstimator):

    """Likelihood-Ratio estimator.

    This class implements the likelihood-ratio estimator with some variance reduction and normalization techniques.
    It actually implements the method called Neural Variational Inference and Learning (NVIL) [1].

    Args:
        model: Model for which it estimates the gradient.
        baseline_models: Local signal predictors for input-dependent baseline. The i-th predictor takes all ancestral
            variables of the i-th latent layer and outputs a B-vector, where B is the mini-batch size.
        alpha: Moving average coefficient of the accumulated signal values. This is the alpha parameter of NVIL, which
            appears in the appendix of [1].
        normalize_variance: If true, variance normalization is enabled.
        n_samples: Number of samples used for Monte Carlo simulations.

    Reference;
        [1]: A. Mnih and K. Gregor. Neural Variational Inference and Learning in Belief Networks. ICML, 2014.

    """
    def __init__(
            self,
            model: VariationalModel,
            baseline_models: Optional[Sequence[Link]]=None,
            alpha: float=0.8,
            normalize_variance=False,
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

        self._n_samples = n_samples

    def to_cpu(self) -> None:
        self.c = cuda.to_cpu(self.c)
        if self.v is not None:
            self.v = cuda.to_cpu(self.v)
        if self.baseline_models is not None:
            self.baseline_models.to_cpu()

    def to_gpu(self, device=None) -> None:
        self.c = cuda.to_gpu(self.c, device)
        if self.v is not None:
            self.v = cuda.to_gpu(self.v, device)
        if self.baseline_models is not None:
            self.baseline_models.to_gpu(device)

    def estimate_gradient(self, x: Variable) -> None:
        xp = cuda.get_array_module(x.data)
        K = self._n_samples
        if K > 1:
            x = Variable(x.data.repeat(K, axis=0), volatile=x.volatile)

        zs = self._model.infer(x)
        ps = self._model.compute_generative_factors(x, zs)
        signals = xp.vstack(self._model.compute_local_signals(zs, ps))  # (L, B)-matrix

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
        ((-p_terms - q_terms) / K).backward()

        if self.baseline_models is not None:
            bl_terms = -F.sum(signals * baselines) / K
            self.baseline_models.cleargrads()
            bl_terms.backward()

    def get_estimator_model(self) -> Optional[Link]:
        return self.baseline_models
