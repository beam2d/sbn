from typing import Optional, Sequence

from chainer import Chain, ChainList, cuda, Link, Variable
import chainer.functions as F
import numpy as np

from sbn.grad_estimator import GradientEstimator
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

    Reference;
        [1]: A. Mnih and K. Gregor. Neural Variational Inference and Learning in Belief Networks. ICML, 2014.

    """
    def __init__(
            self,
            model: VariationalModel,
            baseline_models: Optional[Sequence[Link]]=None,
            alpha: float=0.8,
            normalize_variance=False
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

    def estimate_gradient(self, x: Variable) -> None:
        xp = cuda.get_array_module(x.data)

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
                args.append(z)
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
        (-p_terms - q_terms).backward()

        if self.baseline_models is not None:
            bl_terms = -F.sum(signals * baselines)
            self.baseline_models.cleargrads()
            bl_terms.backward()

    def get_estimator_model(self) -> Optional[Link]:
        return self.baseline_models