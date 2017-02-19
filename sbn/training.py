from contextlib import contextmanager
import copy
import json
import os
from typing import Any, Optional, Tuple

from chainer import Chain, Function, is_debug, Link, Optimizer, set_debug, Variable
from chainer import cuda, functions as F, links as L, optimizers
from chainer.initializers import GlorotUniform
from chainer.iterators import SerialIterator
from chainer.optimizer import WeightDecay
from chainer.serializers import load_npz
from chainer.training import Trainer
from chainer.training.extensions import LogReport, PrintReport, ProgressBar, snapshot, snapshot_object
import numpy as np
import yaml

from sbn.datasets import get_offline_binary_mnist, get_online_binary_mnist
from sbn.datasets import get_offline_binary_omniglot, get_online_binary_omniglot
from sbn.estimators import DiscreteReparameterizationEstimator, LikelihoodRatioEstimator
from sbn.estimators import LocalExpectationGradientEstimator
from sbn.extensions import evaluate_gradient_variance, evaluate_log_likelihood, KeepBestModel, report_training_time
from sbn.extensions import LogLikelihoodEvaluator
from sbn.gradient_estimator import GradientEstimator
from sbn.links import BaselineModel
from sbn.models import VariationalSBN
from sbn.updater import Updater
from sbn.variational_model import VariationalModel


__all__ = ['train_variational_model']


def train_variational_model(
        config: str,
        gpu: int,
        resume: str='',
        debug: bool=False,
        verbose: bool=False
) -> VariationalModel:
    """Trains a variational model.

    Args:
        config: YAML config string.
        gpu: GPU device ID to use (-1 for CPU).
        resume: Path to the trainer snapshot to resume from. If it is 'auto', the latest snapshot is automatically
            chosen from the output directory.
        debug: If True, it runs in the debug mode.
        verbose: If True, it prints messages.

    Following is the YAML config entries.

    - name: Name of the experiment. The output path is set to 'result/<name>'.
    - dataset: Dataset name.
    - binarize_online: If true, binarization is done online (True by default).
    - inference_net: Inference network configuration. It is a dictionary containing the following entries.

      - type: Type of the network (linear by default).
      - units: Shape of each layer (from the shallowest layer to the deepest layer).
        Each entry is in the form [in_size, out_size].
      - optimizer: Optimizer configuration for the network. It is a dictionary containing the following entries.

        - method: Optimization method (sgd by default).
        - (other entries): Other entries are used for initializing the optimizer.

    - generative_net: Generative network configuration. The format is same as inference_net.
    - estimator: Gradient estimator configuration. It is a dictionary containing the following entries.

      - method: Estimation method.
      - optimizer: Optimizer configuration for the estimator. The format is same as that of the inference net.
      - (options for method=likelihood_ratio):

        - use_baseline_model: If true, use baseline models (False by default).
        - alpha: Alpha parameter of the baseline/variance estimation (0.8 by default).
        - normalize_variance: If true, use variance normalization (False by default).
        - use_muprop: If true, use MuProp baseline (False by default).
        - n_samples: Number of samples used for each Monte Carlo simulation.

    - batch_size: Training mini-batch size (100 by default).
    - iteration: Number of iterations to train.
    - eval_batch_size: Evaluation mini-batch size (100 by default).
    - eval_interval: Evaluation interval in iterations (50000 by default).
    - n_eval_epochs: Number of epochs for each evaluation (1 by default).
    - n_eval_samples: Number of samples used for validating the Monte Carlo objective (50 by default).
    - grad_eval_iteration: Number of iterations for evaluating the gradient variance (1000 by default).
    - snapshot_interval: Snapshot interval in iterations (1000000 by default).
    - test_batch_size: Final test mini-batch size (1 by default).
    - n_test_epochs: Number of epochs for the final evaluation (1 by default).
    - n_test_samples: Number of samples used for the final evaluation of the Monte Carlo objective (50000 by default).

    Returns:
        Learned variational model.

    """
    with _debug_mode(debug):
        with cuda.get_device(gpu if gpu >= 0 else None):
            return _train_variational_model(config, gpu, resume, verbose)


@contextmanager
def _debug_mode(debug: bool):
    original_debug = is_debug()
    set_debug(debug)

    original_type_check = Function.type_check_enable
    Function.type_check_enable = debug

    yield

    Function.type_check_enable = original_type_check
    set_debug(original_debug)


def _train_variational_model(config_raw: str, gpu: int, resume: str, verbose: bool) -> VariationalModel:
    config = yaml.load(config_raw)
    exp_name = config['name']
    out_path = os.path.join('result', exp_name)
    use_gpu = gpu >= 0

    # Load datasets
    mean, train, valid, test = _get_dataset(config['dataset'], config.get('binarize_online', True), use_gpu)
    infer_layers, prior_size = _build_layers(config['inference_net'])
    gen_layers, _ = _build_layers(config['generative_net'])

    # Set up a model and optimizers
    model = VariationalSBN(gen_layers, infer_layers, prior_size, mean)
    estimator = _build_estimator(config['estimator'], len(infer_layers), model)
    if use_gpu:
        model.to_gpu()
        estimator.to_gpu()

    gen_optimizer = _build_optimizer(config['generative_net']['optimizer'], model.generative_net)
    infer_optimizer = _build_optimizer(config['inference_net']['optimizer'], model.inference_net)

    estimator_model = estimator.get_estimator_model()
    est_optimizer = _build_optimizer(config['estimator']['optimizer'], estimator_model)

    # Set up a trainer
    train_batch_size = config.get('batch_size', 100)
    train_iterator = SerialIterator(train, train_batch_size)
    updater = Updater(estimator, train_iterator, gen_optimizer, infer_optimizer, est_optimizer)
    trainer = Trainer(updater, (config['iteration'], 'iteration'), out_path)

    # Add extensions
    eval_interval = config.get('eval_interval', 50000), 'iteration'
    eval_batch_size = config.get('eval_batch_size', 100)
    n_eval_epochs = config.get('n_eval_epochs', 1)
    n_eval_samples = config.get('n_eval_samples', 50)

    train_eval_iterator = SerialIterator(train, eval_batch_size, repeat=False, shuffle=False)
    trainer.extend(evaluate_log_likelihood(train_eval_iterator, model, gpu, n_eval_epochs, n_eval_samples),
                   name='train', trigger=eval_interval)

    valid_eval_iterator = SerialIterator(valid, eval_batch_size, repeat=False, shuffle=False)
    trainer.extend(evaluate_log_likelihood(valid_eval_iterator, model, gpu, n_eval_epochs, n_eval_samples),
                   trigger=eval_interval)

    grad_eval_iterator = SerialIterator(train, train_batch_size)
    trainer.extend(evaluate_gradient_variance(
        grad_eval_iterator, model.inference_net, estimator, gpu, config.get('grad_eval_iteration', 1000)),
        trigger=eval_interval)

    best_model = copy.deepcopy(model)
    keep_best_model = KeepBestModel(model, best_model, 'validation/mcb')
    trainer.extend(keep_best_model, trigger=eval_interval)

    report_training_time(trainer, eval_interval)

    snapshot_interval = config.get('snapshot_interval', 1000000), 'iteration'
    trainer.extend(snapshot(), trigger=snapshot_interval)
    trainer.extend(snapshot(filename='last_snapshot'), trigger=eval_interval, name='latest_state_snapshot')
    trainer.extend(snapshot_object(best_model, 'best_model_iter_{.updater.iteration}'), trigger=snapshot_interval)

    trainer.extend(LogReport(trigger=eval_interval))
    if verbose:
        trainer.extend(PrintReport(
            ['iteration', 'train/vb', 'train/mcb', 'validation/vb', 'validation/mcb', 'gradvar/mean',
             'elapsed_time', 'training_time']))
        trainer.extend(ProgressBar())

    if resume == 'auto':
        path = os.path.join(out_path, 'last_snapshot')
        resume = path if os.path.exists(path) else ''
    if resume:
        print('resume from {}'.format(resume))
        load_npz(resume, trainer)

    # Copy the config file to the output directory
    try:
        os.makedirs(out_path)
    except OSError:
        pass
    with open(os.path.join(out_path, 'config.yaml'), 'w') as config_out:
        config_out.write(config_raw)

    # Training
    if verbose:
        print('start training...')
    trainer.run()

    # Test
    if verbose:
        print('testing...')
    test_iter = SerialIterator(test, config.get('test_batch_size', 1), repeat=False, shuffle=False)
    evaluator = LogLikelihoodEvaluator(
        test_iter, best_model, gpu, config.get('n_test_epochs', 1), config.get('n_test_samples', 50000))
    vb, mcb = evaluator.evaluate()
    with open(os.path.join(out_path, 'test_result.json'), 'w') as f:
        json.dump({'iteration': keep_best_model.best_iteration, 'vb': float(vb), 'mcb': float(mcb)}, f)
    if verbose:
        print('...finished')

    return best_model


class NonlinearLayer(Chain):

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__(l1=L.Linear(n_in, n_in, initialW=GlorotUniform()),
                         l2=L.Linear(n_in, n_in, initialW=GlorotUniform()),
                         l3=L.Linear(n_in, n_out, initialW=GlorotUniform()))

    def __call__(self, x: Variable) -> Variable:
        h = F.tanh(self.l1(x))
        h = F.tanh(self.l2(h))
        return self.l3(h)


def _build_layers(config: dict) -> Tuple[Tuple[Link, ...], int]:
    typ = config.get('type', 'linear')
    units = config['units']
    if typ == 'linear':
        return tuple(L.Linear(n_in, n_out) for n_in, n_out in units), units[-1][1]
    elif typ == 'nonlinear':
        return tuple(NonlinearLayer(n_in, n_out) for n_in, n_out in units), units[-1][1]
    else:
        raise ValueError('unsupported layer type: "{}"'.format(typ))


def _build_estimator(config: dict, n_layers: int, model: VariationalModel) -> GradientEstimator:
    method = config['method']
    if method == 'likelihood_ratio':
        use_baseline_model = config.get('use_baseline_model', False)
        baseline_model = [BaselineModel() for _ in range(n_layers)] if use_baseline_model else None
        use_muprop = config.get('use_muprop', False)
        n_samples = config.get('n_samples', 1)
        return LikelihoodRatioEstimator(
            model, baseline_model, float(config.get('alpha', 0.8)), config.get('variance_normalization', False),
            use_muprop, n_samples)
    elif method == 'discrete_reparameterization':
        return DiscreteReparameterizationEstimator(model, config.get('n_samples', 1))
    elif method == 'local_expectation_gradient':
        return LocalExpectationGradientEstimator(model)
    else:
        raise ValueError('unknown estimator type: "{}"'.format(method))


def _build_optimizer(config: dict, target: Optional[Link]) -> Optional[Optimizer]:
    if target is None:
        return None
    method = config.get('method', 'sgd')
    if method == 'sgd':
        if 'momentum' in config:
            opt = optimizers.MomentumSGD(float(config['lr']), float(config['momentum']))
        else:
            opt = optimizers.SGD(float(config['lr']))
    elif method == 'rmsprop':
        opt = optimizers.RMSprop(float(config['lr']), float(config.get('alpha', 0.99)), float(config.get('eps', 1e-8)))
    elif method == 'adam':
        opt = optimizers.Adam(
            float(config['alpha']), float(config.get('beta1', 0.9)), float(config.get('beta2', 0.999)),
            float(config.get('eps', 1e-8)))
    else:
        raise ValueError('unknown optimizer method: "{}"'.format(method))
    opt.setup(target)

    decay = float(config.get('weight_decay', 0.))
    if decay > 0:
        opt.add_hook(WeightDecay(decay))

    return opt


_DATASET_LOADER = {
    'mnist': [
        get_offline_binary_mnist,
        get_online_binary_mnist
    ],
    'omniglot': [
        get_offline_binary_omniglot,
        get_online_binary_omniglot
    ],
}


def _get_dataset(name: str, online: bool, use_gpu: bool) -> Tuple[np.ndarray, Any, Any, Any]:
    if name not in _DATASET_LOADER:
        raise ValueError('dataset "{}" is not supported'.format(name))

    loader = _DATASET_LOADER[name][online]
    train, valid, test = loader()
    if online:
        mean = np.mean(train.base, axis=0, keepdims=True)
        if use_gpu:
            train.base = cuda.to_gpu(train.base)
            valid.base = cuda.to_gpu(valid.base)
            test.base = cuda.to_gpu(test.base)
    else:
        mean = np.mean(train, axis=0, keepdims=True)
        if use_gpu:
            train = cuda.to_gpu(train)
            valid = cuda.to_gpu(valid)
            test = cuda.to_gpu(test)

    return mean, train, valid, test
