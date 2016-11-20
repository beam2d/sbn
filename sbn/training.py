from contextlib import contextmanager
import copy
import json
import os
from typing import Any, Optional, Tuple

from chainer import Chain, Function, is_debug, Link, Optimizer, set_debug, Variable
from chainer import cuda, functions as F, links as L, optimizers
from chainer.iterators import SerialIterator
from chainer.serializers import load_npz
from chainer.training import Trainer
from chainer.training.extensions import LogReport, PrintReport, ProgressBar, snapshot, snapshot_object
import numpy as np
import yaml

from sbn.datasets import get_offline_binary_mnist, get_online_binary_mnist
from sbn.estimators import LikelihoodRatioEstimator
from sbn.extensions import evaluate_gradient_variance, evaluate_log_likelihood, KeepBestModel, report_training_time
from sbn.extensions import LogLikelihoodEvaluator
from sbn.grad_estimator import GradientEstimator
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
) -> VariationalModel
    """Trains a variational model.

    Args:
        config: YAML config string.
        gpu: GPU device ID to use (-1 for CPU).
        resume: Path to the trainer snapshot to resume from.
        debug: If True, it runs in the debug mode.
        verbose: If True, it prints messages.

    Returns:
        Learned variational model.

    """
    with _debug_mode(debug):
        with cuda.get_device(gpu):
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

    mean, train, valid, test = _get_dataset(config['dataset'], config.get('binarize_online', True), use_gpu)
    infer_layers, prior_size = _build_layers(config['inference_net'])
    gen_layers, _ = _build_layers(config['generative_net'])

    model = VariationalSBN(gen_layers, infer_layers, prior_size, mean)
    if use_gpu:
        model.to_gpu()
    estimator = _build_estimator(config['estimator'], len(infer_layers), model)

    gen_optimizer = _build_optimizer(config['generative_net']['optimizer'], gen_layers)
    infer_optimizer = _build_optimizer(config['inference_net']['optimizer'], infer_layers)

    estimator_model = estimator.get_estimator_model()
    est_optimizer = _build_optimizer(config['estimator']['optimizer'], estimator_model)

    train_batch_size = config.get('batch_size', 100)
    train_iterator = SerialIterator(train, train_batch_size)
    updater = Updater(estimator, train_iterator, gen_optimizer, infer_optimizer, est_optimizer)
    trainer = Trainer(updater, (config['iteration'], 'iteration'), out_path)

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
    trainer.extend(keep_best_model)

    report_training_time(trainer)

    snapshot_interval = config.get('snapshot_interval', 100000), 'iteration'
    trainer.extend(snapshot(), trigger=snapshot_interval)
    trainer.extend(snapshot_object(best_model, 'best_model_iter_{.updater.iteration}'), trigger=snapshot_interval)

    trainer.extend(LogReport(trigger=eval_interval))
    if verbose:
        trainer.extend(PrintReport(
            ['iteration', 'train/vb', 'train/mcb', 'validation/vb', 'validation/mcb', 'gradvar/mean',
             'elapsed_time', 'training_time']))
        trainer.extend(ProgressBar())

    if resume:
        load_npz(resume, trainer)
    if verbose:
        print('start training...')
    trainer.run()

    if verbose:
        print('testing...')
    test_iter = SerialIterator(test, config.get('test_batch_size', 1), repeat=False, shuffle=False)
    evaluator = LogLikelihoodEvaluator(
        test_iter, best_model, gpu, config.get('n_test_epochs', 1), config.get('n_test_samples', 50000))
    vb, mcb = evaluator.evaluate()
    with open(os.path.join(out_path, 'test_result.json')) as f:
        json.dump({'iteration': keep_best_model.best_iteration, 'vb': float(vb), 'mcb': float(mcb)}, f)
    if verbose:
        print('...finished')


def _build_layers(config: dict) -> Tuple[Tuple[Link, ...], int]:
    typ = config.get('type', 'linear')
    if typ == 'linear':
        units = config['units']
        return tuple(L.Linear(None, unit) for unit in units), units[-1]
    else:
        raise ValueError('unsupported layer type: "{}"'.format(typ))


class _Baseline(Chain):

    def __init__(self, n_units=200):
        super().__init__(l1=L.Linear(None, n_units), l2=L.Linear(n_units, 1))

    def __call__(self, x: Variable) -> Variable:
        B = len(x.data)
        h = F.tanh(self.l1(x))
        return F.reshape(self.l2(h), (B,))


def _build_estimator(config: dict, n_layers: int, model: VariationalModel) -> GradientEstimator:
    method = config['method']
    if method == 'likelihood_ratio':
        use_baseline_model = config.get('baseline_model', False)
        baseline_model = [_Baseline() for _ in range(n_layers)] if use_baseline_model else None
        return LikelihoodRatioEstimator(
            model, baseline_model, config.get('alpha', 0.8), config.get('variance_normalization', False))
    else:
        raise ValueError('unknown estimator type: "{}"'.format(method))


def _build_optimizer(config: dict, target: Optional[Link]) -> Optional[Optimizer]:
    if target is None:
        return None
    method = config.get('method', 'sgd')
    if method == 'sgd':
        if 'momentum' in config:
            opt = optimizers.MomentumSGD(config['lr'], config['momentum'])
        else:
            opt = optimizers.SGD(config['lr'])
    elif method == 'rmsprop':
        opt = optimizers.RMSprop(config['lr'], config.get('alpha', 0.99), config.get('eps', 1e-8))
    elif method == 'adam':
        opt = optimizers.Adam(
            config['alpha'], config.get('beta1', 0.9), config.get('beta2', 0.999), config.get('eps', 1e-8))
    else:
        raise ValueError('unknown optimizer method: "{}"'.format(method))
    opt.setup(target)
    return opt


def _get_dataset(name: str, online: bool, use_gpu: bool) -> Tuple[np.ndarray, Any, Any, Any]:
    if name == 'mnist':
        if online:
            train, valid, test = get_online_binary_mnist()
            mean = np.mean(train.base, axis=0, keepdims=True)
            if use_gpu:
                train.base = cuda.to_gpu(train.base)
                valid.base = cuda.to_gpu(valid.base)
                test.base = cuda.to_gpu(test.base)
        else:
            train, valid, test = get_offline_binary_mnist()
            mean = np.mean(train, axis=0, keepdims=True)
            if use_gpu:
                train = cuda.to_gpu(train)
                valid = cuda.to_gpu(valid)
                test = cuda.to_gpu(test)
    else:
        raise ValueError('dataset "{}" is not supported'.format(name))

    return mean, train, valid, test
