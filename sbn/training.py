from contextlib import contextmanager
from typing import Any, Tuple

from chainer import cuda, Function, is_debug, set_debug
import numpy as np
import yaml

from sbn.datasets import get_offline_binary_mnist, get_online_binary_mnist
from sbn.variational_model import VariationalModel


__all__ = ['train_variational_model']


def train_variational_model(config: str, gpu: int, debug: bool=False) -> VariationalModel
    """Trains a variational model.

    Args:
        config: YAML config string.
        gpu: GPU device ID to use (-1 for CPU).
        debug: If True, it runs in the debug mode.

    Returns:
        Learned variational model.

    """
    with _debug_mode(debug):
        with cuda.get_device(gpu):
            return _train_variational_model(config, gpu >= 0)


@contextmanager
def _debug_mode(debug: bool):
    original_debug = is_debug()
    set_debug(debug)

    original_type_check = Function.type_check_enable
    Function.type_check_enable = debug

    yield

    Function.type_check_enable = original_type_check
    set_debug(original_debug)


def _train_variational_model(config_raw: str, use_gpu: bool) -> VariationalModel:
    config = yaml.load(config_raw)

    mean, train, valid, test = _get_dataset(config['dataset'], config.get('binarize_online', True), use_gpu)

    pass  # TODO(beam2d): Write it!


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
