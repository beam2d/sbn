import os
from typing import Callable, Dict, Tuple

from chainer import dataset, datasets
import numpy as np

from sbn.datasets.online_binary_arrays import OnlineBinaryArrays


__all__ = ['get_offline_binary_mnist', 'get_online_binary_mnist']


def get_offline_binary_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtains the binarized MNIST dataset.

    This function returns the binarized MNIST dataset used in [1]. This is a fixed binarization of MNIST.

    Reference:
        [1]: H. Larochelle and I. Murray. The neural autoregressive distribution estimator. AISTATS, 2011.

    Returns:
        Tuple of training, validation, and test matrices. The training matrix is of size (50000, 784) and the
        validation and test matrices are of size (10000, 784).

    """
    return _get('train'), _get('valid'), _get('test')


def _get(name: str) -> np.ndarray:
    root = dataset.get_dataset_directory('beam2d/binary_mnist')
    path = os.path.join(root, name + '.npz')
    x = dataset.cache_or_load_file(path, _download(name), np.load)['x']
    return x.astype(np.float32)


def _download(name: str) -> Callable[[str], Dict[str, np.ndarray]]:
    def download(path: str) -> Dict[str, np.ndarray]:
        url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(
            name
        )
        x_path = dataset.cached_download(url)
        x = np.loadtxt(x_path).astype(np.uint8)
        np.savez_compressed(path, x=x)
        return {'x': x}
    return download


def get_online_binary_mnist(seed: int=1091) -> Tuple[OnlineBinaryArrays, OnlineBinaryArrays, OnlineBinaryArrays]:
    """Obtains the MNIST dataset binarized in online manner.

    This function returns the binarized MNIST dataset. Unlike the case of ``get_offline_binary_mnist``, the
    binarization occurs in access to each example.

    Args:
        seed: Seed of the random partition of the training dataset.

    Returns:
        Tuple of training, validation, and test matrices. The training matrix is of size (50000, 784) and the
        validation and test matrices are of size (10000, 784).

    """
    train, test = datasets.get_mnist(withlabel=False)
    rs = np.random.RandomState(seed)
    rs.shuffle(train)
    train = OnlineBinaryArrays(train[:50000])
    valid = OnlineBinaryArrays(train[50000:])
    test = OnlineBinaryArrays(test)
    return train, valid, test
