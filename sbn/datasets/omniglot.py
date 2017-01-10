import os
from typing import Dict, Tuple

from chainer import dataset
import numpy as np
from scipy.io import loadmat

from sbn.datasets.online_binary_arrays import OnlineBinaryArrays


def get_omniglot() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downloads and retrieves the 28x28 Omniglot dataset for generative learning.

    This function returns the Omniglot dataset resized to 28x28 images used in [1]. The training dataset of the
    original source is split into 20288 training images and 4057 validation images.

    Reference:
        [1]: Y. Burda, R. Grosse and R. Salakhutdinov. Importance Weighted Autoencoders. ICLR, 2015.

    Returns:
        Tuple of training, validation, and test matrices. These matrices are of size (20288, 784), (4057, 784), and
        (8070, 784), respectively.

    """
    root = dataset.get_dataset_directory('beam2d/omniglot')
    path = os.path.join(root, 'chardata.npz')
    d = dataset.cache_or_load_file(path, _download, np.load)
    train, val = _random_split(d['train'], 20288)
    return train, val, d['test']


def _download(path: str) -> Dict[str, np.ndarray]:
    url = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'
    matpath = dataset.cached_download(url)
    mat = loadmat(matpath)
    d = {'train': mat['data'].T.astype(np.float32), 'test': mat['testdata'].T.astype(np.float32)}
    np.savez_compressed(path, **d)
    return d


def _random_split(array: np.ndarray, split_at: int, seed: int=1091) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(seed)
    rs.shuffle(array)
    return array[:split_at], array[split_at:]


def get_offline_binary_omniglot() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtains the binarized version of the 28x28 Omniglot dataset.

    Returns:
        Tuple of training, validation, and test matrices. See ``get_omniglot`` for the dataset shape. Each element is
        binarized to {0, 1} value with probability of the pixel intensity.

    """
    train, val, test = get_omniglot()
    return OnlineBinaryArrays(train)[...], OnlineBinaryArrays(val)[...], OnlineBinaryArrays(test)[...]


def get_online_binary_omniglot(seed: int=1091) -> Tuple[OnlineBinaryArrays, OnlineBinaryArrays, OnlineBinaryArrays]:
    """Obtains the 28x28 Omniglot dataset binarized in an online manner.

    This function returns the binarized 28x28 Omniglot dataset. Unlike the case of ``get_offline_binary_omniglot``, the
    binarization occurs in access to each example.

    Returns:
        Tuple of training, validation, and test matrices. See ``get_omniglot`` for the dataset shape.

    """
    train, val, test = get_omniglot()
    return OnlineBinaryArrays(train), OnlineBinaryArrays(val), OnlineBinaryArrays(test)
