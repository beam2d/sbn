#!/usr/bin/env python
from argparse import ArgumentParser
import json
from typing import Dict

from plot import _get_best_log


def get_best_param(method: str, dataset: str, model: str, n_samples: int=1, root: str='result') -> Dict[str, str]:
    def _cond(x):
        return (x['method'] == method and
                x['dataset'] == dataset and
                x['model'] == model and
                int(x['n_samples']) == n_samples)

    _, param = _get_best_log(root, _cond)
    return param


def main() -> None:
    parser = ArgumentParser(description='Print the best parameters')
    parser.add_argument('method', help='method name')
    parser.add_argument('dataset', choices=('mnist', 'omniglot'), help='dataset name')
    parser.add_argument('model', choices=('linear', 'deep', 'nonlinear'), help='model type')
    parser.add_argument('--n_samples', '-K', choices=(1, 400), type=int, default=1, help='number of MC samples')
    parser.add_argument('--root', '-R', default='result', help='result root directory')
    args = parser.parse_args()

    param = get_best_param(args.method, args.dataset, args.model, args.n_samples, args.root)
    for key in sorted(param):
        print('{}:\t{}'.format(key, param[key]))


if __name__ == '__main__':
    main()
