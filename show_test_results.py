#!/usr/bin/env python
from argparse import ArgumentParser
import json
import os
from typing import Any, Dict

from show_best_param import get_best_param


METHODS = ['plainlr', 'stdlr', 'lr', 'muprop', 'leg', 'dr']
METHOD_NAMES = {
    'plainlr': 'LR',
    'stdlr': 'LR+C',
    'lr': 'LR+C+IDB',
    'muprop': 'MuProp+C+IDB',
    'leg': 'LEG',
    'dr': 'RAM',
}


def main() -> None:
    parser = ArgumentParser(description='Print test results of the best configuration of each method')
    parser.add_argument('dataset', choices=('mnist', 'omniglot'), help='dataset name')
    parser.add_argument('model', choices=('linear', 'nonlinear', 'deep'), help='model name')
    parser.add_argument('--n_samples', '-K', choices=(1, 400), type=int, default=1, help='number of MC samples')
    parser.add_argument('--root', '-R', default='result', help='result root directory')
    args = parser.parse_args()

    def _get_best_param(method):
        return get_best_param(method, args.dataset, args.model, args.n_samples, args.root)

    params = [_get_test_result(args.root, _get_best_param(method)) for method in METHODS]

    print('Method', *[METHOD_NAMES[method] for method in METHODS], sep=' & ')
    print('VB', *[param['vb'] for param in params], sep=' & ')
    print('MCB', *[param['mcb'] for param in params], sep=' & ')


def _get_test_result(root: str, param: Dict[str, str]) -> Dict[str, Any]:
    with open(os.path.join(root, param['name'], 'test_result.json')) as f:
        result = json.load(f)
    return result


if __name__ == '__main__':
    main()
