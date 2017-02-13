#!/usr/bin/env python
from argparse import ArgumentParser
import json

import plot


def main() -> None:
    parser = ArgumentParser(description='Print the best parameters')
    parser.add_argument('method', help='method name')
    parser.add_argument('dataset', choices=('mnist', 'omniglot'), help='dataset name')
    parser.add_argument('model', choices=('linear', 'nonlinear'), help='model type')
    parser.add_argument('--n_samples', '-K', choices=(1, 400), type=int, default=1, help='number of MC samples')
    parser.add_argument('--root', '-R', default='result', help='result root directory')
    args = parser.parse_args()

    def _cond(x):
        return (x['method'] == args.method and
                x['dataset'] == args.dataset and
                x['model'] == args.model and
                x['n_samples'] == str(args.n_samples))

    _, exp = plot._get_best_log(args.root, _cond)
    print(json.dumps(exp, indent=4))


if __name__ == '__main__':
    main()
