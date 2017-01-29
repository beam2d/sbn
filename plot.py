#!/usr/bin/env python
from argparse import ArgumentParser
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import seaborn
from typing import Any, Callable, Dict, List, Tuple


Log = List[Dict[str, Any]]


def main() -> None:
    parser = ArgumentParser(description='Plot results')
    parser.add_argument('dataset', choices=('mnist', 'omniglot'), help='dataset name')
    parser.add_argument('model', choices=('linear', 'nonlinear'), help='model type')
    parser.add_argument('n_samples', choices=(1, 400), type=int, help='number of MC samples')
    parser.add_argument('--root', '-R', default='result', help='result root directory')
    parser.add_argument('--outprefix', '-o', default='', help='output prefix')
    parser.add_argument('--outsuffix', '-s', default='.pdf', help='output suffix')
    args = parser.parse_args()

    out_name = '{}{}-{}-K={}{}'.format(args.outprefix, args.dataset, args.model, args.n_samples, args.outsuffix)
    plot(args.root, args.dataset, args.model, args.n_samples, out_name)


def plot(root: str, dataset: str, model: str, n_samples: int, out: str) -> None:
    def _cond(exp):
        return exp['dataset'] == dataset and exp['model'] == model and exp['n_samples'] == str(n_samples)

    log_plainlr, exp_plainlr = _get_best_log(root, lambda exp: _cond(exp) and exp['method'] == 'plainlr')
    log_stdlr, exp_stdlr = _get_best_log(root, lambda exp: _cond(exp) and exp['method'] == 'stdlr')
    log_lr, exp_lr = _get_best_log(root, lambda exp: _cond(exp) and exp['method'] == 'lr')
    log_dr, exp_dr = _get_best_log(root, lambda exp: _cond(exp) and exp['method'] == 'dr')

    plainlr_title = 'LR'
    stdlr_title = 'LR+B'
    lr_title = 'LR+B+IDB'
    dr_title = 'ours'

    figure = plt.figure()
    # figure.suptitle('{} {} n_samples={}'.format(dataset, model, n_samples))

    axes = figure.add_subplot(211)
    axes.set_ylabel('Variational lower bound')
    axes.set_xticklabels([])
    # axes.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100000))
    axes.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    # axes.grid(which='major', color='#999999', linestyle='-')
    # axes.grid(which='minor', color='#999999', linestyle=':')

    def _plot(ax, log, ykey, color, marker, linestyle, label=None):
        xs = [e['iteration'] for e in log]
        ys = [e[ykey] for e in log]
        ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle,
                mec=color, mew=1, mfc='None', markevery=10, label=label)

    color_plainlr = '#ff0000'
    color_stdlr = '#999900'
    color_lr = '#66aa00'
    color_dr = '#0000ff'
    mark_plainlr = '<'
    mark_stdlr = '>'
    mark_lr = 'v'
    mark_dr = 'o'

    _plot(axes, log_plainlr, 'train/vb', color_plainlr, '', 'dotted')
    _plot(axes, log_stdlr, 'train/vb', color_stdlr, '', 'dotted')
    _plot(axes, log_lr, 'train/vb', color_lr, '', 'dotted')
    _plot(axes, log_dr, 'train/vb', color_dr, '', 'dotted')
    _plot(axes, log_plainlr, 'validation/vb', color_plainlr, mark_plainlr, 'solid', plainlr_title)
    _plot(axes, log_stdlr, 'validation/vb', color_stdlr, mark_stdlr, 'solid', stdlr_title)
    _plot(axes, log_lr, 'validation/vb', color_lr, mark_lr, 'solid', lr_title)
    _plot(axes, log_dr, 'validation/vb', color_dr, mark_dr, 'solid', dr_title)

    axes.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.)

    axes = figure.add_subplot(212)
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Gradient variance')
    # axes.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100000))
    # axes.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    axes.set_yscale('log')
    # axes.grid(which='major', color='#999999', linestyle='-')
    # axes.grid(which='minor', color='#999999', linestyle=':')

    _plot(axes, log_plainlr, 'gradvar/mean', color_plainlr, mark_plainlr, 'solid', plainlr_title)
    _plot(axes, log_stdlr, 'gradvar/mean', color_stdlr, mark_stdlr, 'solid', stdlr_title)
    _plot(axes, log_lr, 'gradvar/mean', color_lr, mark_lr, 'solid', lr_title)
    _plot(axes, log_dr, 'gradvar/mean', color_dr, mark_dr, 'solid', dr_title)

    figure.savefig(out)


def _get_best_log(root: str, cond: Callable[[Dict[str, str]], bool]) -> Tuple[Log, str]:
    exps_all = [_parse_log_name(fn) for fn in os.listdir(root)]
    exps = [t for t in exps_all if cond(t)]
    best_vb = float('-inf')
    best_exp = None
    best_log = None
    for exp in exps:
        log = _get_log(os.path.join(root, exp['name'], 'log'))
        vb = max(entry['validation/vb'] for entry in log)
        if vb > best_vb:
            best_vb = vb
            best_exp = exp
            best_log = log
    return best_log, best_exp


def _parse_log_name(name: str) -> Dict[str, str]:
    # hyperparameters may include a small float of the format "Xe-XX"
    orig_name = name
    name = re.sub(r'([0-9]+e-?[0-9]+)', lambda s: '{:.10f}'.format(float(s.expand(r'\1'))).rstrip('0'), name)
    body, info = name.split('#')
    dataset, model, method = body.split('-')
    attrs = [entry.split('=') for entry in info.split('-')]
    d = dict(attrs)
    d['dataset'] = dataset
    d['model'] = model
    d['method'] = method
    d['name'] = orig_name
    return d


def _get_log(path: str) -> Log:
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    main()
