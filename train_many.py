#!/usr/bin/env python
from argparse import ArgumentParser
from itertools import product
import multiprocessing as mp
from queue import Empty, Queue
from string import Template
from threading import Thread
from typing import Any, Dict, List, Tuple

import yaml

from sbn import train_variational_model


def main():
    parser = ArgumentParser(description='Search hyperparameter')
    parser.add_argument('config', help='Path to the variable config YAML')
    parser.add_argument('--gpu', '-g', default='0', help='GPU devices (comma separated)')
    args = parser.parse_args()

    mp.set_start_method('forkserver')

    devices = [int(s) for s in args.gpu.split(',')]

    template_path, settings = _read_var_config(args.config)

    setting_queue = Queue()
    for setting in settings:
        setting_queue.put(setting)

    threads = [Thread(target=_run_experiments_on_a_gpu, args=(template_path, device, setting_queue))
               for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _read_var_config(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    with open(path) as f:
        setting_all = yaml.load(f.read())
    template_path = setting_all.pop('template')
    settings = _dict_product(setting_all)
    return template_path, settings


def _dict_product(d: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    # borrowed from maf: https://github.com/pfi/maf/blob/master/maflib/util.py
    keys = sorted(d)
    values = [d[key] for key in keys]
    values_product = product(*values)
    return [dict(zip(keys, vals)) for vals in values_product]


def _run_experiments_on_a_gpu(template_path: str, device: int, setting_queue: Queue) -> None:
    with open(template_path) as f:
        template_str = f.read()
    template = Template(template_str)

    while True:
        try:
            setting = setting_queue.get()
        except Empty:
            print('device {}: quit thread'.format(device))
            break

        setting_str = {k: str(v) for k, v in setting.items()}
        config = template.substitute(setting_str)

        print('device {}: executing {}'.format(device, setting))
        process = mp.Process(target=_run_experiment, args=(config, device))
        process.daemon = True
        process.start()
        process.join()
        print('device {}: finished'.format(device))


def _run_experiment(config: str, device: int):
    train_variational_model(config, device, resume='auto', verbose=False)


if __name__ == '__main__':
    main()
