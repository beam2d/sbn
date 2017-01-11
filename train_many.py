#!/usr/bin/env python
from argparse import ArgumentParser
from queue import Empty, Queue
import multiprocessing as mp
from string import Template
from threading import Thread
from typing import Any, Dict

from sbn import train_variational_model


def main():
    parser = ArgumentParser(description='Search hyperparameter')
    parser.add_argument('config', help='Path to the config YAML template')
    parser.add_argument('--gpu', '-g', default='0', help='GPU devices (comma separated)')
    args = parser.parse_args()

    mp.set_start_method('forkserver')

    devices = [int(s) for s in args.gpu.split(',')]

    setting_queue = Queue()
    for setting in _SETTINGS:
        setting_queue.put(setting)

    threads = [Thread(target=_run_experiments_on_a_gpu, args=(args.config, device, setting_queue))
               for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


_SETTINGS = [
    {'batch_size': 100, 'decay': 0.001, 'lr': 1e-3, 'n_samples': 1},
    {'batch_size': 100, 'decay': 0.001, 'lr': 3e-3, 'n_samples': 1},
    {'batch_size': 100, 'decay': 0.001, 'lr': 1e-2, 'n_samples': 1},
    {'batch_size': 100, 'decay': 0.001, 'lr': 3e-2, 'n_samples': 1},
    {'batch_size': 100, 'decay': 0.001, 'lr': 1e-3, 'n_samples': 400},
    {'batch_size': 100, 'decay': 0.001, 'lr': 3e-3, 'n_samples': 400},
    {'batch_size': 100, 'decay': 0.001, 'lr': 1e-2, 'n_samples': 400},
    {'batch_size': 100, 'decay': 0.001, 'lr': 3e-2, 'n_samples': 400},
]


def _run_experiments_on_a_gpu(template_path: str, device: int, setting_queue: Queue) -> None:
    while True:
        try:
            setting = setting_queue.get()
        except Empty:
            print('device {}: quit thread'.format(device))
            break

        print('device {}: executing {}'.format(device, setting))
        process = mp.Process(target=_run_experiment, args=(template_path, device, setting))
        process.daemon = True
        process.start()
        process.join()
        print('device {}: finished'.format(device))


def _run_experiment(template_path: str, device: int, setting: Dict[str, Any]):
    with open(template_path) as f:
        template_str = f.read()
    template = Template(template_str)
    setting_str = {k: str(v) for k, v in setting.items()}
    config = template.substitute(setting_str)
    train_variational_model(config, device, resume='', verbose=False)


if __name__ == '__main__':
    main()
