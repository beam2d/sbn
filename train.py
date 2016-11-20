#!/usr/bin/env python
from argparse import ArgumentParser

from sbn import train_variational_model


def main():
    parser = ArgumentParser(description='Train a directed generative model')
    parser.add_argument('config', help='Path to the config YAML file')
    parser.add_argument('--debug', '-D', action='store_true', help='Enable debug mode')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device ID (-1: CPU)')
    parser.add_argument('--resume', '-r', type=str, default='', help='Snapshot to resume training from')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose print mode')
    args = parser.parse_args()

    train_variational_model(args.config. args.gpu, args.resume, args.debug, args.verbose)


if __name__ == '__main__':
    main()
