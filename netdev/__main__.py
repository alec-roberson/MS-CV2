''' __main__.py
The main module for netdev package.
'''
import os
import argparse
from netdev.arch_test import arch_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # subparser argument
    subparsers = parser.add_subparsers(title='command', dest='command')
    # subparser for anchor finding
    F_parser = subparsers.add_parser('F', help='anchor finding')
    F_parser.add_argument('cfg_path',
        help='the path to the config (.cfg) file to test')
    F_parser.add_argument('--batch_size', dest='batch_size', default=32,
        type=int, help='the batch size (default: 32)')
    F_parser.add_argument('--stop_at', dest='stop_at', default=None,
        type=int, help='after which layer to stop the testing loop, if desired')
    # subparser for architecture testing
    T_parser = subparsers.add_parser('T', help='architecture testing')
    T_parser.add_argument('cfg_path',
        help='the relative path to the config (.cfg) file to test')
    T_parser.add_argument('--batch_size', dest='batch_size', default=32,
        type=int, help='the batch size (default: 32)')
    T_parser.add_argument('--stop_at', dest='stop_at', default=None,
        type=int, help='after which layer to stop the testing loop, if desired')
    # parse arguments
    args = parser.parse_args()

    # run the command
    if args.command == 'F':
        pass
        
    elif args.command == 'T':
        # architecture testing
        arch_test(args.cfg_path, args.batch_size, args.stop_at)
