import argparse


# import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--num-stacks',
        type=int,
        default=4)
    parser.add_argument(
        '--num-steps',
        type=int,
        default= 100)# 100
    parser.add_argument(
        '--test-steps',
        type=int,
        default= 500)
    parser.add_argument(
        '--num-frames',
        type=int,
        default= 100000)# 100000
    parser.add_argument(
        '--info',
        type=str,
        default='')

    ## other parameter
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,# 1
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-img',
        type=bool,
        default=True)
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one eval per n updates (default: None)')

    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--h', type=int, default=2)
    parser.add_argument('--sp', type=int, default=10)
    parser.add_argument('--model_type', type=str, default="network", help='model type: "dyna", "network", "optnetwork"')

    args = parser.parse_args()

    return args
