""" Config class for search/augment """
import argparse
import os
from functools import partial
import torch
import time


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(
        name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--model_name', default='nasnet')
        parser.add_argument('--dataset', default='CIFAR10',
                            help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int,
                            default=96, help='batch size')
        parser.add_argument('--lr', type=float,
                            default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float,
                            default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float,
                            default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int,
                            default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600,
                            help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int,
                            default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int,
                            default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float,
                            default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int,
                            default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float,
                            default=0.2, help='drop path prob')
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = '/gdata/cifar10'
        _path = "{}_{}_{}".format(
            self.model_name, time.strftime("%Y%m%d-%H%M%S"), torch.__version__)
        self.path = os.path.join('./experiments', _path)
        self.gpus = parse_gpus(self.gpus)
