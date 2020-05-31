'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import logging
import math
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time


# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()


# def format_time(seconds):
#     days = int(seconds / 3600/24)
#     seconds = seconds - days*3600*24
#     hours = int(seconds / 3600)
#     seconds = seconds - hours*3600
#     minutes = int(seconds / 60)
#     seconds = seconds - minutes*60
#     secondsf = int(seconds)
#     seconds = seconds - secondsf
#     millis = int(seconds*1000)

#     f = ''
#     i = 1
#     if days > 0:
#         f += str(days) + 'D'
#         i += 1
#     if hours > 0 and i <= 2:
#         f += str(hours) + 'h'
#         i += 1
#     if minutes > 0 and i <= 2:
#         f += str(minutes) + 'm'
#         i += 1
#     if secondsf > 0 and i <= 2:
#         f += str(secondsf) + 's'
#         i += 1
#     if millis > 0 and i <= 2:
#         f += str(millis) + 'ms'
#         i += 1
#     if f == '':
#         f = '0ms'
#     return f


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('cifar10-training')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
