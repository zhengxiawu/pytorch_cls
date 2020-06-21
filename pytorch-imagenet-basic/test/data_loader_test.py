import logging
import time

import numpy as np
import torch
import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import pycls.core.config as config
from pycls.core.config import cfg
from pycls.datasets.imagenet import ImageNet, ImageNet_Dataset

log_file_name = "./experiments/dataloader_timetest/{}.log".format(
    str(time.time()))
logging.basicConfig(filename=log_file_name, level=logging.DEBUG)


def time_test(loader):
    overall_batch = []
    pre_tic_time = time.time()
    for cur_iter, (inputs, labels) in enumerate(loader):
        if cur_iter > 1000:
            break
        tic_time = time.time()
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        batch_time = tic_time - pre_tic_time
        pre_tic_time = tic_time
        # print(batch_time)
        # print(batch_time)
        overall_batch.append(batch_time)
        # print('one batch time is:{}'.format(str(batch_time)))
    overall_batch = np.array(overall_batch)
    return np.mean(overall_batch), np.std(overall_batch)


def output_in_file(_str, filename):
    with open(filename, "w") as f:
        f.write(_str + '\n')
    f.close()


if __name__ == "__main__":
    # num_workers = 4
    # print('data_loader test')
    # print('using default dataloader')
    data_path = '/gdata/ImageNet2012'
    # logging.info('test')
    # dataset = ImageNet_Dataset(data_path,
    #                            batch_size=256,
    #                            size=224,
    #                            val_batch_size=200,
    #                            val_size=256,
    #                            min_crop_size=0.08,
    #                            workers=12,
    #                            world_size=1,
    #                            cuda=True,
    #                            use_dali=True,
    #                            dali_cpu=False)
    # train_loader = dataset.train_loader
    # val_loader = dataset.val_loader
    # for i in range(10):
    #     print('epoch:{}'.format(str(i)))
    #     for cur_iter, (inputs, labels) in enumerate(train_loader):
    #         inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
    #         if cur_iter > 10:
    #             break
    #         print('train:'+str(cur_iter))
    #     dataset.prep_for_val()
    #     for cur_iter, (inputs, labels) in enumerate(val_loader):
    #         inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
    #         if cur_iter > 10:
    #             break
    #         print('val:'+str(cur_iter))
    #     dataset.reset()
    for num_workers in range(56, 100, 4):
        logging.info('using DALI GPU dataloader, Finetune workers')
        dataset = ImageNet_Dataset(data_path,
                                   batch_size=256,
                                   size=224,
                                   val_batch_size=200,
                                   val_size=256,
                                   min_crop_size=0.08,
                                   workers=num_workers,
                                   world_size=1,
                                   cuda=True,
                                   use_dali=True,
                                   dali_cpu=False)
        loader = dataset.train_loader
        mean, std = time_test(loader)
        logging.info('mean time is:{}'.format(mean))
        logging.info('time std is:{}'.format(std))

        logging.info('using DALI CPU dataloader')
        dataset = ImageNet_Dataset(data_path,
                                   batch_size=256,
                                   size=224,
                                   val_batch_size=200,
                                   val_size=256,
                                   min_crop_size=0.08,
                                   workers=num_workers,
                                   world_size=1,
                                   cuda=True,
                                   use_dali=True,
                                   dali_cpu=True)
        loader = dataset.train_loader
        mean, std = time_test(loader)
        logging.info('mean time is:{}'.format(mean))
        logging.info('time std is:{}'.format(std))

        logging.info('using Torch CPU dataloader')
        dataset = ImageNet_Dataset(data_path,
                                   batch_size=256,
                                   size=224,
                                   val_batch_size=200,
                                   val_size=256,
                                   min_crop_size=0.08,
                                   workers=num_workers,
                                   world_size=1,
                                   cuda=True,
                                   use_dali=False,
                                   dali_cpu=True)
        loader = dataset.train_loader
        mean, std = time_test(loader)
        logging.info('mean time is:{}'.format(mean))
        logging.info('time std is:{}'.format(std))
