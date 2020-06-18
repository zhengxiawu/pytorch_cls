import torch
import pycls.core.config as config
import time
from pycls.core.config import cfg
from pycls.datasets.imagenet import ImageNet
from pycls.datasets.imagenet import ImageNet_Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import tqdm
import numpy as np


def time_test(loader):
    overall_batch = []
    for cur_iter, (inputs, labels) in enumerate(loader):
        if cur_iter > 200:
            break
        tic_time = time.time()
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        toc_time = time.time()
        batch_time = toc_time - tic_time
        overall_batch.append(batch_time)
        # print('one batch time is:{}'.format(str(batch_time)))
    overall_batch = np.array(overall_batch)
    return np.mean(overall_batch), np.std(overall_batch)


if __name__ == "__main__":
    num_workers = 8
    print('data_loader test')
    print('using default dataloader')
    data_path = '/gdata/ImageNet2012'
    dataset = ImageNet(data_path, 'train')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    mean, std = time_test(loader)
    print('mean time is:{}'.format(mean))
    print('time std is:{}'.format(std))
    print('using DALI GPU dataloader')
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
    tmean, std = time_test(loader)
    print('mean time is:{}'.format(mean))
    print('time std is:{}'.format(std))

    print('using DALI CPU dataloader')
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
    print('mean time is:{}'.format(mean))
    print('time std is:{}'.format(std))

    print('using Torch CPU dataloader')
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
    print('mean time is:{}'.format(mean))
    print('time std is:{}'.format(std))
