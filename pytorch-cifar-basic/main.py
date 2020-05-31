'''Train CIFAR10 with PyTorch.'''
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from thop import profile
from models import get_model
from utils import get_logger, AverageMeter, accuracy, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=300, type=int, help='training epoch')
parser.add_argument('--model_name', default='resnet18',
                    type=str, help='select the model')
parser.add_argument('--data_path', default='/gdata/cifar10',
                    type=str, help='select the model')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
args = parser.parse_args()

save_path = "./experiment/{}".format(args.model_name)
if os.path.isdir(save_path):
    pass
else:
    os.mkdir(save_path)
logger = get_logger(os.path.join(save_path, "logger.log"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
logger.info('torch version is: {0}'.format(torch.__version__))
logger.info('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# make the code more reproducable
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
random.seed(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

# Model
logger.info('==> Building model..')
net = get_model(args.model_name)
macs, params = profile(net, inputs=(torch.randn(1, 3, 32, 32), ))
macs, params = macs / 1000. / 1000., params / 1000. / 1000.
logger.info("The parameter size is: {0}".format((params)))
logger.info("The FLOPS is: {0}".format(macs))
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load(os.path.join(save_path, 'ckpt.pth'))
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, args.epoch)


# Training
def train(epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    net.train()
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        N = inputs.size(0)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
        if batch_idx % 10 == 0 or batch_idx == len(trainloader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, args.epoch, batch_idx, len(trainloader), losses=losses,
                    top1=top1, top5=top5))


def test(epoch):
    global best_acc
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            N = inputs.size(0)
            loss = criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            if batch_idx % 10 == 0 or batch_idx == len(testloader)-1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, args.epoch, batch_idx, len(testloader)-1, losses=losses,
                        top1=top1, top5=top5))
    # Save checkpoint.
    if best_acc < top1.avg:
        best_acc = top1.avg
        is_best = True
        logger.info("Current best Prec@1 = {:.4%}".format(best_acc))
    else:
        is_best = False
    save_checkpoint(net, save_path, is_best)


for epoch in range(0, args.epoch):
    train(epoch)
    lr_scheduler.step()
    test(epoch)
