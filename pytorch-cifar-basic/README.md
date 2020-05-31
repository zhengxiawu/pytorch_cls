# Train CIFAR10 with PyTorch

Training different architectures ([PyTorch](http://pytorch.org/)) on the CIFAR10 dataset without any tricks i.e., auto-augmentation, cutout, droppath, dropout.

## Prerequisites
- Python 3.6+
- PyTorch 1.5+

## Accuracy
| Model             | Acc.        | FLOPS        | param        | training time (hours)|
| ----------------- | ----------- | -----------  | -----------  | -----------          |
| [Lenet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)|   77.56%    |0.65M | 0.06M | 0.63 |
| [googlenet](https://arxiv.org/pdf/1409.4842.pdf)      |   95.26%    |1529M | 6.16M | 6.16 |
| [Mobilenet](https://arxiv.org/pdf/1704.04861.pdf)     |   92.18%    |47M   | 3.21M | 0.85 |
| [MobilenetV2](https://arxiv.org/pdf/1801.04381.pdf)   |   93.81%    | 94M  | 2.296M| 1.95 |
|[MobilenetV3Large](https://arxiv.org/pdf/1905.02244.pdf)|   92.89%   | 79.4M| 2.688M| 1.76 |
|[MobilenetV3Small](https://arxiv.org/pdf/1905.02244.pdf)|   91.37%   | 18.5M| 1.241M| 1.08 |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 95.59%      | 556M |11.173M| 1.61 |
| [ResNet34](https://arxiv.org/abs/1512.03385)          | 95.32%      | 1161M|21.282M| 1.99 |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 95.74%      | 1304M|23.52M | 4.36 |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 95.43%      | 2520M|42.51M | 7.07 |
| [ResNet152](https://arxiv.org/abs/1512.03385)         | 95.91%      | 3736M|58.15M | 9.99 |
| [PreACtResNet18](https://arxiv.org/pdf/1603.05027.pdf)| 95.37%      | 556M |11.17M | 1.22 |
| [PreACtResNet34](https://arxiv.org/pdf/1603.05027.pdf)| 95.12%      | 1161M|21.27M | 1.96 |
| [PreACtResNet50](https://arxiv.org/pdf/1603.05027.pdf)| 95.95%      | 1303M|23.50M | 4.28 |
|[PreACtResNet101](https://arxiv.org/pdf/1603.05027.pdf)| 95.44%      | 2519M|42.50M | 6.98 |
|[PreACtResNet152](https://arxiv.org/pdf/1603.05027.pdf)| 95.76%      | 3735M|58.14M | 9.92 |
| [SENet18](https://arxiv.org/abs/1709.01507)           | 95.46%      | 556M |11.26M | 1.87 |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 95.19%      | 226M |2.32M  | 2.83 |
| [RegNetX_400MF](https://arxiv.org/abs/2003.13678)     | 94.12%      | 471M |4.77M  | 4.77 |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 95.51%      | 472M |5.71M  | 4.91 |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 95.49%      | 779M | 4.77M | 4.18 |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 95.41%      | 1416M| 9.12M | 4.39 |
| [ResNeXt29(4x64d)](https://arxiv.org/abs/1611.05431)  | 95.76%      | 4242M| 27.1M | 11.0 |
| [DenseNet121_Cifar](https://arxiv.org/abs/1608.06993) | 95.28%      | 128M | 1.0M  | 2.46 |
| [DPN26](https://arxiv.org/abs/1707.01629)             | 95.64%      | 670M | 11.5M | 5.69 |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.66%      |2053M | 34.2M |15.43 |
| [EfficientB0](https://arxiv.org/pdf/1905.11946.pdf)   | 93.24%      | 112M | 3.69M | 2.92 |
| [NASNet](https://arxiv.org/pdf/1905.11946.pdf)        | 95.18%      | 615M | 3.83M | 14.7 |
| [AmoebaNet](https://arxiv.org/abs/1802.01548)         | 95.38%      | 499M | 3.14M | 11.99|
| [Darts_V1](https://arxiv.org/abs/1806.09055)          | 95.05%      | 511M | 3.16M | 11.69|
| [Darts_V2](https://arxiv.org/abs/1806.09055)          | 94.97%      | 539M | 3.34M | 12.32|

## Learning rate adjustment
The learning rate is adjusted by the consine learning schedular.

Resume the training with `python main.py --lr=0.1 --model_name resnet18`
