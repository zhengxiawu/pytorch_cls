# pytorch_cls
A classification repo implemented with PyTorch on CIFAR-10 and ImageNet under different training conditions 

## Prerequisites
- Python 3.6+
- PyTorch 1.5.0-cu10.1

## Training Results

### PyTorch-cifar-basic:

Training different architectures ([PyTorch](http://pytorch.org/)) on the CIFAR10 dataset without any tricks i.e., auto-augmentation, cutout, droppath, dropout. The learning rate is adjusted by the consine learning schedular, start from 0.1 with 300 epochs.


| Model             | Acc.        | FLOPS        | param        | training time <br> (hours)|
| ----------------- | :---------: | :---------:  | :---------:  | :---------:               |
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

#### How to use
Training the models in basic train with:

`cd ./pytorch-cifar-basic; python main.py --model_name resnet18`



### PyTorch-cifar:

Code base: [darts](https://github.com/quark0/darts)

Torch version: 1.5.0+cu101

Training different architectures ([PyTorch](http://pytorch.org/)) on the CIFAR10 dataset with and without tricks i.e., cutout, droppath, dropout. The learning rate is adjusted by the consine learning schedular, start from 0.025, batch size 96 with 600 epochs.

| Model   | Acc. | FLOPS | param|training time <br> (hours)|Auxiliary Weight|Drop Path|Cutout|
| -       | :-:  | :--:  | :--: | :-----------------------:|:-------------: |:-------:|:----:|
|NASNet   |95.43%| 615M  | 3.83M| 30.94                    |FALSE           | 0.2     |FALSE |
|NASNet   |97.02%| 615M  | 3.83M| 31.43                    |0.4             | 0.2     |16    |
|AmoebaNet|95.71%| 499M  | 3.14M| 25.90                    |FALSE           | 0.2     |FALSE |
|AmoebaNet|97.04%| 499M  | 3.14M| 31.02                    |0.4             | 0.2     |16    |
|Darts_V1 |95.42%| 511M  | 3.16M| 25.96                    |FALSE           | 0.2     |FALSE |
|Darts_V1 |96.90%| 511M  | 3.16M| 32.06                    |0.4             | 0.2     |16    |
|Darts_V2 |95.42%| 539M  | 3.34M| 28.39                    |FALSE           | 0.2     |FALSE |
|Darts_V2 |97.04%| 539M  | 3.34M| 28.06                    |0.4             | 0.2     |16    |
|ResNet18 |95.60%| 539M  |11.17M| 4.14                     |FALSE           | 0.2     |FALSE |
|ResNet18 |96.33%| 556M  |11.17M| 4.4                      |0.4             | 0.2     |16    |
|ResNet34 |95.46%| 1161M |21.28M| 7.56                     |FALSE           | 0.2     |FALSE |
|ResNet34 |96.84%| 1161M |21.28M| 7.69                     |0.4             | 0.2     |16    |
|ResNet50 |96.62%| 1304M |23.52M| 11.51                    |0.4             | 0.2     |16    |
|ResNet101|95.97%| 2520M |42.51M| 19.7                     |FALSE           | 0.2     |FALSE |
|ResNet101|96.60%| 2520M |42.51M| 19.7                     |0.4             | 0.2     |16    |
|ResNet152|95.81%| 3736M |58.15M| 23.32                    |FALSE           | 0.2     |FALSE |
|ResNet152|96.87%| 3736M |58.15M| 23.32                    |0.4             | 0.2     |16    |

#### Result On different torch versions:

Model: NASNet

Training condition: 
``` python
Namespace(auxiliary=True, auxiliary_weight=0.4, batch_size=96, cutout=True, cutout_length=16, data='/gdata/cifar10', drop_path_prob=0.2, epochs=600, gpu=0, grad_clip=5, init_channels=36, layers=20, learning_rate=0.025, model_name='nasnet', momentum=0.9, report_freq=50, seed=0, weight_decay=0.0003)
```

| Version   | Acc. | training time <br> (hours)|
| -         | :-:  | :----------------------:  |
| 1.5.0     | 97.02| 31.43                     |
| 1.0.1     | 96.79| 32.7025                   |
|1.0.1.post2| 97.11| 40.8297                   |
| 1.1.0     | 96.86| 42.3611                   |
| 1.2.0     | 96.79| 40.7019                   |
| 1.3.0     | 96.86| 34.0188                   |
| 1.3.1     | 96.69| 41.6177                   |
| 1.4.0     | 97.02| 34.2972                   |


### PyTorch_cifar_v2:

Code base: [pt.darts](https://github.com/khanrc/pt.darts)

Torch version: 1.5.0+cu101

Training different architectures ([PyTorch](http://pytorch.org/)) on the CIFAR10 dataset with and without tricks i.e., cutout, droppath, dropout. The learning rate is adjusted by the consine learning schedular, start from 0.025, batch size 96 with 600 epochs.

| Model   | Acc. | FLOPS | param|training time <br> (hours)|Auxiliary Weight|Drop Path|Cutout|
| -       | :-:  | :--:  | :--: | :-----------------------:|:-------------: |:-------:|:----:|
|NASNet   |      | 615M  | 3.83M| 30.94                    |FALSE           | 0.2     |FALSE |
|NASNet   |      | 615M  | 3.83M| 31.43                    |0.4             | 0.2     |16    |
|amoebaNet| 97.2700%| 3.159 M| 502.452 M|39.08416666666667|0.4             |0.2      |16    |
|amoebaNet| 95.4800%| 3.159 M| 502.452 M|29.686944444444446|0.0             |0.0      |0    |
|amoebaNet| 97.1000%| 3.159 M| 502.452 M|29.88388888888889|0.4             |0.0      |16    |
|amoebaNet| 97.2100%| 3.159 M| 502.452 M|30.70722222222222|0.0             |0.2      |16    |
|amoebaNet| 96.6600%| 3.159 M| 502.452 M|35.30833333333333|0.0             |0.0      |16    |
|Darts_V1 |95.60%| 511M  | 3.16M| 33.68                    |0               | 0       |0     |
|Darts_V1 |96.59%| 511M  | 3.16M| 33.50                    |0               | 0       |16    |
|Darts_V1 |      | 511M  | 3.16M| 25.96                    |0               | 0       |0     |
|Darts_V1 |      | 511M  | 3.16M| 25.96                    |0               | 0       |0     |
|Darts_V1 |      | 511M  | 3.16M| 32.06                    |0.4             | 0.2     |16    |
|Darts_V2 |      | 539M  | 3.34M| 28.39                    |FALSE           | 0.2     |FALSE |
|Darts_V2 |      | 539M  | 3.34M| 28.06                    |0.4             | 0.2     |16    |
|ResNet18 |95.68%| 556M  |11.17M| 5.5                      |0.0             | 0.0     |0     |
|ResNet18 |96.36%| 556M  |11.17M| 4.4                      |0.4             | 0.0     |16    |
|ResNet18 |96.71%| 556M  |11.17M| 4.5                      |0.0             | 0.2     |16    |
|ResNet18 |96.52%| 556M  |11.17M| 4.5                      |0.4             | 0.2     |16    |
|ResNet34 |95.69%| 1161M |21.28M| 7.67                     |0.0             | 0.0     |0     |
|ResNet34 |97.10%| 1161M |21.28M| 7.88                     |0.4             | 0.2     |16    |
|ResNet50 |96.85%| 1304M |23.52M| 12.24                    |0.4             | 0.2     |16    |
|ResNet101|      | 2520M |42.51M| 19.7                     |FALSE           | 0.2     |FALSE |
|ResNet101|97.13%| 2520M |42.51M| 19.7                     |0.4             | 0.2     |16    |
|ResNet152|      | 3736M |58.15M| 23.32                    |FALSE           | 0.2     |FALSE |
|ResNet152|      | 3736M |58.15M| 23.32                    |0.4             | 0.2     |16    |
|ResBasic |95.98%| 1481M |10.45M| 11.79                    |0               | 0       |0     |
|ResBasic |96.95%| 1481M |10.45M| 9.2                      |0               | 0       |16    |
|ResBasic |97.27%| 1481M |10.45M| 9.27                     |0.4             | 0       |16    |
|ResBasic |97.33%| 1481M |10.45M| 11.96                    |0               | 0.2     |16    |
|ResBasic |97.31%| 1481M |10.45M| 9.52                     |0.4             | 0.2     |16    |
|ResBottle|96.30%| 1542M |10.71M| 13.63                    |0               | 0       |0     |
|ResBottle|96.68%| 1542M |10.71M| 13.61                    |0               | 0       |16    |
|ResBottle|96.89%| 1542M |10.71M| 16.87                    |0.4             | 0       |16    |
|ResBottle|97.09%| 1542M |10.71M| 17.97                    |0               | 0.2     |16    |
|ResBottle|96.97%| 1542M |10.71M| 17.86                    |0.4             | 0.2     |16    |


## Reference

* [pt.darts](https://github.com/khanrc/pt.darts)
* [darts](https://github.com/quark0/darts)
* [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)