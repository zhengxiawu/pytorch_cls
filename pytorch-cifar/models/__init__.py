from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .nas_models import *
from .mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small

model_dict = {'resnet18': ResNet18(),
              'resnet34': ResNet34(),
              'resnet50': ResNet50(),
              'resnet101': ResNet101(),
              'resnet152': ResNet152(),
              'desnenet': densenet_cifar(),
              'dpn26': DPN26(),
              'dpn92': DPN92(),
              'efficientnetb0': EfficientNetB0(),
              'googlenet': GoogLeNet(),
              'lenet': LeNet(),
              'mobilenet': MobileNet(),
              'mobilenetv2': MobileNetV2(),
              'mobilenetv3large': MobileNetV3_Large(),
              'mobilenetv3small': MobileNetV3_Small(),
              'nasnet': NASNet(),
              'amoebanet': AmoebaNet(),
              'darts_v1': DARTS_V1(),
              'darts_v2': DARTS_V2(),
              'pnasnet_a': PNASNetA(),
              'pnasnet_b': PNASNetB(),
              'preact_resnet18': PreActResNet18(),
              'preact_resnet34': PreActResNet34(),
              'preact_resnet50': PreActResNet50(),
              'preact_resnet101': PreActResNet101(),
              'preact_resnet152': PreActResNet152(),
              'regnetx_200mf': RegNetX_200MF(),
              'regnetx_400mf': RegNetX_400MF(),
              'regnety_400mf': RegNetY_400MF(),
              'resnext29_2x64d': ResNeXt29_2x64d(),
              'resnext29_4x64d': ResNeXt29_4x64d(),
              'resnext29_8x64d': ResNeXt29_8x64d(),
              'resnext29_32x4d': ResNeXt29_32x4d(),
              'senet18': SENet18(),
              # 'shufflenetg2': ShuffleNetG2(),
              # 'shufflenetg3': ShuffleNetG3(),
              # 'shufflenetv2': ShuffleNetV2(),
              'vgg11': VGG('VGG11'),
              'vgg13': VGG('VGG13'),
              'vgg16': VGG('VGG16'),
              'vgg19': VGG('VGG19'),

              }


def get_model(model_name):
    return model_dict[model_name]
