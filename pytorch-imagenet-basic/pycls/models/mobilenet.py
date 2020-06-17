import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pycls.core.net as net
from pycls.core.config import cfg


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def width_multiply(ws, factor, round_nearest=8):
    new_ws = []
    for i in ws:
        new_ws.append(_make_divisible(i * factor, round_nearest))
    return new_ws


def get_act(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'swish':
        return Hswish(inplace=True)
    elif name == 'relu6':
        return nn.ReLU6(inplace=True)
    else:
        raise NotImplementedError


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            Hsigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, w_se, 1, 1, 0, bias=True)
        cx = net.complexity_conv2d(cx, w_se, w_in, 1, 1, 0, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet: 3x3, BN, Swish."""

    def __init__(self, w_in, w_out, conv_act):
        super(StemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv_act = get_act(conv_act)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 2, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class MBHead(nn.Module):
    """MobileNetV2/V3 head: generate by input."""

    def __init__(self, w_in, head_channels, head_acts, nc):
        super(MBHead, self).__init__()
        assert len(head_channels) == len(head_channels)
        self.conv = nn.Conv2d(
            w_in, head_channels[0], 1, stride=1, padding=0, bias=False)
        self.conv_bn = nn.BatchNorm2d(
            head_channels[0], eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv_act = get_act(head_acts[0])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        _head_acts = head_acts[1:]
        _head_channels = head_channels[1:]
        self.linears = []
        pre_w = head_channels[0]
        for i, act in enumerate(_head_acts):
            self.linears.append(nn.Linear(pre_w, _head_channels[i]))
            # self.linears.append(nn.BatchNorm1d(_head_channels[i]))
            self.linears.append(get_act(act))
            pre_w = _head_channels[i]
        if len(self.linears) > 0:
            self.linears = nn.Sequential(*self.linears)
        if cfg.MB.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.MB.DROPOUT_RATIO)
        self.fc = nn.Linear(head_channels[-1], nc, bias=True)

    def forward(self, x):
        x = self.conv_act(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if len(self.linears) > 0:
            x = self.linears(x)
        x = self.dropout(x) if hasattr(self, "dropout") else x
        x = self.fc(x)
        return x

    @ staticmethod
    def complexity(cx, w_in, head_channels, nc):
        cx = net.complexity_conv2d(cx, w_in, head_channels[0], 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, head_channels[0])
        previous_channel = head_channels[0]
        cx["h"], cx["w"] = 1, 1
        for _channel in head_channels[1:]:
            cx = net.complexity_conv2d(cx, previous_channel, _channel, 1, 1, 0)
            # cx = net.complexity_batchnorm2d(cx, _channel)
            previous_channel = _channel
        cx = net.complexity_conv2d(
            cx, head_channels[-1], nc, 1, 1, 0, bias=True)
        return cx


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, dwise_act, se_r, w_out):
        # expansion, 3x3 dwise, BN, Swish, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        w_exp = _make_divisible(w_exp, 8)
        if w_exp != w_in:
            self.exp = nn.Conv2d(w_in, w_exp, 1, stride=1,
                                 padding=0, bias=False)
            self.exp_bn = nn.BatchNorm2d(
                w_exp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
            self.exp_act = get_act(dwise_act)
        dwise_args = {"groups": w_exp, "padding": (
            kernel - 1) // 2, "bias": False}
        self.dwise = nn.Conv2d(w_exp, w_exp, kernel,
                               stride=stride, **dwise_args)
        self.dwise_bn = nn.BatchNorm2d(
            w_exp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        assert dwise_act in ['relu', 'swish']
        self.dwise_act = get_act(dwise_act)
        if se_r > 0:
            se_c = _make_divisible(w_exp * se_r, 8)
            self.se = SE(w_exp, int(se_c))
        self.lin_proj = nn.Conv2d(
            w_exp, w_out, 1, stride=1, padding=0, bias=False)
        self.lin_proj_bn = nn.BatchNorm2d(
            w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = x
        if self.exp:
            f_x = self.exp_act(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_act(self.dwise_bn(self.dwise(f_x)))
        if hasattr(self, 'se'):
            f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and cfg.MB.DC_RATIO > 0.0:
                f_x = net.drop_connect(f_x, cfg.MB.DC_RATIO)
            f_x = x + f_x
        return f_x

    @ staticmethod
    def complexity(cx, w_in, exp_r, kernel, stride, se_r, w_out):
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            cx = net.complexity_conv2d(cx, w_in, w_exp, 1, 1, 0)
            cx = net.complexity_batchnorm2d(cx, w_exp)
        padding = (kernel - 1) // 2
        cx = net.complexity_conv2d(
            cx, w_exp, w_exp, kernel, stride, padding, w_exp)
        cx = net.complexity_batchnorm2d(cx, w_exp)
        if se_r > 0:
            cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = net.complexity_conv2d(cx, w_exp, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class MobileNet(nn.Module):
    """MobileNetV2/V3 model."""

    @ staticmethod
    def get_args():
        return {
            "stem_w": cfg.MB.STEM_W,
            "stem_act": cfg.MB.STEM_ACT,
            "width_mult": cfg.MB.WIDTH_MULT,
            "ws": cfg.MB.WIDTHS,
            "exp_rs": cfg.MB.EXP_RATIOS,
            "se_rs": cfg.MB.SE_RARIOS,
            "ss": cfg.MB.STRIDES,
            "ks": cfg.MB.KERNELS,
            "acts": cfg.MB.ACTS,
            "head_w": cfg.MB.HEAD_W,
            "head_acts": cfg.MB.HEAD_ACTS,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self):
        err_str = "Dataset {} is not supported"
        assert cfg.TRAIN.DATASET in [
            "imagenet"], err_str.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in [
            "imagenet"], err_str.format(cfg.TEST.DATASET)
        super(MobileNet, self).__init__()
        self._construct(**MobileNet.get_args())
        self.apply(net.init_weights)

    def _construct(self, stem_w, stem_act, width_mult, ws, exp_rs, se_rs, ss, ks, acts, head_w, head_acts, nc):
        ws = width_multiply(ws, width_mult)
        stage_params = list(zip(ws, exp_rs, se_rs, ss, ks, acts))
        self.stem = StemIN(3, stem_w, stem_act)
        prev_w = stem_w
        for i, (w, exp_r, se_r, stride, kernel, act) in enumerate(stage_params):
            name = "layer_{}".format(i + 1)
            self.add_module(name, MBConv(
                prev_w, exp_r, kernel, stride, act, se_r, w))
            prev_w = w
        self.head = MBHead(prev_w, head_w, head_acts, nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @ staticmethod
    def complexity(cx):
        """Computes model complexity. If you alter the model, make sure to update."""
        return MobileNet._complexity(cx, **MobileNet.get_args())

    @ staticmethod
    def _complexity(cx, stem_w, stem_act, width_mult, ws, exp_rs, se_rs, ss, ks, acts, head_w, head_acts, nc):
        ws = width_multiply(ws, width_mult)
        stage_params = list(zip(ws, exp_rs, se_rs, ss, ks))
        cx = StemIN.complexity(cx, 3, stem_w)
        prev_w = stem_w
        for w, exp_r, se_r, stride, kernel in stage_params:
            cx = MBConv.complexity(
                cx, prev_w, exp_r, kernel, stride, se_r, w)
            prev_w = w
        cx = MBHead.complexity(cx, prev_w, head_w, nc)
        return cx
