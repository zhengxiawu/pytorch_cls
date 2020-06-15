import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pycls.core.net as net
from pycls.core.config import cfg
from pycls.models.effnet import Swish, SE


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class MBV2Head(nn.Module):
    """MobileNetV2 head: 1x1, BN, Relu, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, nc):
        super(MBV2Head, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 1, stride=1, padding=0, bias=False)
        self.conv_bn = nn.BatchNorm2d(
            w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv_swish = nn.ReLU6(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if cfg.EN.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.EN.DROPOUT_RATIO)
        self.fc = nn.Linear(w_out, nc, bias=True)

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if hasattr(self, "dropout") else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, nc):
        cx = net.complexity_conv2d(cx, w_in, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_out, nc, 1, 1, 0, bias=True)
        return cx


class MBV3Head(nn.Module):
    """MobileNetV3 head: 1x1, BN, Swish, AvgPool, FC, Dropout, FC."""

    def __init__(self, w_in, w_mid, w_out, nc):
        super(MBV3Head, self).__init__()
        self.conv = nn.Conv2d(w_in, w_mid, 1, stride=1, padding=0, bias=False)
        self.conv_bn = nn.BatchNorm2d(
            w_mid, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv_swish = Swish()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if cfg.EN.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.EN.DROPOUT_RATIO)
        self.fc1 = nn.Linear(w_mid, w_out, bias=True)
        self.fc2 = nn.Linear(w_out, nc, bias=True)

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if hasattr(self, "dropout") else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, nc):
        cx = net.complexity_conv2d(cx, w_in, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_out, nc, 1, 1, 0, bias=True)
        return cx


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out):
        # expansion, 3x3 dwise, BN, Swish, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = nn.Conv2d(w_in, w_exp, 1, stride=1,
                                 padding=0, bias=False)
            self.exp_bn = nn.BatchNorm2d(
                w_exp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
            self.exp_swish = Swish()
        dwise_args = {"groups": w_exp, "padding": (
            kernel - 1) // 2, "bias": False}
        self.dwise = nn.Conv2d(w_exp, w_exp, kernel,
                               stride=stride, **dwise_args)
        self.dwise_bn = nn.BatchNorm2d(
            w_exp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.dwise_swish = Swish()
        if se_r > 0:
            self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = nn.Conv2d(
            w_exp, w_out, 1, stride=1, padding=0, bias=False)
        self.lin_proj_bn = nn.BatchNorm2d(
            w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = x
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        if hasattr(self, 'se'):
            f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and cfg.EN.DC_RATIO > 0.0:
                f_x = net.drop_connect(f_x, cfg.EN.DC_RATIO)
            f_x = x + f_x
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_r, kernel, stride, se_r, w_out):
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            cx = net.complexity_conv2d(cx, w_in, w_exp, 1, 1, 0)
            cx = net.complexity_batchnorm2d(cx, w_exp)
        padding = (kernel - 1) // 2
        cx = net.complexity_conv2d(
            cx, w_exp, w_exp, kernel, stride, padding, w_exp)
        cx = net.complexity_batchnorm2d(cx, w_exp)
        cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = net.complexity_conv2d(cx, w_exp, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx
