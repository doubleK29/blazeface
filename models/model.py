import logging
import math
import sys
from itertools import product as product

# import pandas as pd
from math import sqrt as sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# # backbone
from models.module import StemBlock, ShuffleV2Block, Conv, Concat, Detect, C3
from utils.autoanchor import check_anchor_order
from utils.general import check_model
from utils.torch_utils import (
    fuse_conv_and_bn,
    model_info,
    scale_img,
    initialize_weights,
)

sys.path.append("./")  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)


class YOLOV5n_05(nn.Module):
    name = "YOLO5n_05"

    def __init__(self, anchors=None, in_channels=3, num_classes=1, *args, **kwargs):
        super(YOLOV5n_05, self).__init__()
        self.names = [str(i) for i in range(num_classes)]
        self.nc = num_classes
        self.ch = in_channels
        self.anchors = (
            anchors
            if anchors
            else [
                [4, 5, 8, 10, 13, 16],
                [23, 29, 43, 55, 73, 105],
                [146, 217, 231, 300, 335, 433],
            ]
        )
        self.head = [
            Conv(256, 64, 1, 1),  # 7
            nn.Upsample(None, 2, "nearest"),  # 8
            Concat(1),  # 9 - (8, 4)
            C3(128 + 64, 64, 1, False),  # 10
            Conv(64, 64, 1, 1),  # 11
            nn.Upsample(None, 2, "nearest"),  # 12
            Concat(1),  # 13 - (12, 2)
            C3(64 + 64, 64, 1, False),  # 14
            # Detect
            Conv(64, 64, 3, 2),  # 15
            Concat(1),  # 16 -  (15, 11)
            C3(64 + 64, 64, 1, False),  # 17
            # Detect
            Conv(64, 64, 3, 2),  # 18
            Concat(1),  # 19 - (18 - 7)
            C3(64 + 64, 64, 1, False),  # 20
            # Detect
            Detect(num_classes, self.anchors, [64, 64, 64]),  # 21
        ]

        self.backbone = [
            StemBlock(in_channels, 16, 3, 2),  # 0
            ShuffleV2Block(16, 64, 2),  # 1 - P3/8 - small
            [ShuffleV2Block, (64, 64, 1), 3],  # 2
            ShuffleV2Block(64, 128, 2),  # 3 - P4/16 - medium
            [ShuffleV2Block, (128, 128, 1), 7],  # 4
            ShuffleV2Block(128, 256, 2),  # 5 - P5/32 - large
            [ShuffleV2Block, (256, 256, 1), 3],  # 6
        ]
        layers = []
        for m in self.backbone + self.head:
            if isinstance(m, list):
                layers.append(nn.Sequential(*[m[0](*m[1]) for _ in range(m[2])]))
            else:
                layers.append(m)
        self.model = nn.Sequential(*layers)

        self.save_dict = {
            9: 4,
            13: 2,
            16: 11,
            19: 7,
            21: [14, 17, 20],
        }  # {id layer concat/detect: id layer previous}
        self.save_point = []
        for i in self.save_dict.values():
            self.save_point.extend(i if isinstance(i, list) else [i])

        detect_layer = self.model[-1]
        if isinstance(detect_layer, Detect):
            s = 128
            detect_layer.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, in_channels, s, s))
                ]
            )
            detect_layer.anchors /= detect_layer.stride.view(-1, 1, 1)
            check_anchor_order(detect_layer)
            self.stride = detect_layer.stride
            self._initialize_biases()

        initialize_weights(self)
        model_info(self, verbose=False, name=self.name)

    def forward(self, x, augment=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]

                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None
        else:
            return self.forward_once(x)

    def forward_once(self, x):
        y = []
        for index, m in enumerate(self.model):
            if index in self.save_dict.keys():
                x = (
                    [x, y[self.save_dict[index]]]
                    if isinstance(self.save_dict[index], int)
                    else [y[j] for j in self.save_dict[index]]
                )
            x = m(x)
            y.append(x if index in self.save_point else None)
        return x

    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.fuseforward
        return self


class BlazeBlock(nn.Module):
    def __init__(self, inp, oup1, oup2=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        # double-block is used when oup2 is specified.
        self.use_double_block = oup2 is not None
        # pooling is used when stride is not 1
        self.use_pooling = self.stride != 1
        # change padding settings to insure pixel size is kept.
        if self.use_double_block:
            self.channel_pad = oup2 - inp
        else:
            self.channel_pad = oup1 - inp
        padding = (kernel_size - 1) // 2

        # mobile-net like convolution function is defined.
        self.conv1 = nn.Sequential(
            # dw
            # https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315
            # if groups=inp, it acts as depth wise convolution in pytorch
            nn.Conv2d(
                inp,
                inp,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=inp,
                bias=True,
            ),
            nn.BatchNorm2d(inp),
            # piecewise-linear convolution.
            nn.Conv2d(inp, oup1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup1),
        )
        self.act = nn.ReLU(inplace=True)

        # for latter layers, use resnet-like double convolution.
        if self.use_double_block:
            self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    oup1,
                    oup1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    groups=oup1,
                    bias=True,
                ),
                nn.BatchNorm2d(oup1),
                # pw-linear
                nn.Conv2d(oup1, oup2, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup2),
            )

        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x):
        h = self.conv1(x)
        if self.use_double_block:
            h = self.conv2(h)

        # skip connection
        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        return self.act(h + x)


# initialize weights.
def initialize(module):
    # original implementation is unknown
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)


class BlazeFace(nn.Module):
    """Constructs a BlazeFace model
    the original paper
    https://sites.google.com/view/perception-cv4arvr/blazeface
    """

    def __init__(self, channels=24):
        super(BlazeFace, self).__init__()
        # input..128x128
        self.features = nn.Sequential(
            nn.Conv2d(
                3, channels, kernel_size=3, stride=2, padding=1, bias=True
            ),  # pix=64
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            BlazeBlock(channels, channels),
            BlazeBlock(channels, channels),
            BlazeBlock(channels, channels * 2, stride=2),  # pix=32
            BlazeBlock(channels * 2, channels * 2),
            BlazeBlock(channels * 2, channels * 2),
            BlazeBlock(channels * 2, channels, channels * 4, stride=2),  # pix=16
            BlazeBlock(channels * 4, channels, channels * 4),
            BlazeBlock(channels * 4, channels, channels * 4),
        )
        self.apply(initialize)

    def forward(self, x):
        h = self.features(x)
        return h


class BlazeFaceExtra(nn.Module):
    """Constructs a BlazeFace model
    the original paper
    https://sites.google.com/view/perception-cv4arvr/blazeface
    """

    def __init__(self, channels=24):
        super(BlazeFaceExtra, self).__init__()
        self.features = nn.Sequential(
            BlazeBlock(channels * 4, channels, channels * 4, stride=2),  # pix=8
            BlazeBlock(channels * 4, channels, channels * 4),
            BlazeBlock(channels * 4, channels, channels * 4),
        )
        self.apply(initialize)

    def forward(self, x):
        h = self.features(x)
        return h


class YOLOv5_BlazeFace(nn.Module):
    """
    Backbone: blazeface
    Loss + anchor : YOLOv5
    """

    name = "YOLOv5_BlazeFace"
    detect_head = 2

    def __init__(self, n_classes=1, in_channels=3, channels=24, anchors=None):
        super(YOLOv5_BlazeFace, self).__init__()
        self.anchors = (
            anchors
            if anchors
            else [
                [20, 23, 79, 80, 105, 146, 115, 135],
                [126, 127, 117, 147, 126, 141, 193, 193],
            ]
        )
        self.names = [str(i) for i in range(n_classes)]
        self.nc = n_classes
        self.ch = in_channels
        self.blaze = BlazeFace(channels)
        self.extra = BlazeFaceExtra(channels)
        self.detect = Detect(1, self.anchors, [96, 96], kernel_size=3)
        self.model = [self.blaze, self.extra, self.detect]
        detect_layer = self.model[-1]
        if isinstance(detect_layer, Detect):
            s = 16 * 10
            detect_layer.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, in_channels, s, s))
                ]
            )
            detect_layer.anchors /= detect_layer.stride.view(-1, 1, 1)
            check_anchor_order(detect_layer)
            self.stride = detect_layer.stride
            self._initialize_biases()

        model_info(self, verbose=False, name=self.name)

    def forward(self, x):
        source = []
        x = self.model[0](x)
        source.append(x)
        x = self.model[1](x)
        source.append(x)
        y = self.model[2](source)
        return y

    def _initialize_biases(self, cf=None):
        m = self.detect
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        for m in self.model:
            if type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.fuseforward
        return self


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs, c, w, h = 1, 3, 320, 320
    input_ = torch.Tensor(bs, c, w, h).to(device)
    net = YOLOv5_BlazeFace().to(device)
    oup = net(input_)
    print(oup[0].shape)
    print(oup[1].shape)
