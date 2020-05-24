import torch
import torch.nn as nn
from .splat import SplAtConv2d


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 radix=1,
                 cardinality=1,
                 bottleneck_width=64,
                 avd=False,
                 avd_first=False,
                 dilation=1,
                 is_first=False,
                 rectified_conv=False,
                 rectify_avg=False,
                 norm_layer=None,
                 dropblock_prob=0.0,
                 last_gamma=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(group_width,
                                     group_width,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=dilation,
                                     dilation=dilation,
                                     groups=cardinality,
                                     bias=False,
                                     radix=radix,
                                     rectify=rectified_conv,
                                     rectify_avg=rectify_avg,
                                     norm_layer=norm_layer,
                                     dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(group_width,
                                  group_width,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=dilation,
                                  dilation=dilation,
                                  groups=cardinality,
                                  bias=False,
                                  average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(group_width,
                                   group_width,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=dilation,
                                   dilation=dilation,
                                   groups=cardinality,
                                   bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
