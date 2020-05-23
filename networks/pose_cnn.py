# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames  # ANCHOR 输入帧数。用途？

        self.convs = {}  # ANCHOR 为啥要用字典，直接用list它不香吗？
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2,
                                  3)  # 三维卷积核心 (3 * num_input_frames) * 7 * 7，把输入通道改为3*帧数，输出通道从64改成16，输出分辨率减半
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)  # 输出1/4
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)  # 输出1/8
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)  # 输出1/16
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)  # 输出1/32
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)  # 输出1/64
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)  # 输出1/128

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1),
                                   1)  # 卷积核为1，输出6* (num_input_frames - 1)个通道，表示自由度在每两帧之间的变化，输出分辨率1/128

        self.num_convs = len(self.convs)  # 除了pose_conv 之外的层数

        self.relu = nn.ReLU(True)  # True: 覆盖原值，效率更高

        self.net = nn.ModuleList(list(self.convs.values()))  # 将convs变为子module

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)  # 三维张量大小为 (6 * (num_input_frames - 1)) * 1/128原图 * 1/128原图
        out = out.mean(3).mean(2)
        # 在第三个维度和第二个维度 （1/128原图 * 1/128原图） 上取均值，输出的一维张量大小为 6 * (num_input_frames - 1)，即每两帧之间的6个自由度变化
        # ANCHOR 为什么要最后一层用卷积 + 取平均（相当于average_pool 到1个像素）？改成一层或两层FC不香吗？

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)
        # reshape, 第一个维度自适应，第二个维度大小是 num_input_frames - 1，第三个维度大小是 1，第四个维度大小是 6个自由度

        axisangle = out[..., :3]  # 前三个代表角度
        translation = out[..., 3:]  # 后三个代表位移

        return axisangle, translation
