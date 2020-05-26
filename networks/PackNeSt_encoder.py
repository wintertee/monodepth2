# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch.nn as nn
import torch.nn.functional as F

from .Packlayers import PackLayerConv3d, ResidualBlock, Conv2D



class PackNeSt_encoder(nn.Module):
    """
    PackNet network with 3d convolutions (version 01, from the CVPR paper).

    https://arxiv.org/abs/1905.02693

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self):
        super().__init__()
        # Input channels
        in_channels = 3
        # Hyper-parameters
        ni = 64
        n1, n2, n3, n4, n5 = 64, 64, 128, 256, 512
        # n1, n2, n3, n4, n5 = 64, 256, 512, 1024, 2048
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        # Initial convolutional layer
        self.pre_calc = Conv2D(in_channels, ni, 5, 1)

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0], d=2)
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1], d=2)
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2], d=2)
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3], d=2)
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4], d=1)

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2 // 4, num_blocks[0], 1)
        self.conv3 = ResidualBlock(n2, n3 // 4, num_blocks[1], 1)
        self.conv4 = ResidualBlock(n3, n4 // 4, num_blocks[2], 1)
        self.conv5 = ResidualBlock(n4, n5 // 4, num_blocks[3], 1)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """

        x = (x - 0.45) / 0.225
        x = self.pre_calc(x)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        self.features = [x, x1p, x2p, x3p, x4p, x5p]

        return self.features


########################################################################################################################
