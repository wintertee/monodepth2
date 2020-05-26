# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from .Packlayers import UnpackLayerConv3d, Conv2D, InvDepth

########################################################################################################################


class PackNeSt_decoder(nn.Module):
    """
    PackNet network with 3d convolutions (version 01, from the CVPR paper).

    https://arxiv.org/abs/1905.02693

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Input/output channels
        in_channels = 3
        out_channels = 1
        # Hyper-parameters
        ni, no = 64, out_channels
        n1, n2, n3, n4, n5 = 64, 64, 128, 256, 512
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        # Initial convolutional layer
        self.pre_calc = Conv2D(in_channels, ni, 5, 1)

        n1o, n1i = n1, n1 + ni + no
        n2o, n2i = n2, n2 + n1 + no
        n3o, n3i = n3, n3 + n2 + no
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0])
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1])
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2])
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3])
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4])

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.functional.interpolate(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.functional.interpolate(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.functional.interpolate(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """

        # Skips

        skip1 = features[0]
        skip2 = features[1]
        skip3 = features[2]
        skip4 = features[3]
        skip5 = features[4]

        # Decoder
        x5p = features[5]

        unpack5 = self.unpack5(x5p)
        concat5 = torch.cat((unpack5, skip5), 1)
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        concat4 = torch.cat((unpack4, skip4), 1)
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4_layer(iconv4)
        udisp4 = self.unpack_disp4(disp4)

        unpack3 = self.unpack3(iconv4)
        concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3_layer(iconv3)
        udisp3 = self.unpack_disp3(disp3)

        unpack2 = self.unpack2(iconv3)
        concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2_layer(iconv2)
        udisp2 = self.unpack_disp2(disp2)

        unpack1 = self.unpack1(iconv2)
        concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1_layer(iconv1)

        self.outputs = {}
        self.outputs[('disp', 0)] = disp1
        self.outputs[('disp', 1)] = disp2
        self.outputs[('disp', 2)] = disp3
        self.outputs[('disp', 3)] = disp4

        return self.outputs


########################################################################################################################
