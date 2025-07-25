import torch.nn as nn
import torch

from .doubleconv import DoubleConv

class SUNet(nn.Module):
    def __init__(self) -> None:
        super(SUNet, self).__init__()
        # region Encoder

        # region Block #1
        self.conv1 = DoubleConv(
            in_channels = 3,
            out_channels = 32
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size = 2
        )
        #endregion

        # region Block #2
        self.conv2 = DoubleConv(
            in_channels = 32,
            out_channels = 64
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size = 2
        )
        # endregion

        # region Block #3
        self.conv3 = DoubleConv(
            in_channels = 64,
            out_channels = 128
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size = 2
        )
        # endregion

        # region Block #4
        self.conv4 = DoubleConv(
            in_channels = 128,
            out_channels = 256
        )

        self.pool4 = nn.MaxPool2d(
            kernel_size = 2
        )
        # endregion
        
        #endregion

        # region BottleNeck Block
        self.conv5 = DoubleConv(
            in_channels = 256,
            out_channels = 512
        )

        # endregion
        
        # region Decoder
        
        # region Block #1
        self.up1 = nn.Upsample(
            scale_factor = 2,
            mode = 'bilinear',
            align_corners = True
        )

        self.conv6 = DoubleConv(
            in_channels = 768,
            out_channels = 256
        )
        # endregion

        # region Block #2
        self.up2 = nn.Upsample(
            scale_factor = 2,
            mode = 'bilinear',
            align_corners = True
        )

        self.conv7 = DoubleConv(
            in_channels = 384,
            out_channels = 128
        )
        # endregion

        # region Block #3
        self.up3 = nn.Upsample(
            scale_factor = 2,
            mode = 'bilinear',
            align_corners = True
        )

        self.conv8 = DoubleConv(
            in_channels = 192,
            out_channels = 64
        )
        # endregion

        # region Block #4
        self.up4 = nn.Upsample(
            scale_factor = 2,
            mode = 'bilinear',
            align_corners = True
        )

        self.conv9 = DoubleConv(
            in_channels = 96,
            out_channels = 32
        )
        # endregion

        # endregion

        # region Output mask creation
        self.conv10 = nn.Conv2d(
            in_channels = 32,
            out_channels = 2,
            kernel_size = 1
        )
        # endregion

    def forward(self, x):
        # region Forward on Encoder
        x1 = self.conv1(x)

        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))
        
        # endregion

        # region Forward on Decoder
        x6 = self.conv6(torch.cat([self.up1(x5), x4], dim=1))
        x7 = self.conv7(torch.cat([self.up2(x6), x3], dim=1))
        x8 = self.conv8(torch.cat([self.up3(x7), x2], dim=1))
        x9 = self.conv9(torch.cat([self.up4(x8), x1], dim=1))
        # endregion

        # region Mask creation
        x10 = self.conv10(x9)
        # endregion

        return x10
    