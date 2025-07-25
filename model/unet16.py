import torch.nn as nn
from torchvision.models.vgg import vgg16, VGG16_Weights

import torch

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        mid_channels : int,
        out_channels : int,
    ) -> None:
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)

class Unet16(nn.Module):
    def __init__(
        self,
        num_classes : int,
        pretrained : VGG16_Weights,
        num_filters : int = 32,
    ) -> None:
        super(Unet16, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.activation = nn.ReLU()
        self.encoder = vgg16(weights=pretrained).features

        self.conv1 = nn.Sequential(
            self.encoder[0],
            self.activation,

            self.encoder[2],
            self.activation,
        )
        self.conv2 = nn.Sequential(
            self.encoder[5],
            self.activation,

            self.encoder[7],
            self.activation,
        )

        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.activation,

            self.encoder[12],
            self.activation,

            self.encoder[14],
            self.activation
        )

        self.conv4 = nn.Sequential(
            self.encoder[17],
            self.activation,

            self.encoder[19],
            self.activation,

            self.encoder[21],
            self.activation
        )

        self.conv5 = nn.Sequential(
            self.encoder[24],
            self.activation,

            self.encoder[26],
            self.activation,

            self.encoder[28],
            self.activation
        )

        self.bottleneck = DecoderBlock(
            in_channels =  512, 
            mid_channels = num_filters * 16, 
            out_channels = num_filters * 8
        )

        self.dec5 = DecoderBlock(
            in_channels = 512 + num_filters * 8, 
            mid_channels = num_filters * 16,
            out_channels = num_filters * 8
        )

        self.dec4 = DecoderBlock(
            in_channels = 512 + num_filters * 8, 
            mid_channels = num_filters * 16,
            out_channels = num_filters * 8
        )

        self.dec3 = DecoderBlock(
            in_channels = 256 + num_filters * 8, 
            mid_channels = num_filters * 8,
            out_channels = num_filters * 4
        )

        self.dec2 = DecoderBlock(
            in_channels = 128 + num_filters * 4, 
            mid_channels = num_filters * 4,
            out_channels = num_filters
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=64 + num_filters, out_channels=num_filters, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.bottleneck(self.pool(conv5))
        
        dec5 = self.dec5(torch.cat([center, conv5], dim=1))

        dec4 = self.dec4(torch.cat([dec5, conv4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, conv3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, conv2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, conv1], dim=1))

        x_out = self.final(dec1).squeeze(1)

        return x_out
