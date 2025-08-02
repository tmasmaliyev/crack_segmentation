import torch.nn as nn
from torchvision.models.vgg import vgg16, VGG16_Weights

from .decoder_block import DecoderAttBlock, DecoderBlock

import torch

class AttUnet16(nn.Module):
    def __init__(
        self,
        num_classes : int,
        pretrained : VGG16_Weights,
        num_filters : int = 32,
    ) -> None:
        super(AttUnet16, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.activation = nn.ReLU()
        self.encoder = vgg16(weights=pretrained).features
        self.dropout = nn.Dropout2d(0.25)

        self.conv1 = nn.Sequential(
            self.encoder[0],
            nn.BatchNorm2d(self.encoder[0].out_channels),
            self.activation,

            self.encoder[2],
            nn.BatchNorm2d(self.encoder[2].out_channels),
            self.activation,

            self.dropout
        )

        self.conv2 = nn.Sequential(
            self.encoder[5],
            nn.BatchNorm2d(self.encoder[5].out_channels),
            self.activation,

            self.encoder[7],
            nn.BatchNorm2d(self.encoder[7].out_channels),
            self.activation,

            self.dropout
        )

        self.conv3 = nn.Sequential(
            self.encoder[10],
            nn.BatchNorm2d(self.encoder[10].out_channels),
            self.activation,

            self.encoder[12],
            nn.BatchNorm2d(self.encoder[12].out_channels),
            self.activation,

            self.encoder[14],
            nn.BatchNorm2d(self.encoder[14].out_channels),
            self.activation,

            self.dropout
        )

        self.conv4 = nn.Sequential(
            self.encoder[17],
            nn.BatchNorm2d(self.encoder[17].out_channels),
            self.activation,

            self.encoder[19],
            nn.BatchNorm2d(self.encoder[19].out_channels),
            self.activation,

            self.encoder[21],
            nn.BatchNorm2d(self.encoder[21].out_channels),
            self.activation,

            self.dropout
        )

        self.conv5 = nn.Sequential(
            self.encoder[24],
            nn.BatchNorm2d(self.encoder[24].out_channels),
            self.activation,

            self.encoder[26],
            nn.BatchNorm2d(self.encoder[26].out_channels),
            self.activation,

            self.encoder[28],
            nn.BatchNorm2d(self.encoder[28].out_channels),
            self.activation
        )

        self.bottleneck = DecoderAttBlock(
            in_channels =  512, 
            mid_channels = num_filters * 16, 
            out_channels = num_filters * 8
        )

        self.dec5 = DecoderAttBlock(
            in_channels = 512 + num_filters * 8, 
            mid_channels = num_filters * 16,
            out_channels = num_filters * 8
        )

        self.dec4 = DecoderAttBlock(
            in_channels = 512 + num_filters * 8, 
            mid_channels = num_filters * 8,
            out_channels = num_filters * 8
        )

        self.dec3 = DecoderAttBlock(
            in_channels = 256 + num_filters * 8, 
            mid_channels = num_filters * 4,
            out_channels = num_filters * 4
        )

        self.dec2 = DecoderAttBlock(
            in_channels = 128 + num_filters * 4, 
            mid_channels = num_filters * 2,
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

        center = self.bottleneck(self.pool(conv5), conv5)

        dec5 = self.dec5(center, conv4)
        dec4 = self.dec4(dec5, conv3)
        dec3 = self.dec3(dec4, conv2)
        dec2 = self.dec2(dec3, conv1)
        dec1 = self.dec1(dec2)

        x_out = self.final(dec1).squeeze(1)
  
        return x_out