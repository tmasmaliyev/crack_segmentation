import torch.nn as nn
import torch.nn.functional as F

import torch

class Attention(nn.Module):
    def __init__(self, in_channels : int) -> None:
        super(Attention, self).__init__()

        self.W_x = nn.Sequential(
          nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 1,
            stride = 2
          ),
          nn.BatchNorm2d(in_channels)
        )

        self.W_g = nn.Sequential(
          nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 1,
            stride = 1
          ),
          nn.BatchNorm2d(in_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = 1,
                kernel_size = 1
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
    
    def forward(
        self, 
        g : torch.Tensor, 
        x : torch.Tensor
    ) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)

        return x * psi

        