import torch.nn as nn
import torch

from .attention import Attention

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
    
class DecoderAttBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        mid_channels : int,
        out_channels : int,
    ) -> None:
        super(DecoderAttBlock, self).__init__()

        self.dropout = nn.Dropout2d(0.25)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            self.dropout
        )

        self.attention = Attention(in_channels=mid_channels)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            self.dropout
        )
    
    def forward(
        self, 
        g : torch.Tensor,
        x : torch.Tensor, 
    ) -> torch.Tensor:
        g1 = self.block(g)
        g2 = self.upsample(g1)

        x1 = self.attention(g1, x)

        return torch.cat([g2, x1], dim=1)