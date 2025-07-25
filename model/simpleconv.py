import torch.nn as nn

class SimpleConv(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int
    ) -> None:
        super(SimpleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.simpleconv = nn.Sequential(
            nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.simpleconv(x)