import torch.nn as nn


class ConvLeakyReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=True,
                 bn=False):
        super(ConvLeakyReLU, self).__init__()
        self.bn = bn

        if self.bn:
            self.bn2d = nn.BatchNorm2d(out_channels)
            bias = False

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn2d(x)
        x = self.activation(x)
        return x
