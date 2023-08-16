from torch.nn import (Module, Sequential, Dropout,
                      LeakyReLU, Linear,
                      Conv2d, BatchNorm2d,
                      AvgPool2d, MaxPool2d, AdaptiveAvgPool2d)

from typing import Tuple, Union
import torch


class Conv2dBN(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[str, int, Tuple[int, int]],
                 stride: int = 1,
                 activation=LeakyReLU(0.2, inplace=True)):
        super(Conv2dBN, self).__init__()
        self.conv2d_bn = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        return self.conv2d_bn(x)


class InceptionV4A(Module):
    def __init__(self, in_channels):
        super(InceptionV4A, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, 96, (1, 1), stride=1, padding=0)

        self.branch_3x3 = Sequential(
            Conv2dBN(in_channels, 64, (1, 1), stride=1, padding=0),
            Conv2dBN(64, 96, (3, 3), stride=1, padding=1)
        )

        self.branch_3x3_double = Sequential(
            Conv2dBN(in_channels, 64, (1, 1), stride=1, padding=0),
            Conv2dBN(64, 96, (3, 3), stride=1, padding=1),
            Conv2dBN(96, 96, (3, 3), stride=1, padding=1)
        )

        self.branch_pool = Sequential(
            AvgPool2d((3, 3), stride=1, padding=1, count_include_pad=False),
            Conv2dBN(384, 96, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x), self.branch_3x3(x), self.branch_3x3_double(x), self.branch_pool(x)], 1)


class InceptionV4B(Module):
    def __init__(self, in_channels):
        super(InceptionV4B, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, 384, (1, 1), stride=1, padding=0)

        self.branch_7x7 = Sequential(
            Conv2dBN(in_channels, 192, (1, 1), stride=1, padding=0),
            Conv2dBN(192, 224, (1, 7), stride=1, padding=(0, 3)),
            Conv2dBN(224, 256, (7, 1), stride=1, padding=(3, 0))
        )

        self.branch_7x7_double = Sequential(
            Conv2dBN(in_channels, 192, (1, 1), stride=1, padding=0),
            Conv2dBN(192, 192, (7, 1), stride=1, padding=(3, 0)),
            Conv2dBN(192, 224, (1, 7), stride=1, padding=(0, 3)),
            Conv2dBN(224, 224, (7, 1), stride=1, padding=(3, 0)),
            Conv2dBN(224, 256, (1, 7), stride=1, padding=(0, 3))
        )

        self.branch_pool = Sequential(
            AvgPool2d((3, 3), stride=1, padding=1, count_include_pad=False),
            Conv2dBN(in_channels, 128, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x), self.branch_7x7(x), self.branch_7x7_double(x), self.branch_pool(x)], 1)


class InceptionV4C(Module):
    def __init__(self, in_channels):
        super(InceptionV4C, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, 256, (1, 1), stride=1, padding=0)

        self.branch_3x3_root = Conv2dBN(in_channels, 384, (1, 1), stride=1, padding=0)
        self.branch_3x3_node_1 = Conv2dBN(384, 256, (1, 3), stride=1, padding=(0, 1))
        self.branch_3x3_node_2 = Conv2dBN(384, 256, (3, 1), stride=1, padding=(1, 0))

        self.branch_5x5_root = Sequential(
            Conv2dBN(in_channels, 384, (1, 1), stride=1, padding=0),
            Conv2dBN(384, 448, (3, 1), stride=1, padding=(1, 0)),
            Conv2dBN(448, 512, (1, 3), stride=1, padding=(0, 1))
        )
        self.branch_5x5_node_1 = Conv2dBN(512, 256, (1, 3), stride=1, padding=(0, 1))
        self.branch_5x5_node_2 = Conv2dBN(512, 256, (3, 1), stride=1, padding=(1, 0))

        self.branch_pool = Sequential(
            AvgPool2d((3, 3), stride=1, padding=1, count_include_pad=False),
            Conv2dBN(in_channels, 256, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)

        branch_3x3 = torch.cat([self.branch_3x3_node_1(self.branch_3x3_root(x)),
                                self.branch_3x3_node_2(self.branch_3x3_root(x))], 1)

        branch_5x5 = torch.cat([self.branch_5x5_node_1(self.branch_5x5_root(x)),
                                self.branch_5x5_node_2(self.branch_3x3_root(x))], 1)

        branch_pool = self.branch_pool(x)

        return torch.cat([branch_1x1, branch_3x3, branch_5x5, branch_pool], 1)


class ReductionV4A(Module):
    def __init__(self, in_channels, k, l, m, n):
        super(ReductionV4A, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, n, (3, 3), stride=2, padding=0)

        self.branch_3x3 = Sequential(
            Conv2dBN(in_channels, k, (1, 1), stride=1, padding=0),
            Conv2dBN(k, l, (3, 3), stride=1, padding=1),
            Conv2dBN(l, m, (3, 3), stride=2, padding=0)
        )

        self.branch_pool = MaxPool2d((3, 3), stride=2, padding=0)

    def forward(self, x):
        return torch.cat([self.branch_1x1(x), self.branch_3x3(x), self.branch_pool(x)], 1)


class ReductionV4B(Module):
    def __init__(self, in_channels):
        super(ReductionV4B, self).__init__()

        self.branch_3x3 = Sequential(
            Conv2dBN(in_channels, 192, (1, 1), stride=1, padding=0),
            Conv2dBN(192, 192, (3, 3), stride=2, padding=0)
        )

        self.branch_7x7x3 = Sequential(
            Conv2dBN(in_channels, 256, (1, 1), stride=1, padding=0),
            Conv2dBN(256, 256, (1, 7), stride=1, padding=(0, 3)),
            Conv2dBN(256, 320, (7, 1), stride=1, padding=(3, 0)),
            Conv2dBN(320, 320, (3, 3), stride=2, padding=0)
        )

        self.branch_pool = MaxPool2d((3, 3), stride=2, padding=0)

    def forward(self, x):
        return torch.cat([self.branch_3x3(x), self.branch_7x7x3(x), self.branch_pool(x)], 1)


class Stem(Module):
    # todo: 这个stem是给inception v4 / inception resnet v2 使用的
    def __init__(self, in_channels):
        super(Stem, self).__init__()

        self.block_0 = Sequential(
            Conv2dBN(in_channels, 32, (3, 3), stride=2, padding=0),
            Conv2dBN(32, 32, (3, 3), stride=1, padding=0),
            Conv2dBN(32, 64, (3, 3), stride=1, padding=1)
        )

        self.block_1_node_1 = MaxPool2d((3, 3), stride=2, padding=0)
        self.block_1_node_2 = Conv2dBN(64, 96, (3, 3), stride=2, padding=0)

        self.block_2_node_1 = Sequential(
            Conv2dBN(160, 64, (1, 1), stride=1, padding=1),
            Conv2dBN(64, 96, (3, 3), stride=1, padding=0)
        )
        self.block_2_node_2 = Sequential(
            Conv2dBN(160, 64, (1, 1), stride=1, padding=1),
            Conv2dBN(64, 64, (1, 7), stride=1, padding=(0, 3)),
            Conv2dBN(64, 64, (7, 1), stride=1, padding=(3, 0)),
            Conv2dBN(64, 96, (3, 3), stride=1, padding=0)
        )

        self.block_3_node_1 = Conv2dBN(192, 192, (3, 3), stride=2, padding=0)
        self.block_3_node_2 = MaxPool2d((3, 3), stride=2, padding=0)

    def forward(self, x):
        x = self.block_0(x)
        x = torch.cat([self.block_1_node_1(x), self.block_1_node_2(x)], 1)
        x = torch.cat([self.block_2_node_1(x), self.block_2_node_2(x)], 1)
        x = torch.cat([self.block_3_node_1(x), self.block_3_node_2(x)], 1)
        return x


class InceptionV4(Module):
    def __init__(self, in_channels, classes=1000, k=192, l=224, m=256, n=384):
        super(InceptionV4).__init__()

        self.in_block = Stem(in_channels)

        self.mix_block = Sequential(
            InceptionV4A(384),
            InceptionV4A(384),
            InceptionV4A(384),
            InceptionV4A(384),

            ReductionV4A(384, k, l, m, n),

            InceptionV4B(1024),
            InceptionV4B(1024),
            InceptionV4B(1024),
            InceptionV4B(1024),
            InceptionV4B(1024),
            InceptionV4B(1024),
            InceptionV4B(1024),

            ReductionV4B(1024),

            InceptionV4C(1536),
            InceptionV4C(1536),
            InceptionV4C(1536),
        )

        self.global_avg_pooling = AdaptiveAvgPool2d((1, 1))
        self.linear = Linear(1536, classes)

    def forward(self, x):
        x = self.in_block(x)
        x = self.mix_block(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
