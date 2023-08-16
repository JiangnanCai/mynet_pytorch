from torch.nn import (Module, Sequential, Dropout,
                      LeakyReLU, Linear,
                      Conv2d, BatchNorm2d,
                      AvgPool2d, MaxPool2d, AdaptiveAvgPool2d)

import torch
from typing import Union, Tuple


class Conv2dBN(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[str, int, Tuple[int, int]],
                 stride: int = 1,
                 activation=LeakyReLU(0.2, inplace=True)):  # todo： rule与leakyRelu区别
        super(Conv2dBN, self).__init__()
        self.conv2d_bn = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        return self.conv2d_bn(x)


class InceptionModuleA(Module):
    def __init__(self,
                 in_channels: int,
                 pool_channels: int):
        super(InceptionModuleA, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, 64, (1, 1), stride=1, padding=0)

        self.branch_3x3_double = Sequential(
            Conv2dBN(in_channels, 64, (1, 1), stride=1, padding=0),
            Conv2dBN(64, 96, (3, 3), stride=1, padding=1),
            Conv2dBN(96, 96, (3, 3), stride=1, padding=1)
        )

        self.branch_5x5 = Sequential(
            Conv2dBN(in_channels, 48, (1, 1), stride=1, padding=0),
            Conv2dBN(48, 64, (5, 5), stride=1, padding=2)
        )

        self.branch_pool = Sequential(
            AvgPool2d((3, 3), stride=1, padding=1),
            Conv2dBN(in_channels, pool_channels, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x), self.branch_3x3_double(x), self.branch_5x5(x), self.branch_pool(x)], 1)


class InceptionModuleD(Module):
    def __init__(self,
                 in_channels: int):
        super(InceptionModuleD, self).__init__()

        self.branch_3x3 = Conv2dBN(in_channels, 384, (3, 3), stride=2, padding=1)

        self.branch_3x3_double = Sequential(
            Conv2dBN(in_channels, 64, (1, 1), stride=1, padding=0),
            Conv2dBN(64, 96, (3, 3), stride=1, padding=1),
            Conv2dBN(96, 96, (3, 3), stride=2, padding=1)
        )

        self.branch_pool = MaxPool2d((3, 3), stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.branch_3x3(x), self.branch_3x3_double(x), self.branch_pool(x)], 1)


class InceptionModuleB(Module):
    def __init__(self,
                 in_channels: int,
                 channels_7x7: int):
        super(InceptionModuleB, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, 192, (1, 1), stride=1, padding=0)

        self.branch_7x7 = Sequential(
            Conv2dBN(in_channels, channels_7x7, (1, 1), stride=1, padding=0),
            Conv2dBN(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)),
            Conv2dBN(channels_7x7, 192, (7, 1), stride=1, padding=(3, 0))
        )

        self.branch_7x7_double = Sequential(
            Conv2dBN(in_channels, channels_7x7, (1, 1), stride=1, padding=0),
            Conv2dBN(channels_7x7, channels_7x7, (7, 1), stride=1, padding=(3, 0)),
            Conv2dBN(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)),
            Conv2dBN(channels_7x7, channels_7x7, (7, 1), stride=1, padding=(3, 0)),
            Conv2dBN(channels_7x7, 192, (1, 7), stride=1, padding=(0, 3))
        )

        self.branch_pool = Sequential(
            AvgPool2d((3, 3), stride=1, padding=1),
            Conv2dBN(in_channels, 192, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x), self.branch_7x7(x), self.branch_7x7_double(x), self.branch_pool(x)], 1)


class InceptionModuleE(Module):
    def __init__(self,
                 in_channels: int):
        super(InceptionModuleE, self).__init__()

        self.branch_3x3 = Sequential(
            Conv2dBN(in_channels, 192, (1, 1), stride=1, padding=0),
            Conv2dBN(192, 320, (3, 3), stride=2, padding=1)
        )

        self.branch_7x7_3x3 = Sequential(
            Conv2dBN(in_channels, 192, (1, 1), stride=1, padding=0),
            Conv2dBN(192, 192, (1, 7), stride=1, padding=(0, 3)),
            Conv2dBN(192, 192, (7, 1), stride=1, padding=(3, 0)),
            Conv2dBN(192, 192, (3, 3), stride=2, padding=1)
        )

        self.branch_pool = MaxPool2d((3, 3), stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.branch_3x3(x), self.branch_7x7_3x3(x), self.branch_pool(x)], 1)


class InceptionModuleC(Module):
    def __init__(self,
                 in_channels: int):
        super(InceptionModuleC, self).__init__()
        self.branch_1x1 = Conv2dBN(in_channels, 320, (1, 1), stride=1, padding=0)

        self.branch_3x3_root = Conv2dBN(in_channels, 384, (1, 1), stride=1, padding=0)
        self.branch_3x3_node_1 = Conv2dBN(384, 384, (1, 3), stride=1, padding=(0, 1))
        self.branch_3x3_node_2 = Conv2dBN(384, 384, (3, 1), stride=1, padding=(1, 0))

        self.branch_3x3_double_root = Sequential(
            Conv2dBN(in_channels, 448, (1, 1), stride=1, padding=0),
            Conv2dBN(448, 384, (3, 3), stride=1, padding=1)
        )
        self.branch_3x3_double_node_1 = Conv2dBN(384, 384, (1, 3), stride=1, padding=(0, 1))
        self.branch_3x3_double_node_2 = Conv2dBN(384, 384, (3, 1), stride=1, padding=(1, 0))

        self.branch_pool = Sequential(
            AvgPool2d((3, 3), stride=1, padding=1),
            Conv2dBN(in_channels, 192, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)

        branch_3x3 = torch.cat([self.branch_3x3_node_1(self.branch_3x3_root(x)),
                                self.branch_3x3_node_2(self.branch_3x3_root(x))], 1)

        branch_3x3_double = torch.cat([self.branch_3x3_double_node_1(self.branch_3x3_double_root(x)),
                                       self.branch_3x3_double_node_2(self.branch_3x3_double_root(x))], 1)

        branch_pool = self.branch_pool(x)

        return torch.cat([branch_1x1, branch_3x3, branch_3x3_double, branch_pool], 1)


class InceptionAux(Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avg_pool = AvgPool2d((5, 5), stride=3)
        self.conv_0 = Conv2dBN(in_channels, 128, (1, 1), stride=1, padding=0)
        self.conv_1 = Conv2dBN(128, 768, (3, 3), stride=1, padding=0)
        self.conv_1.stddev = 0.01
        self.fc = Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = x.view(x.size(0), -1)  # todo: torch.flatten(x, 1)与x.view(x.size(0), -1)区别
        return self.fc(x)


class InceptionV3(Module):
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 aux_logits=True,
                 init_weight=True):
        super(InceptionV3, self).__init__()

        self.aux_logits = aux_logits

        self.in_block = Sequential(
            Conv2dBN(in_channels, 32, (3, 3), stride=2, padding=0),
            Conv2dBN(32, 32, (3, 3), stride=1, padding=0),
            Conv2dBN(32, 64, (3, 3), stride=1, padding=1),
            MaxPool2d((3, 3), stride=2, padding=0),
            Conv2dBN(64, 80, (1, 1), stride=1, padding=0),
            Conv2dBN(80, 192, (3, 3), stride=1, padding=1),
            MaxPool2d((3, 3), stride=2, padding=0)
        )

        self.mix_block_0 = Sequential(
            InceptionModuleA(192, 32),
            InceptionModuleA(256, 64),
            InceptionModuleA(288, 64),
            InceptionModuleD(288),
            InceptionModuleB(768, 128),
            InceptionModuleB(768, 160),
            InceptionModuleB(768, 160),
            InceptionModuleB(768, 192),
        )

        if self.aux_logits:
            self.aux = InceptionAux(768, num_classes)

        self.mix_block_1 = Sequential(
            InceptionModuleE(768),
            InceptionModuleC(1280),
            InceptionModuleC(2048)
        )

        self.out_block = Sequential(
            Conv2dBN(2048, 1024, 1, stride=1, padding=0),
            AdaptiveAvgPool2d((1, 1))
        )

        self.dropout = Dropout(0.5)
        self.fc = Linear(1024, num_classes)

        if init_weight:
            self._initialize_weight()

    def forward(self, x):
        x = self.in_block(x)
        x = self.mix_block_0(x)
        aux = self.aux(x) if self.aux_logits and self.training else None
        x = self.mix_block_1(x)
        x = self.out_block(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, aux if self.training and self.aux_logits else x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

