from torch.nn import (Module, Sequential, functional,
                      LeakyReLU, Linear, Dropout,
                      Conv2d, BatchNorm2d,
                      AvgPool2d, MaxPool2d, AdaptiveAvgPool2d)
from typing import Union, Tuple
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


class InceptionModuleA(Module):
    def __init__(self,
                 in_channels,
                 out_channels_1x1,
                 mid_channels_3x3,
                 out_channels_3x3,
                 mid_channels_5x5,
                 out_channels_5x5,
                 out_channels_pool):
        super(InceptionModuleA, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, out_channels_1x1, (1, 1), stride=1, padding=0)

        self.branch_3x3 = Sequential(
            Conv2dBN(in_channels, mid_channels_3x3, (1, 1), stride=1, padding=0),
            Conv2dBN(mid_channels_3x3, out_channels_3x3, (3, 3), stride=1, padding=1)
        )

        self.branch_5x5_sep = Sequential(
            Conv2dBN(in_channels, mid_channels_5x5, (1, 1), stride=1, padding=0),
            Conv2dBN(mid_channels_5x5, mid_channels_5x5, (3, 3), stride=1, padding=1),
            Conv2dBN(mid_channels_5x5, out_channels_5x5, (3, 3), stride=1, padding=1)
        )

        self.branch_pool = Sequential(
            MaxPool2d((3, 3), stride=1, padding=1),
            Conv2dBN(in_channels, out_channels_pool, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x),
                          self.branch_3x3(x),
                          self.branch_5x5_sep(x),
                          self.branch_pool(x)], dim=1)


class InceptionModuleB(Module):
    def __init__(self,
                 in_channels,
                 out_channels_1x1,
                 mid_channels_3x3,
                 out_channels_3x3,
                 mid_channels_5x5,
                 out_channels_5x5,
                 out_channels_pool):
        super(InceptionModuleB, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, out_channels_1x1, (1, 1), stride=1, padding=0)

        self.branch_3x3_sep = Sequential(
            Conv2dBN(in_channels, mid_channels_3x3, (1, 1), stride=1, padding=0),
            Conv2dBN(mid_channels_3x3, mid_channels_3x3, (1, 3), stride=1, padding=(0, 1)),
            Conv2dBN(mid_channels_3x3, out_channels_3x3, (3, 1), stride=1, padding=(1, 0))
        )

        self.branch_5x5_sep = Sequential(
            Conv2dBN(in_channels, mid_channels_5x5, (1, 1), stride=1, padding=0),
            Conv2dBN(mid_channels_5x5, mid_channels_5x5, (1, 3), stride=1, padding=(0, 1)),
            Conv2dBN(mid_channels_5x5, mid_channels_5x5, (3, 1), stride=1, padding=(1, 0)),
            Conv2dBN(mid_channels_5x5, mid_channels_5x5, (1, 3), stride=1, padding=(0, 1)),
            Conv2dBN(mid_channels_5x5, out_channels_5x5, (3, 1), stride=1, padding=(1, 0))
        )

        self.branch_pool = Sequential(
            MaxPool2d((3, 3), stride=1, padding=1),
            Conv2dBN(in_channels, out_channels_pool, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x),
                          self.branch_3x3_sep(x),
                          self.branch_5x5_sep(x),
                          self.branch_pool(x)], dim=1)


class InceptionModuleC(Module):
    def __init__(self,
                 in_channels,
                 out_channels_1x1,
                 mid_channels_3x3,
                 out_channels_3x3,
                 mid_channels_5x5,
                 out_channels_5x5,
                 out_channels_pool):
        super(InceptionModuleC, self).__init__()

        self.branch_1x1 = Conv2dBN(in_channels, out_channels_1x1, (1, 1), stride=1, padding=0)

        self.branch_3x3_sep_root = Conv2dBN(in_channels, mid_channels_3x3, (1, 1), stride=1, padding=0)
        self.branch_3x3_sep_node_0 = Conv2dBN(mid_channels_3x3, out_channels_3x3, (1, 3), stride=1, padding=(0, 1))
        self.branch_3x3_sep_node_1 = Conv2dBN(mid_channels_3x3, out_channels_3x3, (3, 1), stride=1, padding=(1, 0))

        self.branch_5x5_sep_root = Sequential(
            Conv2dBN(in_channels, mid_channels_5x5, (1, 1), stride=1, padding=0),
            Conv2dBN(mid_channels_5x5, out_channels_5x5, (3, 3), stride=1, padding=1)
        )
        self.branch_5x5_sep_node_0 = Conv2dBN(out_channels_5x5, out_channels_5x5, (3, 1), stride=1, padding=(1, 0))
        self.branch_5x5_sep_node_1 = Conv2dBN(out_channels_5x5, out_channels_5x5, (1, 3), stride=1, padding=(0, 1))

        self.branch_pool = Sequential(
            MaxPool2d((3, 3), stride=1, padding=1),
            Conv2dBN(in_channels, out_channels_pool, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x),
                          torch.cat([self.branch_3x3_sep_node_0(self.branch_3x3_sep_root(x)),
                                     self.branch_3x3_sep_node_1(self.branch_3x3_sep_root(x))], dim=1),
                          torch.cat([self.branch_5x5_sep_node_0(self.branch_5x5_sep_root(x)),
                                     self.branch_5x5_sep_node_1(self.branch_5x5_sep_root(x))], dim=1),
                          self.branch_pool(x)], dim=1)


class InceptionModuleD(Module):
    def __init__(self,
                 in_channels,
                 mid_channels_3x3,
                 out_channels_3x3,
                 mid_channels_5x5,
                 out_channels_5x5):
        super(InceptionModuleD, self).__init__()

        self.branch_3x3 = Sequential(
            Conv2dBN(in_channels, mid_channels_3x3, (1, 1), stride=1, padding=0),
            Conv2dBN(mid_channels_3x3, out_channels_3x3, (3, 3), stride=2, padding=1)
        )

        self.branch_5x5 = Sequential(
            Conv2dBN(in_channels, mid_channels_5x5, (1, 1), stride=1, padding=0),
            Conv2dBN(mid_channels_5x5, out_channels_5x5, (3, 3), stride=1, padding=1),
            Conv2dBN(out_channels_5x5, out_channels_5x5, (3, 3), stride=2, padding=1)
        )

        self.branch_pool = MaxPool2d((3, 3), stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.branch_3x3(x),
                          self.branch_5x5(x),
                          self.branch_pool(x)], dim=1)


class InceptionAux(Module):
    def __init__(self,
                 in_channels,
                 num_classes):
        super(InceptionAux, self).__init__()

        self.avg_pool = AvgPool2d((5, 5), stride=3)
        self.conv_0 = Conv2dBN(in_channels, 128, (1, 1), stride=1, padding=0)
        self.conv_1 = Conv2d(128, 768, (5, 5), stride=1, padding=0)
        self.dropout = Dropout(0.5)
        self.linear = Linear(768, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class InceptionV2(Module):
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 aux_logits=True,
                 init_weight=True):
        super(InceptionV2, self).__init__()

        self.aux_logits = aux_logits

        self.in_block = Sequential(
            Conv2dBN(in_channels, 32, (3, 3), stride=2, padding=0),
            Conv2dBN(32, 32, (3, 3), stride=1, padding=0),
            Conv2dBN(32, 64, (3, 3), stride=1, padding=1),
            MaxPool2d((3, 3), stride=2, padding=0),
            Conv2dBN(64, 80, (3, 3), stride=1, padding=0),
            Conv2dBN(80, 192, (3, 3), stride=2, padding=1),
            Conv2dBN(192, 288, (3, 3), stride=1, padding=1)
        )

        self.inception_block_1 = Sequential(
            InceptionModuleA(288, 64, 64, 64, 64, 96, 32),
            InceptionModuleA(256, 64, 64, 96, 64, 96, 64),
            InceptionModuleD(320, 128, 160, 64, 96),
        )

        self.aux_classifier_0 = InceptionAux(576, num_classes)

        self.inception_block_2 = Sequential(
            InceptionModuleB(576, 224, 64, 96, 96, 128, 128),
            InceptionModuleB(576, 192, 96, 128, 96, 128, 128),
            InceptionModuleB(576, 160, 128, 160, 128, 128, 128),
            InceptionModuleB(576, 96, 128, 192, 160, 160, 128),
        )

        self.inception_block_3 = Sequential(
            InceptionModuleD(576, 128, 192, 192, 256),
            InceptionModuleC(1024, 352, 192, 160, 160, 112, 128),
            InceptionModuleC(1024, 352, 192, 160, 192, 112, 128),
        )

        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(0.5)
        self.linear = Linear(1024, num_classes)

        if init_weight:
            self._initialize_weight()

    def forward(self, x):
        x = self.in_block(x)
        x = self.inception_block_1(x)
        aux_0 = self.aux_classifier_0(x) if self.training and self.aux_logits else None
        x = self.inception_block_2(x)
        x = self.inception_block_3(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x, aux_0 if self.aux_logits and self.training else x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_tensor = torch.randn([32, 3, 299, 299]).float()

    net = InceptionV2(in_channels=3, num_classes=5)
    net.eval()
    res = net(dummy_tensor)
