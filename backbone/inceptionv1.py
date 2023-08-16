from torch.nn import (Module, Sequential, functional, Dropout,
                      LeakyReLU, Linear, ReLU,
                      Conv2d, BatchNorm2d,
                      AvgPool2d, MaxPool2d, AdaptiveAvgPool2d)

import torch
from typing import Union, Tuple


class Conv2dBasic(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[str, int, Tuple[int, int]],
                 stride: int = 1,
                 activation=ReLU(inplace=True)):
        super(Conv2dBasic, self).__init__()

        self.conv_basic = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation
        )

    def forward(self, x):
        return self.conv_basic(x)


class Stem(Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.stem = Sequential(
            Conv2dBasic(in_channels, 64, (7, 7), stride=2, padding=3),
            MaxPool2d((3, 3), stride=2, ceil_mode=True),
            Conv2dBasic(64, 64, (1, 1), stride=1, padding=0),  # 有什么用吗？尺寸和通道数完全没有变化，只是加多了一个激活函数？
            Conv2dBasic(64, 192, (3, 3), stride=1, padding=1),  # 这里和论文有些不一样
            MaxPool2d((3, 3), stride=2, ceil_mode=True)
        )

    def forward(self, x):
        return self.stem(x)


class InceptionV1Block(Module):
    def __init__(self,
                 in_channels,
                 out_channels_1x1,
                 mid_channels_3x3,
                 out_channels_3x3,
                 mid_channels_5x5,
                 out_channels_5x5,
                 pool_channels):
        super(InceptionV1Block, self).__init__()

        self.branch_1x1 = Conv2dBasic(in_channels, out_channels_1x1, (1, 1), stride=1, padding=0)

        self.branch_3x3 = Sequential(
            Conv2dBasic(in_channels, mid_channels_3x3, (1, 1), stride=1, padding=0),
            Conv2dBasic(mid_channels_3x3, out_channels_3x3, (3, 3), stride=1, padding=1)
        )

        self.branch_5x5 = Sequential(
            Conv2dBasic(in_channels, mid_channels_5x5, (1, 1), stride=1, padding=0),
            Conv2dBasic(mid_channels_5x5, out_channels_5x5, (5, 5), stride=1, padding=2)
        )

        self.branch_pool = Sequential(
            MaxPool2d((3, 3), stride=1, padding=1),
            Conv2dBasic(in_channels, pool_channels, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x), self.branch_3x3(x), self.branch_5x5(x), self.branch_pool(x)], 1)


class InceptionV1Aux(Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionV1Aux, self).__init__()
        self.avg_pool = AvgPool2d((5, 5), stride=3)
        self.conv = Conv2dBasic(in_channels, 128, (1, 1), stride=1, padding=0)

        self.fc_1 = Linear(2048, 1024)
        self.fc_2 = Linear(1024, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)

        x = torch.flatten(x, 1)
        x = functional.dropout(x, 0.5, training=self.training)  # todo: self.training 在哪里

        x = functional.relu(self.fc_1(x), inplace=True)
        x = functional.dropout(x, 0.5, training=self.training)

        x = self.fc_2(x)
        return x


class InceptionV1(Module):
    def __init__(self,
                 in_channels,
                 num_classes=1000,
                 aux_logits=True,
                 init_weight=False):
        super(InceptionV1, self).__init__()

        self.aux_logits = aux_logits

        self.in_block = Stem(in_channels)

        self.mix_block_1 = Sequential(
            InceptionV1Block(192, 64, 96, 128, 16, 32, 32),
            InceptionV1Block(256, 128, 128, 192, 32, 96, 64),
            MaxPool2d((3, 3), stride=2, padding=1),
            InceptionV1Block(480, 192, 96, 208, 16, 48, 64)
        )
        self.aux_classifier_1 = InceptionV1Aux(512, num_classes)
        self.mix_block_2 = Sequential(
            InceptionV1Block(512, 160, 112, 224, 24, 64, 64),
            InceptionV1Block(512, 128, 128, 256, 24, 64, 64),
            InceptionV1Block(512, 112, 144, 288, 32, 64, 64)
        )
        self.aux_classifier_2 = InceptionV1Aux(528, num_classes)
        self.mix_block_3 = Sequential(
            InceptionV1Block(528, 256, 160, 320, 32, 128, 128),
            MaxPool2d((3, 3), stride=2, ceil_mode=True),
            InceptionV1Block(832, 256, 160, 320, 32, 128, 128),
            InceptionV1Block(832, 384, 192, 384, 48, 128, 128)
        )

        # self.out_block
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(0.4)
        self.fc = Linear(1024, num_classes)

        if init_weight:
            self._initialize_weight()

    def forward(self, x):
        # in block (stem) ----------------------------------------------------------------------------------------------
        x = self.in_block(x)
        # mix block (Inception part) -----------------------------------------------------------------------------------
        x = self.mix_block_1(x)
        aux_1 = self.aux_classifier_1(x) if self.training and self.aux_logits else None
        x = self.mix_block_2(x)
        aux_2 = self.aux_classifier_2(x) if self.training and self.aux_logits else None
        x = self.mix_block_3(x)
        # out block ----------------------------------------------------------------------------------------------------
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)  # batch, num_classes
        return x, aux_2, aux_1 if self.training and self.aux_logits else x

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

    net = InceptionV1(in_channels=3, num_classes=5)
    net.eval()
    res = net(dummy_tensor)
