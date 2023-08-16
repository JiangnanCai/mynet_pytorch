import math
import torch.nn
from torch.nn import functional, init, Module, Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear
from torch.utils import model_zoo
import torch
#
# __all__ = ['xception']
#
# model_url = {
#     'xception': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
# }


class SeparableConv2d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1),
                 stride=1,
                 padding=0,
                 bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv = Conv2d(in_channels=in_channels, out_channels=in_channels,  # todo: 输入输出都是in_channels
                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                     groups=in_channels, bias=bias)
        self.pointwise_conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(1, 1), stride=1, padding=0,
                                     groups=1, bias=bias)
        # todo: 为什么不是先pointwise再depthwise

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class EntryFLow(Module):
    def __init__(self):
        super(EntryFLow, self).__init__()
        self.conv_1 = Sequential(
            Conv2d(3, 32, (3, 3), stride=1, padding=1, bias=False),
            BatchNorm2d(32),
            ReLU(inplace=True)
        )

        self.conv_2 = Sequential(
            Conv2d(32, 64, (3, 3), stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True)
        )

        self.block_1_residual = Sequential(
            SeparableConv2d(64, 128, (3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            SeparableConv2d(128, 128, (3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            MaxPool2d((3, 3), stride=2, padding=1)
        )

        self.block_1_shortcut = Sequential(
            Conv2d(64, 128, (1, 1), stride=2, padding=0),
            BatchNorm2d(128)
        )

        self.block_2_residual = Sequential(
            ReLU(inplace=True),
            SeparableConv2d(128, 256, (3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            SeparableConv2d(256, 256, (3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            MaxPool2d((3, 3), stride=2, padding=1)
        )

        self.block_2_shortcut = Sequential(
            Conv2d(128, 256, (1, 1), stride=2, padding=0),
            BatchNorm2d(256)
        )

        self.block_3_residual = Sequential(
            ReLU(inplace=True),
            SeparableConv2d(256, 728, (3, 3), stride=1, padding=1),
            BatchNorm2d(728),
            ReLU(inplace=True),
            SeparableConv2d(728, 728, (3, 3), stride=1, padding=1),
            BatchNorm2d(728),
            MaxPool2d((3, 3), stride=1, padding=1)  # no down-sampling
        )

        self.block_3_shortcut = Sequential(
            Conv2d(256, 728, (1, 1), stride=1, padding=0),  # no down-sampling
            BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.block_1_residual(x) + self.block_1_shortcut(x)
        x = self.block_2_residual(x) + self.block_2_shortcut(x)
        x = self.block_3_residual(x) + self.block_3_shortcut(x)
        return x


class MiddleFlowBlock(Module):
    def __init__(self):
        super(MiddleFlowBlock, self).__init__()

        self.shortcut = Sequential()
        self.residual = Sequential(
            ReLU(inplace=True),
            SeparableConv2d(728, 728, (3, 3), stride=1, padding=1),
            BatchNorm2d(728),
            ReLU(inplace=True),
            SeparableConv2d(728, 728, (3, 3), stride=1, padding=1),
            BatchNorm2d(728),
            ReLU(inplace=True),
            SeparableConv2d(728, 728, (3, 3), stride=1, padding=1),
            BatchNorm2d(728),
        )

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)


class MiddleFlow(Module):
    def __init__(self):
        super(MiddleFlow, self).__init__()
        self.middle_block = MiddleFlow._make_flow(8)

    @staticmethod
    def _make_flow(repeats):
        flows = [MiddleFlowBlock() for _ in range(repeats)]
        return Sequential(*flows)

    def forward(self, x):
        return self.middle_block(x)


class ExitFlow(Module):
    def __init__(self):
        super(ExitFlow, self).__init__()
        self.block_1_residual = Sequential(
            ReLU(inplace=True),
            SeparableConv2d(728, 728, (3, 3), stride=1, padding=1),
            BatchNorm2d(728),
            ReLU(inplace=True),
            SeparableConv2d(728, 1024, (3, 3), stride=1, padding=1),
            BatchNorm2d(1024),
            MaxPool2d((3, 3), stride=2, padding=1),
        )

        self.block_1_shortcut = Sequential(
            Conv2d(728, 1024, (1, 1), stride=2, padding=0),
            BatchNorm2d(1024)
        )

        self.block_2 = Sequential(
            SeparableConv2d(1024, 1536, (3, 3), stride=1, padding=1),
            BatchNorm2d(1536),
            ReLU(inplace=True),
            SeparableConv2d(1536, 2048, (3, 3), stride=1, padding=1),
            BatchNorm2d(2048),
            ReLU(inplace=True),
        )

        self.avg_pool = AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.block_1_residual(x) + self.block_1_shortcut(x)
        x = self.block_2(x)
        return self.avg_pool(x)


class Xception(Module):
    def __init__(self, num_classes):
        super(Xception, self).__init__()
        self.entry_flow = EntryFLow()
        self.middle_flow = MiddleFlow()
        self.exit_flow = ExitFlow()

        self.fc = Linear(2048, num_classes)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_tensor = torch.randn([32, 3, 299, 299]).float()

    net = Xception(num_classes=5)
    net.eval()
    res = net(dummy_tensor)
