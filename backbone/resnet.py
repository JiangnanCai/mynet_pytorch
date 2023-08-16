import torch
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, init, functional
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()

        self.block_residual = Sequential(
            Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
            BatchNorm2d(out_channels),
        )
        self.block_shortcut = Sequential()

        self.relu_final = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.block_residual(x)
        shortcut = self.downsample(x) if self.downsample else self.block_shortcut(x)
        return self.relu_final(residual + shortcut)


class Bottleneck(Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None):
        super().__init__()
        self.block_residual = Sequential(
            Conv2d(in_channels, out_channels, (1, 1), stride=1, padding=0),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, (3, 3), stride=stride, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels * self.expansion, (1, 1), stride=1, padding=0),
            BatchNorm2d(out_channels * self.expansion),
        )
        self.block_shortcut = Sequential()

        self.final_relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.block_residual(x)
        shortcut = self.downsample(x) if self.downsample else self.block_shortcut(x)
        return self.final_relu(residual + shortcut)


class BottleneckDetNet(Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 block_type='A'):
        super(BottleneckDetNet, self).__init__()
        self.block_residual = Sequential(
            Conv2d(in_channels, out_channels, (1, 1), stride=1, padding=0, bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, (3, 3), stride=stride, padding=2, bias=False, dilation=2),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, self.expansion * out_channels, (1, 1), stride=1, padding=0, bias=False),
            BatchNorm2d(self.expansion * out_channels)
        )

        self.downsample = Sequential(
            Conv2d(in_channels, self.expansion * out_channels, (1, 1), stride=stride, padding=0, bias=False),
            BatchNorm2d(self.expansion * out_channels)
        ) if stride != 1 or in_channels != self.expansion * out_channels or block_type == 'B' else Sequential()

    def forward(self, x):
        return functional.relu(self.block_residual(x) + self.downsample(x))


class ResNet(Module):
    channels = 64

    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_block = Sequential(
            Conv2d(3, 64, (7, 7), stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d((3, 3), stride=2, padding=1)
        )

        self.layer_1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer_5 = ResNet._make_detnet_layer(512)  # why 2048?

        self.out_block = Sequential(
            Conv2d(256, 30, (3, 3), stride=1, padding=0, bias=False),
            BatchNorm2d(30),
            ReLU(inplace=True),
            Conv2d(30, 30, (3, 3), stride=2, padding=1, bias=False),
            BatchNorm2d(30),
        )

        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, channels, num_block, stride=1):

        downsample = Sequential(
            Conv2d(self.channels, channels * block.expansion, (1, 1), stride=stride, padding=0),
            BatchNorm2d(channels * block.expansion)
        ) if stride != 1 or self.channels != channels * block.expansion else None

        layers = [block(self.channels, channels, stride, downsample)]
        self.channels = channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.channels, channels))
        return Sequential(*layers)

    @staticmethod
    def _make_detnet_layer(in_channels):
        return Sequential(
            BottleneckDetNet(in_channels, 256, block_type='B'),
            BottleneckDetNet(256, 256, block_type='A'),
            BottleneckDetNet(256, 256, block_type='A'),
        )

    def forward(self, x):

        x = self.in_block(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.out_block(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model


if __name__ == '__main__':
    device = torch.device("cuda")
    model = resnet34().to(device)
    dummy_input = torch.randn(5, 3, 512, 512).to(device)
    model.eval()
    output = model(dummy_input)
    print(output.size())
