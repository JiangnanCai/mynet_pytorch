import torch.nn as nn
from module.conv import ConvLeakyReLU
from module.layer import Squeeze


class YoloV1Backbone(nn.Module):
    # 这个backbone只是受到googlenet的启发，其实除了在yolov1会用，别的地方都不会用
    def __init__(self,
                 conv_only=False,
                 bn=True,
                 init_weight=True):
        super(YoloV1Backbone, self).__init__()

        self.conv_only = conv_only
        self.bn = bn

        self.features = self._make_conv_layer()

        # fc layer
        if not self.conv_only:
            self.fc = nn.Sequential(
                nn.AvgPool2d((7, 7), stride=None, padding=0),
                Squeeze(),
                nn.Linear(1024, 1000)
            )

        if init_weight:
            self._initialize_weights()

    def _make_conv_layer(self):
        return nn.Sequential(
            ConvLeakyReLU(3, 64, (7, 7), stride=2, padding=3, bn=self.bn),
            nn.MaxPool2d((2, 2), stride=None, padding=0),

            ConvLeakyReLU(64, 192, (3, 3), stride=1, padding=1, bn=self.bn),
            nn.MaxPool2d((2, 2), stride=None, padding=0),

            ConvLeakyReLU(192, 128, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(128, 256, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(256, 256, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(256, 512, (3, 3), stride=1, padding=1, bn=self.bn),
            nn.MaxPool2d((2, 2), stride=None, padding=0),

            ConvLeakyReLU(512, 256, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(256, 512, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(512, 256, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(256, 512, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(512, 256, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(256, 512, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(512, 256, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(256, 512, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(512, 512, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(512, 1024, (3, 3), stride=1, padding=1, bn=self.bn),
            nn.MaxPool2d((2, 2), stride=None, padding=0),

            ConvLeakyReLU(1024, 512, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(512, 1024, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(1024, 512, (1, 1), stride=1, padding=0, bn=self.bn),
            ConvLeakyReLU(512, 1024, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(1024, 1024, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(1024, 1024, (3, 3), stride=2, padding=1, bn=self.bn),
        )

    def forward(self, x):
        x = self.features(x)
        if not self.conv_only:
            x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
