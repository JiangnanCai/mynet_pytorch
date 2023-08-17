import cv2
import torch.nn as nn
import torch
from module.conv import ConvLeakyReLU
from module.pool import GlobalAvgPool2d
import torch.utils.model_zoo as model_zoo


class DarkNet19(nn.Module):
    def __init__(self, pretrained=True):
        super(DarkNet19, self).__init__()
        self.conv_0 = nn.Sequential(
            ConvLeakyReLU(3, 32, (3, 3), stride=1, padding=1, bn=True),
            nn.MaxPool2d((2, 2), stride=2, padding=0)
        )

        self.conv_1 = nn.Sequential(
            ConvLeakyReLU(32, 64, (3, 3), stride=1, padding=1, bn=True),
            nn.MaxPool2d((2, 2), stride=2, padding=0)
        )

        self.conv_2 = nn.Sequential(
            ConvLeakyReLU(64, 128, (3, 3), stride=1, padding=1, bn=True),
            ConvLeakyReLU(128, 64, (1, 1), stride=1, padding=0, bn=True),
            ConvLeakyReLU(64, 128, (3, 3), stride=1, padding=1, bn=True),
            nn.MaxPool2d((2, 2), stride=2, padding=0)
        )

        self.conv_3 = nn.Sequential(
            ConvLeakyReLU(128, 256, (3, 3), stride=1, padding=1, bn=True),
            ConvLeakyReLU(256, 128, (1, 1), stride=1, padding=0, bn=True),
            ConvLeakyReLU(128, 256, (3, 3), stride=1, padding=1, bn=True),
            nn.MaxPool2d((2, 2), stride=2, padding=0)
        )

        self.conv_4 = nn.Sequential(
            ConvLeakyReLU(256, 512, (3, 3), stride=1, padding=1, bn=True),
            ConvLeakyReLU(512, 256, (1, 1), stride=1, padding=0, bn=True),
            ConvLeakyReLU(256, 512, (3, 3), stride=1, padding=1, bn=True),
            ConvLeakyReLU(512, 256, (1, 1), stride=1, padding=1, bn=True),
            ConvLeakyReLU(256, 512, (3, 3), stride=1, padding=1, bn=True),
            nn.MaxPool2d((2, 2), stride=2, padding=0)
        )

        self.conv_5 = nn.Sequential(
            ConvLeakyReLU(512, 1024, (3, 3), stride=1, padding=1, bn=True),
            ConvLeakyReLU(1024, 512, (1, 1), stride=1, padding=0, bn=True),
            ConvLeakyReLU(512, 1024, (3, 3), stride=1, padding=1, bn=True),
            ConvLeakyReLU(1024, 512, (1, 1), stride=1, padding=1, bn=True),
            ConvLeakyReLU(512, 1024, (3, 3), stride=1, padding=1, bn=True),
        )

        self.features = nn.Sequential(
            self.conv_0,
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4,
            self.conv_5
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1000, (1, 1), stride=1, padding=0, bias=True),
            GlobalAvgPool2d(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = self.classifier(x)
        return x

