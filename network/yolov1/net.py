import torch
import torch.nn as nn
from backbone.yolov1_backbone import YoloV1Backbone
from module.conv import ConvLeakyReLU
from module.layer import Flatten


class YoloV1(nn.Module):
    def __init__(self,
                 backbone,
                 num_grids=7,
                 num_bboxes=2,
                 num_classes=20,
                 bn=True):
        super(YoloV1, self).__init__()

        self.bn = bn

        self.num_grids = num_grids
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.backbone = backbone
        self.conv_layers = self._make_conv_layer()
        self.fc_layers = self._make_fc_layer()

    def _make_conv_layer(self):
        return nn.Sequential(
            ConvLeakyReLU(1024, 1024, (3, 3), stride=1, padding=1, bn=self.bn),
            ConvLeakyReLU(1024, 1024, (3, 3), stride=1, padding=1, bn=self.bn),
        )

    def _make_fc_layer(self):
        S, B, C = self.num_grids, self.num_bboxes, self.num_classes
        return nn.Sequential(
            Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

    def forward(self, x):
        S, B, C = self.num_grids, self.num_bboxes, self.num_classes
        x = self.backbone(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, S, S, 5 * B + C)
        return x


if __name__ == '__main__':
    from torch.autograd import Variable

    backbone = YoloV1Backbone(conv_only=True, bn=True, init_weight=True)
    yolov1 = YoloV1(backbone=backbone)
    image = torch.randn(10, 3, 448, 448)
    image = Variable(image)

    output = yolov1(image)
    print(output.size())
