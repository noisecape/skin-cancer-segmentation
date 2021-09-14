import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from model.resnet import BasicBlock, ResNet


class JiGen(nn.Module):

    def __init__(self, P=30):
        super(JiGen, self).__init__()
        self.name = 'jigen'
        self.P = P
        self.pretext_model = JiGenPretext(P=P)
        self.segmentation_model = JiGenSegmentation()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.1)

    def forward(self, x, pretext):
        if pretext:
            return self.pretext_model(x)
        else:
            return self.segmentation_model(x)


class JiGenPretext(nn.Module):

    def __init__(self, P):
        super(JiGenPretext, self).__init__()
        self.P = P
        self.name = 'jigen'
        self.P = P
        self.model = ResNet(BasicBlock, [3, 4, 6, 3])

        self.pretext_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.P))

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.pretext_head(x)
        return x


class JiGenSegmentation(nn.Module):

    def __init__(self):
        super(JiGenSegmentation, self).__init__()
        self.name = 'jigen'
        self.model = ResNet(BasicBlock, [3, 4, 6, 3])
        self.fcn = nn.Sequential(
            # input [N x 2048 x 1 x 1]
            # output [N x 1024 x 4 x 4]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # output [N x 512 x 8 x 8]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # output [N x 256 x 16 x 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # output [N x 128 x 32 x 32]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # output [N x 1 x 64 x 64]
            # For images of 64x64
            # nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            # For images of 128x128
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # output [N x 1 x 128 x 128]
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=False)
            # no sigmoid because the loss is BCEWithLogitLoss
            # which implicitly implements the Sigmoid function.
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fcn(x)
        return x