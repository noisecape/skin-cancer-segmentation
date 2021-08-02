import random

import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision.models.resnet import resnet50


class JiGen(nn.Module):

    def __init__(self, P=30):
        super(JiGen, self).__init__()
        self.name = 'jigsaw'
        self.P = P
        self.resnet = resnet50()
        self.model = torch.nn.Sequential(self.resnet.conv1, self.resnet.bn1,
                                         self.resnet.relu, self.resnet.maxpool,
                                         self.resnet.layer1, self.resnet.layer2,
                                         self.resnet.layer3, self.resnet.layer4,
                                         self.resnet.avgpool)

        self.pretext_head = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, self.P))
        self.fcn = nn.Sequential(
            # input [N x 2048 x 1 x 1]
            # output [N x 1024 x 4 x 4]
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # output [N x 512 x 8 x 8]
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # output [N x 256 x 16 x 16]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # output [N x 128 x 32 x 32]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # output [N x 1 x 64 x 64]
            # For images of 64x64
            # nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            # For images of 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # output [N x 1 x 128 x 128]
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
            # no sigmoid because the loss is BCEWithLogitLoss
            # which implicitly implements the Sigmoid function.
        )

    def forward(self, x, pretext=False):
        x = self.model(x)
        if pretext:
            x = torch.flatten(x, 1)
            x = self.pretext_head(x)
        else:
            x = self.fcn(x)
        return x


# batch = torch.randn((64, 3, 128, 128))
# model = JiGen()
# criterion = torch.nn.CrossEntropyLoss()
# label = [random.randint(0, 29) for _ in range(64)]
# label = torch.tensor(label)
# output = model(batch)
# loss = criterion(output, label)
# print(loss)