import torch
from torchvision.models.resnet import resnet34
import torch.nn as nn


class CustomSegmentation(nn.Module):

    def __init__(self, n_augmentations=4):
        super(CustomSegmentation, self).__init__()
        self.name = 'custom_approach'
        resnet = resnet34()
        self.n_augmentations = n_augmentations
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                      resnet.layer1, resnet.layer2, resnet.layer3,
                                      resnet.layer4, resnet.avgpool)

        self.head_pretext = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True),
                                          nn.Dropout(0.5), nn.Linear(64, self.n_augmentations))
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

    def forward(self, x, pretext=False):
        if pretext:
            x = self.backbone(x)
            x = torch.flatten(x, 1)
            return self.head_pretext(x)
        else:
            x = self.backbone(x)
            return self.fcn(x)


# model = CustomSegmentation(n_augmentations=4)
# x = torch.randn((64, 3, 128, 128))
# result = model(x, pretext=True)
# print(result)