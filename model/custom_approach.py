import torch
import torch.nn as nn
from model.resnet import ResNet, BasicBlock


class CustomSegmentation(nn.Module):
    """
    Implements the Personal Model. The backbone architecture is the ResNet-34, implemented by using a customized
    version of the pytorch's Resnet implementation.
    """

    def __init__(self, n_augmentations=4):
        super(CustomSegmentation, self).__init__()
        self.name = 'custom_approach'
        self.n_augmentations = n_augmentations
        self.backbone = ResNet(BasicBlock, [3, 4, 6, 3])
        # the head used for the pretext task
        self.head_pretext = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True),
                                          nn.Dropout(0.5), nn.Linear(64, self.n_augmentations))
        # the head used for the segmentation task
        self.fcn = nn.Sequential(
            # input [N x 512 x 1 x 1]
            # output [N x 256 x 4 x 4]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # output [N x 128 x 8 x 8]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # output [N x 64 x 16 x 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # output [N x 32 x 32 x 32]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # output [N x 16 x 64 x 64]
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