import torch
import torchvision
import torch.nn as nn
from torchvision.models.resnet import resnet50


class SimCLR(nn.Module):

    def __init__(self, out_dim=2048, fine_tune=False):
        super(SimCLR, self).__init__()
        resnet = resnet50()
        self.fine_tune = fine_tune
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                      resnet.layer1, resnet.layer2, resnet.layer3,
                                      resnet.layer4, resnet.avgpool)
        # used for pretext task
        self.pretext_head = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        # used for segmentation task
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
            # output [N x 64 x 64 x 64]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # output [N x 1 x 128 x 128]
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            # no sigmoid because the loss is BCEWithLogitLoss
            # which implicitly implements the Sigmoid function.
        )

    def forward(self, x):
        if self.fine_tune:
            x = self.backbone(x)
            output = self.fcn(x)
            return output
        else:
            x = self.backbone(x)
            x = torch.flatten(x, 1)
            output = self.pretext_head(x)
            return output

# x = torch.randn((1, 3, 128, 128))
# gt = torch.randn((1, 1, 128, 128))
# model = SimCLR(out_dim=2048, fine_tune=True)
# result = model(x)
# criterion = nn.BCEWithLogitsLoss()
# loss = criterion(result, gt)
# print(loss)
