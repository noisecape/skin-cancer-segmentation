import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from model.resnet import BasicBlock, ResNet


class JiGen(nn.Module):
    """
    Implements the JiGen architecure. References: https://github.com/fmcarlucci/JigenDG.
    The model solves simultaneously the pretext and segmentation task, hence it is composed by two
    submodels that shares the same backbone architectures, while two distinctive heads are used for
    their specific task.
    """

    def __init__(self, conf, P=30):
        super(JiGen, self).__init__()
        self.name = 'jigen'
        self.P = P
        # model used for the pretext task
        self.pretext_model = JiGenPretext(P=P)
        # model used for the segmentation task
        self.segmentation_model = JiGenSegmentation()
        self.conf = conf
        # the optimizer used for JiGen changes only for experiment 1.
        # For experiment 2, 3 and 4 the same optimizer was used.
        if conf == 1:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x, pretext):
        if pretext:
            return self.pretext_model(x)
        else:
            return self.segmentation_model(x)


class JiGenPretext(nn.Module):
    """
    This class implements the submodel of the JiGen algorithm for the pretext task.
    The backbone is a Resnet-34, while the head is small neural network composed by one hidden layer.
    """

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
    """
        This class implements the submodel of the JiGen algorithm for the segmentation task.
        The backbone is a Resnet-34, while the head is FCN that produce an image of the same size of the
        one given in input, that is 128x128x3.
    """

    def __init__(self):
        super(JiGenSegmentation, self).__init__()
        self.name = 'jigen'
        self.model = ResNet(BasicBlock, [3, 4, 6, 3])
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

    def forward(self, x):
        x = self.model(x)
        x = self.fcn(x)
        return x