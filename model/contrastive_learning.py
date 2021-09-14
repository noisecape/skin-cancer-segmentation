import torch
import torchvision
import torch.nn as nn
from model.resnet import ResNet, BasicBlock
import os
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SimCLR(nn.Module):

    def __init__(self, out_dim=512):
        super(SimCLR, self).__init__()
        self.name = 'contrastive_learning'
        self.backbone = ResNet(BasicBlock, [3, 4, 6, 3])
        # used for pretext task
        self.pretext_head = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        # used for segmentation task
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

    def augment_data(self, img):
        augmenter = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((img.shape[2],
                                                                                              img.shape[2])),
                                                    torchvision.transforms.ColorJitter(0.9, 0.9, 0.9, 0.5),
                                                    torchvision.transforms.GaussianBlur(5, sigma=(1.5, 3.5))])
        augmented_1 = augmenter(img)
        augmented_2 = augmenter(img)
        # self.visualize_image(augmented_1)
        # self.visualize_image(augmented_2)
        return augmented_1, augmented_2

    def visualize_image(self, x):
        x = x[0].cpu()
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()

    def forward(self, x, pretext=False):
        if pretext:
            emb_1, emb_2 = self.augment_data(x)

            emb_1 = self.backbone(emb_1)
            emb_1 = torch.flatten(emb_1, 1)
            emb_1 = self.pretext_head(emb_1)

            emb_2 = self.backbone(emb_2)
            emb_2 = torch.flatten(emb_2, 1)
            emb_2 = self.pretext_head(emb_2)
            return emb_1, emb_2
        else:
            x = self.backbone(x)
            output = self.fcn(x)
            return output

# pretext sample
# x = torch.randn((64, 3, 128, 128))
# model = SimCLR(out_dim=512)
# emb_1, emb_2 = model(x, pretext=True)
# criterion = ContrastiveLoss()
# loss = criterion(emb_1, emb_2)
# print(loss)
#
# # segmentation sample
# x = torch.randn((64, 3, 128, 128))
# gt = torch.randn((64, 1, 128, 128))
# model = SimCLR(out_dim=512)
# prediction = model(x, pretext=False)
# criterion = torch.nn.BCEWithLogitsLoss()
# loss = criterion(prediction, gt)
# print(loss)
