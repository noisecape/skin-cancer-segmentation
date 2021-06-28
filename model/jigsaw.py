import torch
import torch.nn as nn
import torchvision
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class JiGen(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass


class ContextRestoration(nn.Module):

    def __init__(self, in_channel=3, out_channel=1):
        super(ContextRestoration, self).__init__()
        self.skip_connections = []
        self.n_features = [64, 128, 256, 512]
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.double_conv = nn.Sequential(
            # Double Convolutions
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottom_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.reconstruction = nn.Sequential(
            # Transpose Convolution
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
            # Double Convolution
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Transpose Convolution
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            # Double Convolutions
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Transpose Convolution
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            # Double Convolutions
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Transpose Convolutions
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            # Double Convolutions
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        # downstream
        for f in self.n_features:
            x = self.double_conv(x)
            self.skip_connections.append(x)
            x = self.max_pool(x)
        x = self.bottom_conv(x)
        # upstream
        for f in reversed(self.n_features):
            pass
        print(x)
        return x


class ContrastiveLearning(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass


x = torch.randn((1, 3, 128, 128))
model = ContextRestoration()
out = model(x)
print(out)