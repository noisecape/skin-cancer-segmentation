import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                            kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                            kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):

    def __init__(self, in_channel=3, out_channel=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.skip_connections = []
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.upscale = []
        self.features = features
        for f in features:
            self.downs.append(DoubleConv(in_channel, f))
            in_channel = f
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottom_layer = DoubleConv(self.features[-1], self.features[-1]*2)
        for f in reversed(features):
            self.ups.append(DoubleConv(f*2, f))
            self.upscale.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
        self.final_layer = nn.Conv2d(in_channels=features[0], out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        # down
        for u in self.downs:
            x = u(x)
            self.skip_connections.append(x)
            x = self.pooling(x)
        # bottom
        x = self.bottom_layer(x)
        # up
        for d, u, s in zip(self.ups, self.upscale, reversed(self.skip_connections)):
            x = u(x)
            x = torch.cat((x, s), 1)
            x = d(x)
        x = self.final_layer(x)
        return x


class ContextRestoration(nn.Module):

    def __init__(self, in_channel=3, out_channel=1):
        super(ContextRestoration, self).__init__()
        self.unet = Unet()

    def forward(self, x):
        return self.unet(x)


# x = torch.randn((1, 3, 128, 128))
# model = ContextRestoration()
# result = model(x)
# assert x.shape == (1, 3, 128, 128)
# print()
