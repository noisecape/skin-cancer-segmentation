import torch
import torch.nn as nn
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, features=[64, 128, 256, 512], name='context_restoration'):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.name = name

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv_pretext = nn.Conv2d(features[0], 3, kernel_size=1)
        self.final_conv_segmentation = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x, pretext=True):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        if pretext:
            return self.final_conv_pretext(x)
        else:
            return self.final_conv_segmentation(x)

# import torch
# import torch.nn as nn
# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
#
# class DoubleConv(nn.Module):
#
#     def __init__(self, in_channel, out_channel):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                             kernel_size=3, stride=1, padding=1, bias=False),
#                                   nn.BatchNorm2d(out_channel),
#                                   nn.ReLU(inplace=True),
#                                   nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                             kernel_size=3, stride=1, padding=1, bias=False),
#                                   nn.BatchNorm2d(out_channel),
#                                   nn.ReLU(inplace=True))
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class Unet(nn.Module):
#
#     def __init__(self, in_channel, features=[64, 128, 256, 512]):
#         super(Unet, self).__init__()
#         self.skip_connections = []
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
#         self.features = features
#         self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         for f in self.features:
#             self.downs.append(DoubleConv(in_channel, f))
#             in_channel = f
#
#         self.bottom_layer = DoubleConv(self.features[-1], self.features[-1]*2)
#         for f in reversed(features):
#             self.ups.append(nn.ConvTranspose2d(in_channels=f*2, out_channels=f,
#                                                kernel_size=2, stride=2))
#             self.ups.append(DoubleConv(in_channel=f*2, out_channel=f))
#
#         self.final_layer_pretext = nn.Conv2d(in_channels=features[0], out_channels=3, kernel_size=1)
#         self.final_layer_segmentation = nn.Conv2d(in_channels=features[0], out_channels=1, kernel_size=1)
#
#     def forward(self, x, pretext=False):
#         # down
#         for u in self.downs:
#             x = u(x)
#             self.skip_connections.append(x)
#             x = self.pooling(x)
#         # bottom
#         x = self.bottom_layer(x)
#         # up
#         self.skip_connections.reverse()
#         for idx in range(0, len(self.ups)-1, 2):
#             x = self.ups[idx](x)
#             skip_connection = self.skip_connections[idx//2]
#             x = torch.cat((x, skip_connection), 1)
#             x = self.ups[idx+1](x)
#
#         if pretext:
#             x = self.final_layer_pretext(x)
#         else:
#             x = self.final_layer_segmentation(x)
#         return x
#
#
# class ContextRestoration(nn.Module):
#
#     def __init__(self, in_channel=3):
#         super(ContextRestoration, self).__init__()
#         self.name = 'context_restoration'
#         self.unet = Unet(in_channel).to(DEVICE)
#
#     def forward(self, x, pretext=True):
#         return self.unet(x, pretext=pretext)



# Pretext sample
# x = torch.randn((64, 3, 128, 128)).to(DEVICE)
# model = ContextRestoration().to(DEVICE)
# result = model(x, pretext=True)
# criterion = torch.nn.MSELoss()
# loss = criterion(result, x)
# assert x.shape == (64, 3, 128, 128)
# print()

# Segmentation sample
# x = torch.randn((64, 3, 128, 128))
# gt = torch.randn((64, 1, 128, 128))
# model = ContextRestoration()
# prediction = model(x, pretext=False)
# criterion = torch.nn.BCEWithLogitsLoss()
# loss = criterion(prediction, gt)
# assert prediction.shape == (64, 1, 128, 128)
# print()
