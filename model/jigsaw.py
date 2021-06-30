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


class ContrastiveLearning(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass


x = torch.randn((1, 3, 128, 128))
model = ContextRestoration()
out = model(x)
print(out)