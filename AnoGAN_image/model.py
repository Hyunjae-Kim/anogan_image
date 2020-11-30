import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torchvision import transforms, datasets
import torch.nn.init as init

class Generator(nn.Module):
    def __init__(self, lat_num=100, channel=128):
        super(Generator, self).__init__()
        
        self.lat_num = lat_num
        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(lat_num, channel*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel*8),
            nn.ReLU())
        self.tp_conv2 = nn.Sequential(
            nn.ConvTranspose2d(channel*8, channel*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel*4),
            nn.ReLU())
        self.tp_conv3 = nn.Sequential(
            nn.ConvTranspose2d(channel*4, channel*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel*2),
            nn.ReLU())
        self.tp_conv4 = nn.Sequential(
            nn.ConvTranspose2d(channel*2, channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU())

        self.tp_conv5 = nn.ConvTranspose2d(channel, 1, kernel_size=4, stride=2, padding=1, bias=False)
        
    def forward(self, noise):
        noise = noise.view(-1, self.lat_num, 1, 1)
        h = self.tp_conv1(noise)
        h = self.tp_conv2(h)
        h = self.tp_conv3(h)
        h = self.tp_conv4(h)
        h = self.tp_conv5(h)
        h = torch.tanh(h)    ## on
        return h

class Discriminator(nn.Module):
    def __init__(self, channel=128):
        super(Discriminator, self).__init__()
        
        self.channel:int = channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channel*2),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel*2, channel*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channel*4),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel*4, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(negative_slope=0.2))
        
    def forward(self, x, _return_activations=False):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        feature = h.view(h.size()[0], -1)

        output = F.avg_pool2d(h, kernel_size=h.size()[2:]).view(h.size()[0], -1)
        return output, feature
