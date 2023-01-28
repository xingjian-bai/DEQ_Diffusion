#%%
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import time
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import math

class ResNetLayer(nn.Module):
    """ResNet layer with GroupNorm and ReLU activation"""
    def __init__(self, n_channels, n_inner_channels, kernel_size=2, num_groups=8):
        """
        Args:
            n_channels: number of input channels
            n_inner_channels: number of channels in the inner convolution
            kernel_size: kernel size of the inner convolution
            num_groups: number of groups for GroupNorm
        """
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))
    # we will choose n_channels in the above layer to be smaller than n_inner_channels