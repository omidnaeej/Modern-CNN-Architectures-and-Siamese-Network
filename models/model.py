import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from utils.metrics import *
from utils.visualization import *

class BlockA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockA, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3, 
            stride=1,
            padding=1,
            bias=False      
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BlockB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockB, self).__init__()

        self.block_a = BlockA(in_channels, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block_a(x)
        return self.block(x) * x

class BlockC(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(BlockC, self).__init__()
        self.block_a = BlockA(in_channels, out_channels)  # Ensure channels match
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block_a(x)  # Apply BlockA first
        batch_size, channels, _, _ = x.size()
        y = self.global_pool(x).view(batch_size, channels)  # Global pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y  # Channel-wise recalibration

class DepthwiseSeparableConv(nn.Module):
    """Implements Depthwise Separable Convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BlockD(nn.Module):
    def __init__(self, in_channels, out_channels, use_separable_conv=False):
        super(BlockD, self).__init__()
        self.use_shortcut = in_channels == out_channels

        if use_separable_conv:
            self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 Conv for channel matching when needed
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if not self.use_shortcut else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.batch_norm(x)

        if not self.use_shortcut:
            residual = self.shortcut(residual)

        x += residual  # Skip connection
        x = self.relu(x)
        return x

class BlockE(nn.Module):
    def __init__(self, in_channels, out_channels, b=4, g=8):
        super(BlockE, self).__init__()

        b_channels = out_channels // b  # Reduce dimensions for grouped conv

        # First 1x1 convolution
        self.conv1x1_1 = nn.Conv2d(in_channels, b_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(b_channels)
        self.relu = nn.ReLU(inplace=True)

        # Grouped 3x3 convolution
        self.conv3x3 = nn.Conv2d(b_channels, b_channels, kernel_size=3, padding=1, groups=in_channels // g, bias=False)
        self.bn2 = nn.BatchNorm2d(b_channels)

        # Second 1x1 convolution
        self.conv1x1_2 = nn.Conv2d(b_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Branch 2: Projection if input channels ≠ output channels
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x  # Keep original input for skip connection

        # Main Path
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3x3(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1x1_2(out)
        out = self.bn3(out)

        # Branch 2: Apply projection if in_channels ≠ out_channels
        if self.projection is not None:
            identity = self.projection(identity)

        # Branch 3: Direct skip connection if dimensions match
        out += identity  # Skip connection
        out = self.relu(out)

        return out

class BaseCNN(nn.Module):
    def __init__(self, block_config, use_separable_conv=False):
        super(BaseCNN, self).__init__()

        block_map = {'A': BlockA, 'B': BlockB, 'C': BlockC, 'D': BlockD, 'E': BlockE}

        # Initial layers based on the table
        self.initial_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Channel sizes between layers (according to table)
        channels = [128, 128, 128, 256, 256, 256, 512, 512, 512]

        # Create the 8 configurable blocks
        self.blocks = nn.ModuleList()
        for i, block in enumerate(block_config):
            in_channels = channels[i]
            out_channels = channels[i + 1] if i < len(channels)-1 else channels[i]

            if block == 'D':
                self.blocks.append(block_map[block](in_channels, out_channels, use_separable_conv))
            else:
                self.blocks.append(block_map[block](in_channels, out_channels))

            # Add MaxPool at specific locations based on table
            if i in [2, 4, 7]:
                self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(512, 10)

    def forward(self, x, test_mode=False):
        x = self.initial_layers(x)

        for block in self.blocks:
            x = block(x)  # Forward pass through each block

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        distance = torch.norm(emb1 - emb2, p=2, dim=1)
        loss = label * distance**2 + (1 - label) * torch.max(torch.tensor(0, dtype=torch.float32, device=emb1.device), torch.tensor(self.margin**2, dtype=torch.float32, device=emb1.device) - distance**2)
        return loss.mean()

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=512):
        super(SiameseNetwork, self).__init__()

        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.efficientnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.ReLU()
        )

    def forward(self, img1, img2):
        emb1 = self.fc(self.feature_extractor(img1).view(img1.size(0), -1))
        emb2 = self.fc(self.feature_extractor(img2).view(img2.size(0), -1))

        return emb1, emb2

