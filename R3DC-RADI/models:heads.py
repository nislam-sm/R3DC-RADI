"""
Output heads for R3DC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """Depth prediction head."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)  # Output in [0, 1] in log space
        return x


class ReliabilityHead(nn.Module):
    """Reliability prediction head."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)  # Output in [0, 1]
        return x


class UncertaintyHead(nn.Module):
    """Aleatoric uncertainty prediction head."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.softplus(x)  # Positive uncertainty
        return x


class AuxiliaryHead(nn.Module):
    """Auxiliary depth head for deep supervision."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x