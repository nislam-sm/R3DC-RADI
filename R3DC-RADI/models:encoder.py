"""
Encoder modules for R3DC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ResidualBlock(nn.Module):
    """Pre-activation residual block with DropPath."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dropout_rate: float = 0.1, use_dcn: bool = False):
        super().__init__()
        self.stride = stride
        self.use_dcn = use_dcn
        
        # First convolution
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = self._make_conv(in_channels, out_channels, stride, use_dcn)
        
        # Second convolution
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = self._make_conv(out_channels, out_channels, 1, use_dcn)
        
        # Skip connection
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            if use_dcn:
                # Note: DCN skip is not standard - use regular conv
                self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        
        self.dropout_rate = dropout_rate
        self._reset_parameters()
    
    def _make_conv(self, in_c: int, out_c: int, stride: int, use_dcn: bool):
        if use_dcn:
            return DeformConv2d(in_c, out_c, 3, padding=1, stride=stride)
        return nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride)
    
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize
            return x.div(keep_prob) * random_tensor
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        
        if self.skip is not None:
            identity = self.skip(x)
        
        out = out + identity
        return self.drop_path(out)


class RGBEncoder(nn.Module):
    """RGB encoder with residual blocks."""
    
    def __init__(self, base_width: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.base = base_width
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width // 2, 3, padding=1),
            nn.BatchNorm2d(base_width // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width // 2, base_width // 2, 3, padding=1),
            nn.BatchNorm2d(base_width // 2),
            nn.ReLU(inplace=True),
        )
        
        # Encoder stages
        self.stage1 = self._make_stage(base_width // 2, base_width, 1, dropout_rate, use_dcn=False)
        self.stage2 = self._make_stage(base_width, base_width * 2, 2, dropout_rate, use_dcn=False)
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, 2, dropout_rate, use_dcn=False)
        
    def _make_stage(self, in_c: int, out_c: int, stride: int, dropout: float, use_dcn: bool):
        blocks = []
        blocks.append(ResidualBlock(in_c, out_c, stride, dropout, use_dcn))
        blocks.append(ResidualBlock(out_c, out_c, 1, dropout, use_dcn))
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        
        x = self.stem(x)
        features.append(x)  # scale 1
        
        x = self.stage1(x)
        features.append(x)  # scale 1/2
        
        x = self.stage2(x)
        features.append(x)  # scale 1/4
        
        x = self.stage3(x)
        features.append(x)  # scale 1/8
        
        return features


class DepthEncoder(nn.Module):
    """Depth encoder with deformable convolutions."""
    
    def __init__(self, base_width: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.base = base_width
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(2, base_width // 2, 3, padding=1),
            nn.BatchNorm2d(base_width // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width // 2, base_width // 2, 3, padding=1),
            nn.BatchNorm2d(base_width // 2),
            nn.ReLU(inplace=True),
        )
        
        # Encoder stages with DCN
        self.stage1 = self._make_stage(base_width // 2, base_width, 1, dropout_rate, use_dcn=True)
        self.stage2 = self._make_stage(base_width, base_width * 2, 2, dropout_rate, use_dcn=True)
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, 2, dropout_rate, use_dcn=True)
        
    def _make_stage(self, in_c: int, out_c: int, stride: int, dropout: float, use_dcn: bool):
        blocks = []
        blocks.append(ResidualBlock(in_c, out_c, stride, dropout, use_dcn))
        blocks.append(ResidualBlock(out_c, out_c, 1, dropout, use_dcn))
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        
        x = self.stem(x)
        features.append(x)  # scale 1
        
        x = self.stage1(x)
        features.append(x)  # scale 1/2
        
        x = self.stage2(x)
        features.append(x)  # scale 1/4
        
        x = self.stage3(x)
        features.append(x)  # scale 1/8
        
        return features