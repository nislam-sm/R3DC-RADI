"""
Decoder modules for R3DC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from .attention import CrossModalAttention, CBAM
from .encoder import ResidualBlock


class EfficientUpBlock(nn.Module):
    """Efficient FPN decoder block."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 skip_channels: int, rgb_channels: int,
                 dropout_rate: float = 0.1, num_heads: int = 4,
                 max_tokens: int = 512):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # DCN fusion with skip
        self.fusion = DeformConv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        
        # Residual block
        self.res_block = ResidualBlock(out_channels, out_channels, 1, dropout_rate, use_dcn=False)
        
        # CBAM
        self.cbam = CBAM(out_channels)
        
        # Cross-modal attention
        self.cma = CrossModalAttention(out_channels, num_heads, max_tokens)
        
        # Skip channel matching
        self.skip_conv = None
        if skip_channels != out_channels:
            self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Match spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Match skip channels
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
        
        # Concatenate and fuse
        fused = torch.cat([x, skip], dim=1)
        out = self.fusion(fused)
        
        # Residual block
        out = self.res_block(out)
        
        # CBAM
        out = self.cbam(out)
        
        # Cross-modal attention
        out = self.cma(out, rgb)
        
        return out


class FPNDecoder(nn.Module):
    """FPN decoder with EfficientUpBlocks."""
    
    def __init__(self, base_width: int = 64, dropout_rate: float = 0.1,
                 num_heads: int = 4, max_tokens: int = 512):
        super().__init__()
        self.base = base_width
        
        # EfficientUpBlocks for each scale (from coarse to fine)
        self.block4 = EfficientUpBlock(
            base_width * 4, base_width * 4,
            base_width * 4, base_width * 4,
            dropout_rate, num_heads, max_tokens
        )
        
        self.block3 = EfficientUpBlock(
            base_width * 4, base_width * 2,
            base_width * 2, base_width * 2,
            dropout_rate, num_heads, max_tokens
        )
        
        self.block2 = EfficientUpBlock(
            base_width * 2, base_width,
            base_width, base_width,
            dropout_rate, num_heads, max_tokens
        )
        
        self.block1 = EfficientUpBlock(
            base_width, base_width // 2,
            base_width // 2, base_width // 2,
            dropout_rate, num_heads // 2, max_tokens
        )
        
        # Final projection
        self.final_proj = nn.Conv2d(base_width // 2, base_width // 2, 1)
        
    def forward(self, features: dict) -> torch.Tensor:
        """
        Args:
            features: dict with 'rgb' and 'depth' features at each scale
        Returns:
            Full-resolution decoder features
        """
        rgb_feats = features['rgb']  # List of features at different scales
        depth_feats = features['depth']
        
        # Bottom-up features (coarse to fine)
        # features at scale 1/8, 1/4, 1/2, 1
        # rgb_feats: [stem, scale1, scale2, scale3]
        # depth_feats: [stem, scale1, scale2, scale3]
        
        # Block 4: scale 1/8 -> 1/4
        x = depth_feats[3]  # scale 1/8
        rgb = rgb_feats[2]  # scale 1/4
        skip = depth_feats[2]  # scale 1/4
        x = self.block4(x, skip, rgb)
        dec4 = x
        
        # Block 3: scale 1/4 -> 1/2
        rgb = rgb_feats[1]  # scale 1/2
        skip = depth_feats[1]  # scale 1/2
        x = self.block3(x, skip, rgb)
        dec3 = x
        
        # Block 2: scale 1/2 -> 1
        rgb = rgb_feats[0]  # scale 1
        skip = depth_feats[0]  # scale 1
        x = self.block2(x, skip, rgb)
        dec2 = x
        
        # Block 1: scale 1 -> full
        rgb = rgb_feats[0]  # scale 1
        skip = depth_feats[0]  # scale 1
        x = self.block1(x, skip, rgb)
        dec1 = x
        
        # Final projection
        out = self.final_proj(dec1)
        
        return out