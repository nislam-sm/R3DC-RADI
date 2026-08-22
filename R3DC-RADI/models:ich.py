"""
Indoor Calibration Head (ICH) for adapting foundation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IndoorCalibrationHead(nn.Module):
    """Lightweight adapter for calibrating foundation model priors."""
    
    def __init__(self, in_channels: int, hidden_channels: int = 128, 
                 out_channels: int = 2):  # scale and shift
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, out_channels),
        )
        
    def forward(self, features: torch.Tensor) -> tuple:
        """
        Args:
            features: Pooled features from foundation model
        Returns:
            scale: Scale factor (global)
            shift: Shift factor (global)
        """
        # Global average pooling if features are spatial
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Predict scale and shift
        out = self.mlp(features)  # (B, 2)
        scale = out[:, 0:1]  # (B, 1)
        shift = out[:, 1:2]  # (B, 1)
        
        return scale, shift


class FoundationAdapter(nn.Module):
    """Adapter for freezing foundation models and adding ICH."""
    
    def __init__(self, foundation_model: nn.Module, 
                 feature_channels: int = 384,  # ViT-S small
                 hidden_channels: int = 128):
        super().__init__()
        self.foundation = foundation_model
        self.ich = IndoorCalibrationHead(feature_channels, hidden_channels)
        
        # Freeze foundation model
        for param in self.foundation.parameters():
            param.requires_grad = False
            
    def forward(self, rgb: torch.Tensor) -> tuple:
        """
        Args:
            rgb: RGB image (B, 3, H, W)
        Returns:
            scale: Scale factor
            shift: Shift factor
            features: Foundation features (for subsequent use)
        """
        # Get foundation features
        features = self.foundation.get_intermediate_features(rgb)
        
        # Apply ICH
        scale, shift = self.ich(features)
        
        return scale, shift, features