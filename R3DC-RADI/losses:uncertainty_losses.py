"""
Uncertainty-related loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplaceNLLLoss(nn.Module):
    """Laplace negative log-likelihood loss for uncertainty training."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                uncertainty: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth (B, 1, H, W)
            target: Target depth (B, 1, H, W)
            uncertainty: Predicted aleatoric uncertainty sigma (B, 1, H, W)
            mask: Valid pixels mask (B, 1, H, W)
        """
        if mask is None:
            mask = torch.ones_like(pred)
        
        # Apply mask
        pred = pred * mask
        target = target * mask
        uncertainty = uncertainty * mask
        
        # Laplace NLL
        diff = torch.abs(pred - target)
        nll = diff / (uncertainty + 1e-8) + torch.log(uncertainty + 1e-8)
        
        # Apply mask and compute mean
        loss = (nll * mask).sum() / (mask.sum() + 1e-8)
        
        return loss


class ReliabilityRegularization(nn.Module):
    """Reliability regularization for training reliability heads."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, reliability: torch.Tensor, uncertainty: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Regularize reliability to be consistent with uncertainty.
        Lower uncertainty should correspond to higher reliability.
        """
        if mask is None:
            mask = torch.ones_like(reliability)
        
        # Normalize uncertainty to [0, 1]
        uncertainty_norm = 1 - torch.exp(-uncertainty / self.temperature)
        
        # Compute consistency loss
        loss = F.mse_loss(reliability * mask, uncertainty_norm * mask)
        
        return loss