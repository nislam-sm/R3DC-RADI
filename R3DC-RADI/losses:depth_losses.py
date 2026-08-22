"""
Depth-specific loss functions for R3DC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SILogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss."""
    
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth in log space (B, 1, H, W)
            target: Target depth in log space (B, 1, H, W)
            mask: Valid pixels mask (B, 1, H, W)
        """
        if mask is None:
            mask = torch.ones_like(pred)
        
        # Apply mask
        pred = pred * mask
        target = target * mask
        
        # Compute differences
        diff = pred - target
        
        # Compute squared difference and sum
        n = mask.sum()
        if n < 1:
            return torch.tensor(0.0, device=pred.device)
        
        diff2 = diff ** 2
        sum_diff = diff.sum()
        
        loss = diff2.sum() / n - 0.85 * (sum_diff ** 2) / (n ** 2)
        
        return loss


class FocalBerHuLoss(nn.Module):
    """Focal-BerHu loss for emphasizing hard pixels."""
    
    def __init__(self, gamma: float = 2.0, c_factor: float = 0.2):
        super().__init__()
        self.gamma = gamma
        self.c_factor = c_factor
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth in metric space (B, 1, H, W)
            target: Target depth in metric space (B, 1, H, W)
            mask: Valid pixels mask (B, 1, H, W)
        """
        if mask is None:
            mask = torch.ones_like(pred)
        
        # Apply mask
        pred = pred * mask
        target = target * mask
        mask = mask.bool()
        
        # Compute absolute difference
        diff = torch.abs(pred - target)
        
        # Compute c threshold
        max_diff = (diff[mask]).max()
        c = self.c_factor * max_diff
        
        # BerHu loss
        berhu = torch.where(
            diff <= c,
            diff,
            (diff ** 2 + c ** 2) / (2 * c)
        )
        
        # Focal weight
        focal_weight = (1 - torch.exp(-diff)) ** self.gamma
        
        # Apply mask and compute loss
        loss = (focal_weight * berhu * mask).sum() / mask.sum()
        
        return loss


class SSIMLoss(nn.Module):
    """Structural Similarity loss."""
    
    def __init__(self, window_size: int = 7, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
        # Precompute Gaussian window
        self.register_buffer('window', self._create_window(window_size))
        
    def _create_window(self, window_size: int):
        """Create a 2D Gaussian window."""
        sigma = window_size / 6.0
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = torch.outer(gauss, gauss)
        window = window.unsqueeze(0).unsqueeze(0)  # 1, 1, W, W
        return window
    
    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM for a batch."""
        window = self.window.to(x.device)
        
        # Pad
        pad = self.window_size // 2
        
        # Compute means
        mu_x = F.conv2d(x, window, padding=pad, groups=x.shape[1])
        mu_y = F.conv2d(y, window, padding=pad, groups=y.shape[1])
        
        # Compute variances and covariance
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x2 = F.conv2d(x ** 2, window, padding=pad, groups=x.shape[1]) - mu_x2
        sigma_y2 = F.conv2d(y ** 2, window, padding=pad, groups=y.shape[1]) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=x.shape[1]) - mu_xy
        
        # SSIM
        numerator = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        ssim = numerator / (denominator + 1e-8)
        
        return ssim.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        ssim = self._ssim(pred, target)
        return 1 - ssim


class VirtualNormalLoss(nn.Module):
    """Virtual normal loss for surface consistency."""
    
    def __init__(self, num_triplets: int = 1024):
        super().__init__()
        self.num_triplets = num_triplets
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth (B, 1, H, W)
            target: Target depth (B, 1, H, W)
            mask: Valid pixels mask (B, 1, H, W)
        """
        if mask is None:
            mask = torch.ones_like(pred)
        
        B, _, H, W = pred.shape
        
        # Sample random triplets
        # For simplicity, we'll use a fixed grid sampling
        # In practice, you'd want random sampling
        
        # Convert depth to 3D points (simplified, assumes pinhole camera)
        # This is a simplified version; in practice, you'd use proper intrinsics
        
        # For now, we'll use a simplified normal consistency loss
        # Compute gradients as proxy for surface normals
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Compute mean differences
        dx_loss = F.l1_loss(pred_dx, target_dx, reduction='mean')
        dy_loss = F.l1_loss(pred_dy, target_dy, reduction='mean')
        
        return (dx_loss + dy_loss) * 0.5


class DepthNormalConsistencyLoss(nn.Module):
    """Depth-normal consistency loss."""
    
    def __init__(self):
        super().__init__()
        
    def _compute_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from depth."""
        B, _, H, W = depth.shape
        
        # Compute gradients
        dz_dx = depth[:, :, :, 2:] - depth[:, :, :, :-2]
        dz_dy = depth[:, :, 2:, :] - depth[:, :, :-2, :]
        
        # Pad to maintain size
        dz_dx = F.pad(dz_dx, (1, 1, 0, 0))
        dz_dy = F.pad(dz_dy, (0, 0, 1, 1))
        
        # Compute normals
        normals = torch.cat([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=1)
        normals = F.normalize(normals, p=2, dim=1)
        
        return normals
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute depth-normal consistency loss."""
        if mask is None:
            mask = torch.ones_like(pred)
        
        # Compute normals
        pred_normals = self._compute_normals(pred)
        target_normals = self._compute_normals(target)
        
        # Compute dot product
        dot = (pred_normals * target_normals).sum(dim=1, keepdim=True)
        dot = torch.clamp(dot, -1, 1)
        
        # Loss: 1 - dot product
        loss = 1 - dot
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


class GradConsistencyLoss(nn.Module):
    """Gradient consistency loss."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute gradient consistency loss."""
        if mask is None:
            mask = torch.ones_like(pred)
        
        # Compute gradients
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Apply mask to gradients
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        
        # Compute L1 loss
        loss_x = (torch.abs(pred_grad_x - target_grad_x) * mask_x).sum() / (mask_x.sum() + 1e-8)
        loss_y = (torch.abs(pred_grad_y - target_grad_y) * mask_y).sum() / (mask_y.sum() + 1e-8)
        
        return (loss_x + loss_y) * 0.5


class AnchorLoss(nn.Module):
    """Sparse anchor consistency loss."""
    
    def forward(self, pred: torch.Tensor, sparse_depth: torch.Tensor, 
                sparse_mask: torch.Tensor) -> torch.Tensor:
        """Compute loss at sparse anchor locations."""
        diff = torch.abs(pred - sparse_depth)
        loss = (diff * sparse_mask).sum() / (sparse_mask.sum() + 1e-8)
        return loss