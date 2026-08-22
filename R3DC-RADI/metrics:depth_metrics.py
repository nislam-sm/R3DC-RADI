"""
Depth evaluation metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Compute RMSE."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Compute MAE."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.mean(torch.abs(pred - target)).item()


def compute_abs_rel(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Compute absolute relative error."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.mean(torch.abs(pred - target) / (target + 1e-8)).item()


def compute_silog(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Compute scale-invariant log error."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
    n = diff.numel()
    silog = torch.sqrt(torch.mean(diff ** 2) - (torch.mean(diff) ** 2))
    return silog.item()


def compute_delta(pred: torch.Tensor, target: torch.Tensor, 
                  threshold: float, mask: torch.Tensor = None) -> float:
    """Compute threshold accuracy delta_n."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    ratio = torch.max(pred / (target + 1e-8), target / (pred + 1e-8))
    return (ratio < threshold).float().mean().item()


def compute_delta_metrics(pred: torch.Tensor, target: torch.Tensor, 
                          mask: torch.Tensor = None) -> dict:
    """Compute all delta metrics."""
    return {
        f'delta_{t}': compute_delta(pred, target, t, mask)
        for t in [1.25, 1.25 ** 2, 1.25 ** 3]
    }


def compute_irmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Compute inverse RMSE (for KITTI)."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    inv_pred = 1 / (pred + 1e-8)
    inv_target = 1 / (target + 1e-8)
    return torch.sqrt(torch.mean((inv_pred - inv_target) ** 2)).item() * 1000  # 1/km


def compute_imae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Compute inverse MAE (for KITTI)."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    inv_pred = 1 / (pred + 1e-8)
    inv_target = 1 / (target + 1e-8)
    return torch.mean(torch.abs(inv_pred - inv_target)).item() * 1000  # 1/km


def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, 
                          mask: torch.Tensor = None) -> dict:
    """Compute all depth metrics."""
    metrics = {}
    
    metrics['rmse'] = compute_rmse(pred, target, mask)
    metrics['mae'] = compute_mae(pred, target, mask)
    metrics['abs_rel'] = compute_abs_rel(pred, target, mask)
    metrics['silog'] = compute_silog(pred, target, mask)
    metrics.update(compute_delta_metrics(pred, target, mask))
    
    return metrics


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, 
                        sparse_mask: torch.Tensor = None) -> dict:
    """
    Compute all metrics including KITTI-specific ones.
    """
    # Valid pixels mask
    mask = (target > 0).float()
    if sparse_mask is not None:
        mask = mask * (1 - sparse_mask)  # Exclude sparse anchors
    
    # Standard metrics
    metrics = compute_depth_metrics(pred, target, mask)
    
    # KITTI inverse metrics
    metrics['irmse'] = compute_irmse(pred, target, mask)
    metrics['imae'] = compute_imae(pred, target, mask)
    
    return metrics