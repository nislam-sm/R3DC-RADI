"""
RADI: Reliability-Aware Depth Index evaluation protocol.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr


class RADI:
    """Reliability-Aware Depth Index."""
    
    def __init__(self, num_calibration_bins: int = 15, tolerance: float = 0.10):
        self.num_bins = num_calibration_bins
        self.tolerance = tolerance
        
    def compute_rec(self, reliability: torch.Tensor, error: torch.Tensor, 
                    mask: torch.Tensor = None) -> tuple:
        """
        Compute Reliability-Error Correlation (REC).
        
        Args:
            reliability: Per-pixel reliability (B, 1, H, W) or flattened
            error: Absolute error (B, 1, H, W) or flattened
            mask: Valid pixels mask (B, 1, H, W) or flattened
        
        Returns:
            spearman_rho: Spearman rank correlation
            p_value: Statistical significance
        """
        if mask is not None:
            reliability = reliability[mask]
            error = error[mask]
        
        # Flatten
        reliability = reliability.flatten().cpu().numpy()
        error = error.flatten().cpu().numpy()
        
        # Compute Spearman correlation
        # Note: We correlate reliability with negative error
        # Higher reliability should correspond to lower error
        rho, p_value = spearmanr(reliability, -error)
        
        return rho, p_value
    
    def compute_rbs(self, depth0: torch.Tensor, depth1: torch.Tensor, 
                    target: torch.Tensor, mask: torch.Tensor = None) -> float:
        """
        Compute Revision Benefit Score (RBS).
        
        Args:
            depth0: Coarse depth prediction
            depth1: Refined depth prediction
            target: Ground truth depth
            mask: Valid pixels mask
        
        Returns:
            rbs: Percentage improvement in RMSE
        """
        if mask is not None:
            depth0 = depth0[mask]
            depth1 = depth1[mask]
            target = target[mask]
        
        rmse0 = torch.sqrt(torch.mean((depth0 - target) ** 2))
        rmse1 = torch.sqrt(torch.mean((depth1 - target) ** 2))
        
        rbs = ((rmse0 - rmse1) / (rmse0 + 1e-8)) * 100
        return rbs.item()
    
    def compute_cal(self, reliability: torch.Tensor, error: torch.Tensor,
                    target: torch.Tensor, mask: torch.Tensor = None) -> float:
        """
        Compute Calibration Error (CAL/ECE).
        
        Args:
            reliability: Per-pixel reliability (B, 1, H, W)
            error: Absolute error (B, 1, H, W)
            target: Ground truth depth (B, 1, H, W)
            mask: Valid pixels mask
        
        Returns:
            cal: Expected calibration error
        """
        if mask is not None:
            reliability = reliability[mask]
            error = error[mask]
            target = target[mask]
        
        # Flatten
        reliability = reliability.flatten()
        error = error.flatten()
        target = target.flatten()
        
        # Compute relative error
        rel_error = error / (target + 1e-8)
        correct = (rel_error < self.tolerance).float()
        
        # Bin reliability values
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=reliability.device)
        bin_indices = torch.bucketize(reliability, bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        cal = 0.0
        total = len(reliability)
        
        for b in range(self.num_bins):
            bin_mask = (bin_indices == b)
            n_b = bin_mask.sum().float()
            
            if n_b > 0:
                avg_reliability = reliability[bin_mask].mean()
                avg_accuracy = correct[bin_mask].mean()
                cal += (n_b / total) * torch.abs(avg_reliability - avg_accuracy)
        
        return cal.item()
    
    def compute_ause(self, reliability: torch.Tensor, error: torch.Tensor,
                     num_steps: int = 100) -> float:
        """
        Compute Area Under Sparsification Error (AUSE).
        
        Args:
            reliability: Per-pixel reliability (B, 1, H, W)
            error: Absolute error (B, 1, H, W)
            num_steps: Number of sparsification steps
        
        Returns:
            ause: Area under sparsification error
        """
        # Flatten
        reliability = reliability.flatten()
        error = error.flatten()
        
        # Sort by reliability (descending)
        sorted_indices = torch.argsort(reliability, descending=True)
        sorted_error = error[sorted_indices]
        
        # Compute sparsification curve
        n = len(sorted_error)
        sparsification_error = []
        oracle_error = []
        
        for step in range(num_steps + 1):
            keep_ratio = 1 - step / num_steps
            keep = int(n * keep_ratio)
            
            # Predicted ordering error
            pred_error = sorted_error[:keep].mean()
            sparsification_error.append(pred_error)
            
            # Oracle ordering (remove largest errors)
            oracle_indices = torch.argsort(error, descending=True)
            oracle_keep = oracle_indices[:keep]
            oracle_err = error[oracle_keep].mean()
            oracle_error.append(oracle_err)
        
        # Convert to numpy
        sparsification_error = np.array([e.cpu().numpy() for e in sparsification_error])
        oracle_error = np.array([e.cpu().numpy() for e in oracle_error])
        
        # Compute AUSE (area between curves)
        steps = np.linspace(0, 1, num_steps + 1)
        ause = np.trapz(sparsification_error - oracle_error, steps)
        
        return float(ause)
    
    def compute_region_masks(self, depth: torch.Tensor, rgb: torch.Tensor = None) -> dict:
        """
        Compute region masks for edge, textureless, and far-depth regions.
        
        Args:
            depth: Depth map (B, 1, H, W)
            rgb: RGB image (B, 3, H, W) for edge/texture detection
        
        Returns:
            dict with region masks
        """
        B, _, H, W = depth.shape
        masks = {}
        
        # All pixels mask
        masks['all'] = torch.ones_like(depth)
        
        # Edge mask (from depth)
        depth_grad = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
        depth_grad = F.pad(depth_grad, (0, 0, 0, 1))
        masks['edge'] = (depth_grad > 0.05).float()
        
        # Textureless mask (from RGB)
        if rgb is not None:
            # Convert to grayscale
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            
            # Compute local standard deviation
            # Use average pooling to compute local variance
            kernel_size = 7
            pad = kernel_size // 2
            gray_pad = F.pad(gray, (pad, pad, pad, pad), mode='reflect')
            
            # Compute local mean
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=gray.device) / (kernel_size ** 2)
            mean = F.conv2d(gray_pad, kernel)
            
            # Compute local variance
            var = F.conv2d(gray_pad ** 2, kernel) - mean ** 2
            std = torch.sqrt(var + 1e-8)
            
            masks['textureless'] = (std < 8.0 / 255.0).float()
        else:
            masks['textureless'] = torch.ones_like(depth)
        
        # Far-depth mask
        d_max = self.config.model.d_max if hasattr(self, 'config') else 80.0
        masks['far_depth'] = (depth > 0.75 * d_max).float()
        
        return masks
    
    def compute_radi(self, predictions: dict, targets: dict, rgb: torch.Tensor = None) -> dict:
        """
        Compute full RADI metrics.
        
        Args:
            predictions: Dict with 'reliability', 'd0', 'd1'
            targets: Dict with 'depth', 'sparse_mask'
            rgb: RGB image for region computation
        
        Returns:
            dict with RADI metrics per region
        """
        reliability = predictions['reliability']
        depth0 = predictions['d0_metric'] if 'd0_metric' in predictions else predictions['d0']
        depth1 = predictions['d1_metric'] if 'd1_metric' in predictions else predictions['d1']
        target = targets['depth']
        
        # Compute error
        error = torch.abs(depth1 - target)
        
        # Valid mask
        valid_mask = (target > 0).float()
        if 'sparse_mask' in targets:
            valid_mask = valid_mask * (1 - targets['sparse_mask'])
        
        # Compute region masks
        region_masks = self.compute_region_masks(target, rgb)
        
        results = {}
        
        for region_name, region_mask in region_masks.items():
            # Combine masks
            mask = valid_mask * region_mask
            
            # Skip if no pixels
            if mask.sum() < 1:
                continue
            
            # Compute REC
            rho, p_value = self.compute_rec(reliability, error, mask)
            
            # Compute RBS
            rbs = self.compute_rbs(depth0, depth1, target, mask)
            
            results[region_name] = {
                'rec': rho,
                'rec_p_value': p_value,
                'rbs': rbs,
            }
        
        # Compute global CAL
        results['global_cal'] = self.compute_cal(reliability, error, target, valid_mask)
        
        # Compute global AUSE
        results['global_ause'] = self.compute_ause(reliability, error, valid_mask)
        
        return results