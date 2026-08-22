"""
Composite loss function for R3DC.
"""

import torch
import torch.nn as nn

from .depth_losses import (
    SILogLoss, FocalBerHuLoss, SSIMLoss, VirtualNormalLoss,
    DepthNormalConsistencyLoss, GradConsistencyLoss, AnchorLoss
)
from .uncertainty_losses import LaplaceNLLLoss, ReliabilityRegularization


class R3DCLoss(nn.Module):
    """Composite loss for R3DC training."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        loss_config = config.loss
        
        # Loss components
        self.silog_loss = SILogLoss(config.model.epsilon)
        self.focal_berhu_loss = FocalBerHuLoss(
            loss_config.focal_gamma,
            loss_config.berhu_c_factor
        )
        self.ssim_loss = SSIMLoss(loss_config.ssim_window_size)
        self.vnl_loss = VirtualNormalLoss(loss_config.num_triplets)
        self.dnc_loss = DepthNormalConsistencyLoss()
        self.grad_loss = GradConsistencyLoss()
        self.anchor_loss = AnchorLoss()
        self.laplace_nll_loss = LaplaceNLLLoss()
        self.reliability_reg = ReliabilityRegularization()
        
        # Loss weights
        self.w_silog = loss_config.weight_silog
        self.w_focal_berhu = loss_config.weight_focal_berhu
        self.w_ssim = loss_config.weight_ssim
        self.w_vnl = loss_config.weight_vnl
        self.w_dnc = loss_config.weight_dnc
        self.w_grad = loss_config.weight_grad
        self.w_anchor = loss_config.weight_anchor
        self.w_unc = loss_config.weight_unc
        self.w_aux = loss_config.weight_aux
        
    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Args:
            predictions: Dict from model forward pass
            targets: Dict with 'depth', 'sparse_depth', 'sparse_mask'
        Returns:
            Dict with loss components and total loss
        """
        # Extract predictions
        d0 = predictions['d0']
        d1 = predictions['d1']
        reliability = predictions['reliability']
        uncertainty = predictions['uncertainty']
        aux1 = predictions['aux1']
        aux2 = predictions['aux2']
        d0_metric = predictions['d0_metric']
        d1_metric = predictions['d1_metric']
        
        # Extract targets
        depth = targets['depth']  # Metric depth
        sparse_depth = targets['sparse_depth']  # Metric sparse depth
        sparse_mask = targets['sparse_mask']
        
        # Compute valid mask
        valid_mask = (depth > 0).float()
        
        # Log-normalize targets for losses in log space
        eps = self.config.model.epsilon
        d_min = self.config.model.d_min
        d_max = self.config.model.d_max
        
        depth_log = (torch.log(depth + eps) - torch.log(d_min + eps)) / \
                    (torch.log(d_max + eps) - torch.log(d_min + eps))
        sparse_depth_log = (torch.log(sparse_depth + eps) - torch.log(d_min + eps)) / \
                           (torch.log(d_max + eps) - torch.log(d_min + eps))
        
        # Compute losses
        losses = {}
        
        # SILog loss (on log-space predictions)
        losses['silog'] = self.silog_loss(d1, depth_log, valid_mask)
        losses['silog_d0'] = self.silog_loss(d0, depth_log, valid_mask)
        
        # Focal-BerHu loss (on metric-space predictions)
        losses['focal_berhu'] = self.focal_berhu_loss(d1_metric, depth, valid_mask)
        
        # SSIM loss
        losses['ssim'] = self.ssim_loss(d1_metric, depth, valid_mask)
        
        # Virtual normal loss
        losses['vnl'] = self.vnl_loss(d1_metric, depth, valid_mask)
        
        # Depth-normal consistency
        losses['dnc'] = self.dnc_loss(d1_metric, depth, valid_mask)
        
        # Gradient consistency
        losses['grad'] = self.grad_loss(d1_metric, depth, valid_mask)
        
        # Anchor loss (log space)
        losses['anchor'] = self.anchor_loss(d1, sparse_depth_log, sparse_mask)
        
        # Uncertainty NLL loss
        losses['uncertainty'] = self.laplace_nll_loss(d1_metric, depth, uncertainty, valid_mask)
        
        # Auxiliary losses
        losses['aux1'] = self.focal_berhu_loss(
            self._log_denormalize(aux1), depth, 
            F.interpolate(valid_mask, size=aux1.shape[2:], mode='nearest')
        )
        losses['aux2'] = self.focal_berhu_loss(
            self._log_denormalize(aux2), depth,
            F.interpolate(valid_mask, size=aux2.shape[2:], mode='nearest')
        )
        
        # Composite loss
        total_loss = (
            self.w_silog * (losses['silog'] + 0.5 * losses['silog_d0']) +
            self.w_focal_berhu * losses['focal_berhu'] +
            self.w_ssim * losses['ssim'] +
            self.w_vnl * losses['vnl'] +
            self.w_dnc * losses['dnc'] +
            self.w_grad * losses['grad'] +
            self.w_anchor * losses['anchor'] +
            self.w_unc * losses['uncertainty'] +
            self.w_aux * (losses['aux1'] + losses['aux2'])
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def _log_denormalize(self, depth_log: torch.Tensor) -> torch.Tensor:
        """Denormalize from log space to metric."""
        eps = self.config.model.epsilon
        d_min = self.config.model.d_min
        d_max = self.config.model.d_max
        
        return torch.exp(depth_log * (torch.log(d_max + eps) - torch.log(d_min + eps)) + 
                        torch.log(d_min + eps)) - eps