"""
R3DC: Reliability-Aware Reveal-to-Revise Depth Completion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RGBEncoder, DepthEncoder
from .decoder import FPNDecoder
from .attention import TransformerBlock
from .cspn import ReliabilityGatedCSPN
from .heads import DepthHead, ReliabilityHead, UncertaintyHead, AuxiliaryHead
from .ich import FoundationAdapter


class R3DC(nn.Module):
    """Main R3DC model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = config.model.base_width
        self.use_ich = config.dataset.dataset_name == "nyu"
        
        if self.use_ich:
            # Use foundation model adapter for indoor scenes
            # Note: You need to load the actual foundation model
            # This is a placeholder structure
            from .ich import FoundationAdapter
            # self.foundation_adapter = FoundationAdapter(foundation_model, ...)
            # For now, we'll use a dummy adapter
            self.foundation_adapter = None
            self.feature_dim = 384  # ViT-S output dimension
        else:
            # Outdoor encoders
            self.rgb_encoder = RGBEncoder(self.base, config.model.dropout_path_rate)
            self.depth_encoder = DepthEncoder(self.base, config.model.dropout_path_rate)
            
            # Cross-modal attention channels at each scale
            self.cma_channels = [
                self.base // 2,  # scale 1
                self.base,       # scale 1/2
                self.base * 2,   # scale 1/4
            ]
            
            # Transformer bottleneck
            self.bottleneck = TransformerBlock(
                self.base * 4, 
                config.model.num_heads,
                config.model.max_tokens,
                dropout=0.1,
                dropout_path=config.model.dropout_path_rate
            )
            
            # FPN Decoder
            self.decoder = FPNDecoder(
                self.base, 
                config.model.dropout_path_rate,
                config.model.num_heads,
                config.model.max_tokens
            )
            
            # Output heads
            self.depth_head = DepthHead(self.base // 2)
            self.reliability_head = ReliabilityHead(self.base // 2)
            self.uncertainty_head = UncertaintyHead(self.base // 2)
            
            # Auxiliary heads for deep supervision
            self.aux1 = AuxiliaryHead(self.base * 2)  # scale 1/2
            self.aux2 = AuxiliaryHead(self.base * 4)  # scale 1/4
        
        # CSPN++ refinement
        self.refiner = ReliabilityGatedCSPN(
            self.base // 2,  # in_channels (decoder feature channels)
            config.model.kernel_size,
            config.model.num_propagation_steps,
            center_weight=0.2
        )
        
        # EMA parameters (for inference)
        self.ema_params = None
        
    def _log_normalize(self, depth: torch.Tensor) -> torch.Tensor:
        """Log-normalize depth to [0, 1]."""
        eps = self.config.model.epsilon
        d_min = self.config.model.d_min
        d_max = self.config.model.d_max
        
        depth = torch.clamp(depth, min=d_min, max=d_max)
        return (torch.log(depth + eps) - torch.log(d_min + eps)) / \
               (torch.log(d_max + eps) - torch.log(d_min + eps))
    
    def _log_denormalize(self, depth: torch.Tensor) -> torch.Tensor:
        """Denormalize from log space to metric."""
        eps = self.config.model.epsilon
        d_min = self.config.model.d_min
        d_max = self.config.model.d_max
        
        return torch.exp(depth * (torch.log(d_max + eps) - torch.log(d_min + eps)) + 
                        torch.log(d_min + eps)) - eps
    
    def forward(self, rgb: torch.Tensor, sparse_depth: torch.Tensor, 
                sparse_mask: torch.Tensor) -> dict:
        """
        Forward pass of R3DC.
        
        Args:
            rgb: RGB image (B, 3, H, W)
            sparse_depth: Sparse depth map (B, 1, H, W)
            sparse_mask: Validity mask (B, 1, H, W)
            
        Returns:
            dict with:
                d0: Coarse depth in log space
                d1: Refined depth in log space
                reliability: Per-pixel reliability [0, 1]
                uncertainty: Aleatoric uncertainty
                aux1: Auxiliary depth at scale 1/2
                aux2: Auxiliary depth at scale 1/4
                d0_metric: Coarse depth in metric space
                d1_metric: Refined depth in metric space
        """
        # Log-normalize input depth
        sparse_depth_log = self._log_normalize(sparse_depth)
        
        if self.use_ich:
            # Indoor: use foundation model adapter
            # For now, we'll use a simplified path
            # This should be replaced with actual foundation model usage
            depth_features = self.depth_encoder(torch.cat([sparse_depth_log, sparse_mask], dim=1))
            rgb_features = self.rgb_encoder(rgb)
            features = {'rgb': rgb_features, 'depth': depth_features}
        else:
            # Outdoor: dual-stream encoding
            rgb_features = self.rgb_encoder(rgb)
            depth_input = torch.cat([sparse_depth_log, sparse_mask], dim=1)
            depth_features = self.depth_encoder(depth_input)
            
            features = {'rgb': rgb_features, 'depth': depth_features}
        
        # Decode to get coarse depth and features
        decoder_features = self.decoder(features)
        
        # Get output heads
        d0 = self.depth_head(decoder_features)
        reliability = self.reliability_head(decoder_features)
        uncertainty = self.uncertainty_head(decoder_features)
        
        # Auxiliary outputs for deep supervision
        # Note: Need to access intermediate decoder features
        aux1 = self.aux1(features['depth'][2])  # scale 1/4
        aux2 = self.aux2(features['depth'][3])  # scale 1/8
        
        # Refine with CSPN++
        d1 = self.refiner(d0, reliability, decoder_features, 
                         sparse_depth_log, sparse_mask)
        
        # Denormalize to metric space
        d0_metric = self._log_denormalize(d0)
        d1_metric = self._log_denormalize(d1)
        
        return {
            'd0': d0,
            'd1': d1,
            'reliability': reliability,
            'uncertainty': uncertainty,
            'aux1': aux1,
            'aux2': aux2,
            'd0_metric': d0_metric,
            'd1_metric': d1_metric,
        }
    
    def get_ema_state_dict(self):
        """Get EMA parameters for inference."""
        if self.ema_params is None:
            return self.state_dict()
        return self.ema_params
    
    def update_ema(self, decay: float = 0.9999):
        """Update EMA parameters."""
        if self.ema_params is None:
            self.ema_params = {k: v.clone() for k, v in self.state_dict().items()}
        else:
            for k, v in self.state_dict().items():
                self.ema_params[k] = decay * self.ema_params[k] + (1 - decay) * v