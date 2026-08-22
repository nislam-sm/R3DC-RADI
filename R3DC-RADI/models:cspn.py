"""
CSPN++ refinement with reliability gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffinityNetwork(nn.Module):
    """Affinity network for CSPN++ with reliability gating."""
    
    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_neighbors = kernel_size * kernel_size - 1  # Exclude center
        
        # Affinity prediction
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 4, self.num_neighbors, 3, padding=1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: decoder features concatenated with reliability map
        Returns:
            Affinity weights for each neighbor (excluding center)
            Shape: (B, num_neighbors, H, W)
        """
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)
        affinities = self.conv2(x)
        return affinities


class ReliabilityGatedCSPN(nn.Module):
    """CSPN++ with reliability-gated propagation."""
    
    def __init__(self, in_channels: int, kernel_size: int = 3, 
                 num_steps: int = 6, center_weight: float = 0.2):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_steps = num_steps
        self.center_weight = center_weight
        self.num_neighbors = kernel_size * kernel_size - 1
        
        # Affinity network
        self.affinity_net = AffinityNetwork(in_channels, kernel_size)
        
        # Neighbor offsets
        self.register_buffer('offsets', self._get_offsets())
        
    def _get_offsets(self) -> torch.Tensor:
        """Get neighbor offsets for the kernel."""
        offsets = []
        center = self.kernel_size // 2
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == center and j == center:
                    continue
                offsets.append((i - center, j - center))
        return torch.tensor(offsets, dtype=torch.long)
    
    def _gather_neighbors(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Gather neighbor values for each pixel."""
        B, C, H, W = x.shape
        num_neighbors = offsets.shape[0]
        
        # Pad for neighbor access
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        
        # Gather neighbors
        neighbors = []
        for offset in offsets:
            dy, dx = offset
            neighbor = x_pad[:, :, pad + dy:pad + dy + H, pad + dx:pad + dx + W]
            neighbors.append(neighbor)
        
        # Stack: (B, num_neighbors, C, H, W)
        return torch.stack(neighbors, dim=1)
    
    def forward(self, coarse_depth: torch.Tensor, reliability: torch.Tensor, 
                decoder_features: torch.Tensor, sparse_depth: torch.Tensor, 
                sparse_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_depth: Coarse depth prediction (B, 1, H, W)
            reliability: Per-pixel reliability (B, 1, H, W)
            decoder_features: Decoder features (B, C, H, W)
            sparse_depth: Sparse depth anchors (B, 1, H, W)
            sparse_mask: Validity mask (B, 1, H, W)
        Returns:
            Refined depth (B, 1, H, W)
        """
        B, C, H, W = decoder_features.shape
        
        # Concatenate decoder features with reliability for affinity prediction
        affinity_input = torch.cat([decoder_features, reliability], dim=1)
        
        # Predict affinities
        affinities = self.affinity_net(affinity_input)  # (B, num_neighbors, H, W)
        
        # Normalize affinities to sum to center_weight
        affinities = F.softmax(affinities, dim=1) * self.center_weight
        # Center weight is 1 - sum(neighbor weights) = 1 - center_weight
        # But we keep center weight fixed at center_weight as in the paper
        
        # Propagation
        depth = coarse_depth.clone()
        
        for _ in range(self.num_steps):
            # Gather neighbor values
            neighbors = self._gather_neighbors(depth, self.offsets)  # (B, num_neighbors, 1, H, W)
            neighbors = neighbors.squeeze(2)  # (B, num_neighbors, H, W)
            
            # Apply affinities
            propagated = (affinities * neighbors).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # Add center contribution (fixed weight)
            propagated = propagated + (1 - self.center_weight) * depth
            
            # Apply Dirichlet boundary conditions at sparse anchors
            depth = sparse_mask * sparse_depth + (1 - sparse_mask) * propagated
        
        # Fall back to coarse where no propagation reached
        final_depth = depth + coarse_depth * (1 - sparse_mask)
        
        return final_depth