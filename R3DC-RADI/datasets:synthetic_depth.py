"""
Synthetic depth generation for aerial datasets.
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter


class SyntheticDepthGenerator:
    """Generate synthetic depth for aerial images."""
    
    def __init__(self, height: int, width: int, d_min: float, d_max: float,
                 seed: int = None, num_objects: tuple = (8, 18),
                 amp_range: tuple = (-20, 20), noise_std: float = 1.5):
        self.height = height
        self.width = width
        self.d_min = d_min
        self.d_max = d_max
        self.num_objects = num_objects
        self.amp_range = amp_range
        self.noise_std = noise_std
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate(self, rgb: np.ndarray) -> np.ndarray:
        """Generate synthetic depth for an RGB image."""
        H, W = self.height, self.width
        
        # Base depth (linear gradient)
        y_coords = np.linspace(0, 1, H)[:, None]
        x_coords = np.linspace(0, 1, W)[None, :]
        
        # Base ground plane
        # Values chosen to approximate typical aerial scenes
        D_base = 15 + 25 * (1 - y_coords)
        
        # Lateral terrain variations
        D_lat = 12 * np.sin(4 * np.pi * x_coords + np.pi * y_coords) + \
                8 * np.cos(6 * np.pi * x_coords + 2 * np.pi * y_coords)
        
        # Object primitives (buildings, vehicles, etc.)
        D_obj = np.zeros((H, W))
        n_objects = np.random.randint(self.num_objects[0], self.num_objects[1] + 1)
        
        for _ in range(n_objects):
            amp = np.random.uniform(self.amp_range[0], self.amp_range[1])
            cy = np.random.uniform(0, H)
            cx = np.random.uniform(0, W)
            sigma_y = 0.031 * H
            sigma_x = 0.027 * W
            
            # Gaussian primitive
            y_grid = np.arange(H)[:, None] - cy
            x_grid = np.arange(W)[None, :] - cx
            primitive = amp * np.exp(-(y_grid ** 2) / (2 * sigma_y ** 2) - 
                                     (x_grid ** 2) / (2 * sigma_x ** 2))
            D_obj += primitive
        
        # Combine
        D = D_base + D_lat + D_obj
        
        # Add noise
        noise = np.random.randn(H, W) * self.noise_std
        D = D + noise
        
        # Clip to valid range
        D = np.clip(D, self.d_min, self.d_max)
        
        return D.astype(np.float32)
    
    def generate_sparse_mask(self, rgb: np.ndarray, density: float = 0.025,
                             edge_sensitivity: float = 0.5) -> np.ndarray:
        """Generate sparse anchor mask with edge-aware sampling."""
        H, W = self.height, self.width
        
        # Center-weighted density
        y_coords = np.linspace(0, 1, H)[:, None]
        x_coords = np.linspace(0, 1, W)[None, :]
        
        density_map = density * np.exp(-4 * (x_coords - 0.5) ** 2 - 4 * (y_coords - 0.5) ** 2)
        
        # Edge-aware factor (reduce density at edges)
        # Convert RGB to grayscale for edge detection
        if rgb is not None:
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            
            # Compute edge map
            from scipy.ndimage import sobel
            edge_map = np.sqrt(sobel(gray, axis=0) ** 2 + sobel(gray, axis=1) ** 2)
            edge_map = gaussian_filter(edge_map, sigma=edge_sensitivity)
            
            # Normalize edge map
            edge_map = edge_map / (edge_map.max() + 1e-8)
            edge_map = 1 - edge_map  # Invert: lower density at edges
        else:
            edge_map = np.ones((H, W))
        
        # Apply edge-aware factor
        density_map = density_map * edge_map
        
        # Clip
        density_map = np.clip(density_map, 5e-4, 2.5e-2)
        
        # Sample points based on density
        mask = np.random.rand(H, W) < density_map
        
        return mask.astype(np.float32)