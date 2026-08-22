"""
Data augmentation transforms for depth completion.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class RandomHorizontalFlip:
    """Random horizontal flip for RGB and depth."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, data):
        if random.random() < self.prob:
            data['rgb'] = torch.flip(data['rgb'], dims=[-1])
            if 'depth' in data:
                data['depth'] = torch.flip(data['depth'], dims=[-1])
            if 'sparse_depth' in data:
                data['sparse_depth'] = torch.flip(data['sparse_depth'], dims=[-1])
            if 'sparse_mask' in data:
                data['sparse_mask'] = torch.flip(data['sparse_mask'], dims=[-1])
        return data


class ColorJitter:
    """Color jitter augmentation for RGB."""
    
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, hue: float = 0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, data):
        # Convert to numpy for color operations
        rgb = data['rgb'].permute(1, 2, 0).numpy()
        
        # Apply color jitter
        # Brightness
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            rgb = rgb * factor
        
        # Contrast
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = rgb.mean(axis=(0, 1), keepdims=True)
            rgb = (rgb - mean) * factor + mean
        
        # Saturation
        if self.saturation > 0:
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            gray = np.stack([gray, gray, gray], axis=-1)
            rgb = rgb * factor + gray * (1 - factor)
        
        # Hue
        if self.hue > 0:
            # Simple hue rotation in HSV space
            hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-self.hue, self.hue) * 180) % 180
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # Clip
        rgb = np.clip(rgb, 0, 1)
        
        data['rgb'] = torch.from_numpy(rgb).permute(2, 0, 1)
        return data


class GammaAdjust:
    """Gamma adjustment."""
    
    def __init__(self, min_gamma: float = 0.8, max_gamma: float = 1.2):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
    
    def __call__(self, data):
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        data['rgb'] = data['rgb'] ** gamma
        return data


class SparseDropout:
    """Randomly drop sparse depth points."""
    
    def __init__(self, prob: float = 0.3):
        self.prob = prob
    
    def __call__(self, data):
        if 'sparse_depth' in data and 'sparse_mask' in data:
            mask = data['sparse_mask'].numpy()
            valid_indices = np.where(mask > 0)
            
            if len(valid_indices[0]) > 0:
                # Randomly drop points
                keep_prob = 1 - self.prob
                keep = np.random.random(len(valid_indices[0])) < keep_prob
                
                new_mask = np.zeros_like(mask)
                new_depth = np.zeros_like(data['sparse_depth'].numpy())
                
                new_mask[valid_indices[0][keep], valid_indices[1][keep]] = 1
                new_depth[valid_indices[0][keep], valid_indices[1][keep]] = \
                    data['sparse_depth'].numpy()[valid_indices[0][keep], valid_indices[1][keep]]
                
                data['sparse_mask'] = torch.from_numpy(new_mask)
                data['sparse_depth'] = torch.from_numpy(new_depth)
        
        return data


class CutMix:
    """CutMix augmentation for depth completion."""
    
    def __init__(self, prob: float = 0.3, lambda_min: float = 0.3, lambda_max: float = 0.7):
        self.prob = prob
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
    
    def __call__(self, data):
        # This is a simplified version. Full CutMix requires pairing images.
        # In practice, you'd implement this with a batch of images.
        return data


class Resize:
    """Resize RGB and depth maps."""
    
    def __init__(self, size: tuple):
        self.size = size
    
    def __call__(self, data):
        H, W = self.size
        
        if data['rgb'].shape[1:] != (H, W):
            data['rgb'] = F.interpolate(data['rgb'].unsqueeze(0), size=(H, W), 
                                       mode='bilinear', align_corners=False).squeeze(0)
            
            if 'depth' in data:
                data['depth'] = F.interpolate(data['depth'].unsqueeze(0), size=(H, W),
                                             mode='nearest').squeeze(0)
            
            if 'sparse_depth' in data:
                data['sparse_depth'] = F.interpolate(data['sparse_depth'].unsqueeze(0), size=(H, W),
                                                    mode='nearest').squeeze(0)
            
            if 'sparse_mask' in data:
                data['sparse_mask'] = F.interpolate(data['sparse_mask'].unsqueeze(0), size=(H, W),
                                                   mode='nearest').squeeze(0)
        
        return data


class Normalize:
    """Normalize RGB image."""
    
    def __init__(self, mean: tuple = (0.485, 0.456, 0.406), 
                 std: tuple = (0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, data):
        data['rgb'] = (data['rgb'] - self.mean) / self.std
        return data