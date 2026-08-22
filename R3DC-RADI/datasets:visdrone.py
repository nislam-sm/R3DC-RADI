"""
VisDrone dataset for aerial depth completion.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

from .synthetic_depth import SyntheticDepthGenerator


class VisDroneDataset(Dataset):
    """VisDrone dataset with synthetic depth."""
    
    def __init__(self, data_root: str, split: str = 'train',
                 transform=None, target_transform=None,
                 synthetic_seed: int = 42):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load file list
        self.file_list = self._load_file_list(split)
        
        # Synthetic depth generator
        self.depth_generator = SyntheticDepthGenerator(
            height=384, width=640,
            d_min=1.0, d_max=80.0,
            seed=synthetic_seed,
            num_objects=(8, 18),
            amp_range=(-20, 20),
            noise_std=1.5
        )
        
        # Cache for generated depths
        self.cache = {}
    
    def _load_file_list(self, split: str) -> list:
        """Load file list for the given split."""
        split_file = os.path.join(self.data_root, 'splits', f'{split}.txt')
        
        with open(split_file, 'r') as f:
            files = [line.strip() for line in f.readlines()]
        
        return files
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load RGB image."""
        image = Image.open(path)
        image = np.array(image)
        return image
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> dict:
        file_id = self.file_list[idx]
        
        # Construct paths
        image_path = os.path.join(self.data_root, 'images', f'{file_id}.jpg')
        
        # Load image
        image = self._load_image(image_path)
        
        # Generate synthetic depth
        if file_id not in self.cache:
            depth = self.depth_generator.generate(image)
            sparse_mask = self.depth_generator.generate_sparse_mask(image)
            self.cache[file_id] = (depth, sparse_mask)
        else:
            depth, sparse_mask = self.cache[file_id]
        
        # Create sparse depth
        sparse_depth = depth * sparse_mask
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).float()
        sparse_mask = torch.from_numpy(sparse_mask).unsqueeze(0).float()
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            depth = self.target_transform(depth)
            sparse_depth = self.target_transform(sparse_depth)
            sparse_mask = self.target_transform(sparse_mask)
        
        return {
            'rgb': image,
            'depth': depth,
            'sparse_depth': sparse_depth,
            'sparse_mask': sparse_mask,
            'file_id': file_id,
        }