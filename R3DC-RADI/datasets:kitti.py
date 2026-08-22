"""
KITTI Depth Completion dataset.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2


class KITTIDepthDataset(Dataset):
    """KITTI depth completion dataset."""
    
    def __init__(self, data_root: str, split: str = 'train', 
                 transform=None, target_transform=None,
                 subsample_ratio: float = 1.0, uniform_subsample: bool = False):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.subsample_ratio = subsample_ratio
        self.uniform_subsample = uniform_subsample
        
        # Load file list
        if split == 'train':
            self.file_list = self._load_file_list('train')
        elif split == 'val':
            self.file_list = self._load_file_list('val')
        elif split == 'select':
            self.file_list = self._load_file_list('select')
        elif split == 'test':
            self.file_list = self._load_file_list('test')
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _load_file_list(self, split: str) -> list:
        """Load file list for the given split."""
        split_file = os.path.join(self.data_root, 'splits', f'{split}.txt')
        
        with open(split_file, 'r') as f:
            files = [line.strip() for line in f.readlines()]
        
        return files
    
    def _load_depth(self, path: str) -> np.ndarray:
        """Load depth map from PNG file."""
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth file not found: {path}")
        depth = depth.astype(np.float32) / 256.0  # Convert to meters
        return depth
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load RGB image."""
        image = Image.open(path)
        image = np.array(image)
        return image
    
    def _load_sparse_depth(self, path: str) -> np.ndarray:
        """Load sparse depth map."""
        sparse_depth = self._load_depth(path)
        return sparse_depth
    
    def _load_ground_truth(self, path: str) -> np.ndarray:
        """Load ground truth depth map."""
        return self._load_depth(path)
    
    def _subsample_sparse_depth(self, sparse_depth: np.ndarray, 
                                ratio: float) -> np.ndarray:
        """Subsample sparse depth points."""
        if ratio >= 1.0:
            return sparse_depth
        
        valid_mask = sparse_depth > 0
        valid_indices = np.where(valid_mask)
        
        n_valid = len(valid_indices[0])
        n_keep = int(n_valid * ratio)
        
        if n_keep == 0:
            return np.zeros_like(sparse_depth)
        
        # Randomly select points to keep
        indices = np.random.choice(n_valid, n_keep, replace=False)
        
        subsampled = np.zeros_like(sparse_depth)
        subsampled[valid_indices[0][indices], valid_indices[1][indices]] = \
            sparse_depth[valid_indices[0][indices], valid_indices[1][indices]]
        
        return subsampled
    
    def _uniform_subsample_sparse_depth(self, sparse_depth: np.ndarray,
                                        ratio: float) -> np.ndarray:
        """Uniformly subsample sparse depth points."""
        if ratio >= 1.0:
            return sparse_depth
        
        valid_mask = sparse_depth > 0
        
        # Use grid-based subsampling
        H, W = sparse_depth.shape
        
        # Determine grid size
        grid_size = int(np.sqrt(1 / ratio))
        grid_size = max(1, grid_size)
        
        subsampled = np.zeros_like(sparse_depth)
        
        for i in range(0, H, grid_size):
            for j in range(0, W, grid_size):
                patch = sparse_depth[i:min(i+grid_size, H), j:min(j+grid_size, W)]
                valid_patch = patch > 0
                if valid_patch.any():
                    # Take first valid point in patch
                    idx = np.where(valid_patch)
                    subsampled[i+idx[0][0], j+idx[1][0]] = patch[idx[0][0], idx[1][0]]
        
        return subsampled
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> dict:
        file_id = self.file_list[idx]
        
        # Construct paths
        image_path = os.path.join(self.data_root, 'image', f'{file_id}.png')
        sparse_path = os.path.join(self.data_root, 'depth', f'{file_id}.png')
        
        if self.split != 'test':
            gt_path = os.path.join(self.data_root, 'gt', f'{file_id}.png')
        
        # Load data
        image = self._load_image(image_path)
        sparse_depth = self._load_sparse_depth(sparse_path)
        
        if self.split != 'test':
            gt_depth = self._load_ground_truth(gt_path)
        
        # Subsample sparse depth
        if self.subsample_ratio < 1.0:
            if self.uniform_subsample:
                sparse_depth = self._uniform_subsample_sparse_depth(
                    sparse_depth, self.subsample_ratio
                )
            else:
                sparse_depth = self._subsample_sparse_depth(
                    sparse_depth, self.subsample_ratio
                )
        
        # Create sparse mask
        sparse_mask = (sparse_depth > 0).astype(np.float32)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).float()
        sparse_mask = torch.from_numpy(sparse_mask).unsqueeze(0).float()
        
        if self.split != 'test':
            gt_depth = torch.from_numpy(gt_depth).unsqueeze(0).float()
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None and self.split != 'test':
            gt_depth = self.target_transform(gt_depth)
            sparse_depth = self.target_transform(sparse_depth)
            sparse_mask = self.target_transform(sparse_mask)
        
        result = {
            'rgb': image,
            'sparse_depth': sparse_depth,
            'sparse_mask': sparse_mask,
            'file_id': file_id,
        }
        
        if self.split != 'test':
            result['depth'] = gt_depth
        
        return result