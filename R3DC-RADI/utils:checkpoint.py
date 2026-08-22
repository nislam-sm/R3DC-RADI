"""
Checkpoint utilities for saving and loading models.
"""

import os
import torch
import shutil
from typing import Optional, Dict, Any


def save_checkpoint(state: Dict[str, Any], is_best: bool = False, 
                    filename: str = 'checkpoint.pth.tar'):
    """Save training checkpoint."""
    torch.save(state, filename)
    
    if is_best:
        best_filename = filename.replace('checkpoint', 'model_best')
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filename: str, model: torch.nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    device: torch.device = None) -> Dict[str, Any]:
    """Load training checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filename, map_location=device)
    
    # Load model
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint


def load_ema_checkpoint(filename: str, model: torch.nn.Module, 
                        device: torch.device = None) -> torch.nn.Module:
    """Load EMA checkpoint for inference."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filename, map_location=device)
    
    # Try to load EMA weights
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("Checkpoint does not contain 'ema_state_dict' or 'state_dict'")
    
    return model


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the latest checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def get_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the best checkpoint file."""
    best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_path):
        return best_path
    return None