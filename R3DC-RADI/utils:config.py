"""
Configuration utilities.
"""

import os
import json
import yaml
from typing import Any, Dict, Union
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file."""
    if save_path.endswith('.json'):
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
        with open(save_path, 'w') as f:
            yaml.dump(config, f)
    else:
        raise ValueError(f"Unsupported config format: {save_path}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with nested updates."""
    import copy
    config = copy.deepcopy(config)
    
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    
    return config


def dict_to_namespace(config: Dict[str, Any]) -> Any:
    """Convert dictionary to namespace for easier access."""
    import argparse
    
    namespace = argparse.Namespace()
    
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    
    return namespace


def get_device() -> torch.device:
    """Get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False