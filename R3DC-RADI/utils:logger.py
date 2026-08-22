"""
Logging utilities for training and evaluation.
"""

import os
import json
import logging
import time
from datetime import datetime
import torch


class Logger:
    """Logger for training and evaluation."""
    
    def __init__(self, log_dir: str, name: str = "r3dc"):
        self.log_dir = log_dir
        self.name = name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics = {}
        self.metrics_file = os.path.join(log_dir, 'metrics.json')
    
    def log(self, message: str, level: str = "info"):
        """Log a message."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
    
    def log_metrics(self, metrics: dict, step: int, epoch: int):
        """Log metrics."""
        # Add to storage
        if epoch not in self.metrics:
            self.metrics[epoch] = {}
        
        for key, value in metrics.items():
            if key not in self.metrics[epoch]:
                self.metrics[epoch][key] = []
            self.metrics[epoch][key].append(value)
        
        # Log to console
        message = f"Epoch {epoch}, Step {step}: "
        message += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        self.logger.info(message)
        
        # Save to file
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_model_info(self, model: torch.nn.Module, config: dict):
        """Log model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Config: {json.dumps(config, indent=2)}")
        self.logger.info("=" * 50)


class Timer:
    """Timer for measuring execution time."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.total_time = 0
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.total_time += elapsed
        return elapsed
    
    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.total_time = 0
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time