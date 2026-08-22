"""
Default configuration for R3DC model.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class ModelConfig:
    """Model configuration."""
    # Architecture
    base_width: int = 64
    num_propagation_steps: int = 6
    kernel_size: int = 3
    max_tokens: int = 512
    num_heads: int = 8
    dropout_path_rate: float = 0.1
    group_norm_groups: int = 32
    
    # Depth normalization
    d_min: float = 0.0
    d_max: float = 80.0
    epsilon: float = 1e-3
    
    # Output heads
    predict_reliability: bool = True
    predict_uncertainty: bool = True
    predict_depth: bool = True


@dataclass
class LossConfig:
    """Loss configuration."""
    # Loss weights
    weight_silog: float = 1.0
    weight_focal_berhu: float = 0.6
    weight_ssim: float = 0.2
    weight_anchor: float = 0.15
    weight_vnl: float = 0.1
    weight_dnc: float = 0.05
    weight_grad: float = 0.05
    weight_unc: float = 0.05
    weight_aux: float = 0.1
    
    # Focal-BerHu parameters
    focal_gamma: float = 2.0
    berhu_c_factor: float = 0.2
    
    # SSIM parameters
    ssim_window_size: int = 7
    
    # VNL parameters
    num_triplets: int = 1024
    
    # DNC parameters
    dnc_delta_h: Tuple[int, int] = (1, 0)
    dnc_delta_v: Tuple[int, int] = (0, 1)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    # AdamW
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler: str = "cosine_warm_restart"  # cosine_warm_restart, cosine, step
    warmup_epochs: int = 0
    t0: int = 10
    t_mult: int = 2
    eta_min: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_name: str = "kitti"
    data_root: str = "./data"
    split: str = "train"
    input_height: int = 352
    input_width: int = 1216
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    
    # Augmentation
    use_augmentation: bool = True
    horizontal_flip: bool = True
    color_jitter: bool = True
    gamma_adjust: bool = True
    sparse_dropout: float = 0.0  # Probability of dropping sparse points
    cutmix_prob: float = 0.0
    cutmix_lambda_min: float = 0.3
    cutmix_lambda_max: float = 0.7


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 20
    gradient_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 1
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    device: str = "cuda"
    seed: int = 42
    ema_decay: float = 0.9999
    use_amp: bool = True
    use_ddp: bool = False


@dataclass
class Config:
    """Full configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        if self.dataset.dataset_name == "kitti":
            self.model.d_min = 0.0
            self.model.d_max = 80.0
            self.dataset.input_height = 352
            self.dataset.input_width = 1216
        elif self.dataset.dataset_name == "nyu":
            self.model.d_min = 0.001
            self.model.d_max = 10.0
            self.dataset.input_height = 518
            self.dataset.input_width = 518
        elif self.dataset.dataset_name == "visdrone":
            self.model.d_min = 1.0
            self.model.d_max = 80.0
            self.dataset.input_height = 384
            self.dataset.input_width = 640
        elif self.dataset.dataset_name == "drone_videos":
            self.model.d_min = 0.0
            self.model.d_max = 50.0
            self.dataset.input_height = 384
            self.dataset.input_width = 640