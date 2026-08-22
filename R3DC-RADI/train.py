"""
Training script for R3DC.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from configs.default import Config
from models.r3dc import R3DC
from losses.composite_loss import R3DCLoss
from metrics.depth_metrics import compute_all_metrics
from metrics.radi import RADI
from datasets.kitti import KITTIDepthDataset
from datasets.nyu import NYUDepthDataset
from datasets.visdrone import VisDroneDataset
from datasets.drone_videos import DroneVideosDataset
from utils.logger import Logger, Timer
from utils.config import set_seed, get_device
from utils.checkpoint import save_checkpoint, load_checkpoint


def get_dataset(config):
    """Get dataset based on config."""
    dataset_name = config.dataset.dataset_name
    
    if dataset_name == 'kitti':
        return KITTIDepthDataset
    elif dataset_name == 'nyu':
        return NYUDepthDataset
    elif dataset_name == 'visdrone':
        return VisDroneDataset
    elif dataset_name == 'drone_videos':
        return DroneVideosDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_epoch(model, dataloader, criterion, optimizer, scheduler, 
                scaler, epoch, config, logger, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        rgb = batch['rgb'].to(device)
        sparse_depth = batch['sparse_depth'].to(device)
        sparse_mask = batch['sparse_mask'].to(device)
        target_depth = batch['depth'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config.training.use_amp):
            predictions = model(rgb, sparse_depth, sparse_mask)
            
            targets = {
                'depth': target_depth,
                'sparse_depth': sparse_depth,
                'sparse_mask': sparse_mask,
            }
            
            losses = criterion(predictions, targets)
            loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config.training.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA
        if hasattr(model, 'update_ema'):
            model.update_ema(config.training.ema_decay)
        
        # Logging
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % config.training.log_interval == 0:
            log_dict = {
                'train_loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
            }
            
            # Add individual loss components
            for k, v in losses.items():
                if k != 'total':
                    log_dict[f'train_{k}'] = v.item()
            
            pbar.set_postfix(log_dict)
            
            # Log to wandb
            if config.training.use_wandb:
                wandb.log({
                    **log_dict,
                    'step': epoch * len(dataloader) + batch_idx,
                })
    
    # Update scheduler
    scheduler.step()
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, epoch, config, logger, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Metrics accumulators
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")
    
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            rgb = batch['rgb'].to(device)
            sparse_depth = batch['sparse_depth'].to(device)
            sparse_mask = batch['sparse_mask'].to(device)
            target_depth = batch['depth'].to(device)
            
            # Forward pass
            with autocast(enabled=config.training.use_amp):
                predictions = model(rgb, sparse_depth, sparse_mask)
                
                targets = {
                    'depth': target_depth,
                    'sparse_depth': sparse_depth,
                    'sparse_mask': sparse_mask,
                }
                
                losses = criterion(predictions, targets)
                loss = losses['total']
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store for metric computation
            all_predictions.append(predictions['d1_metric'].cpu())
            all_targets.append(target_depth.cpu())
            
            pbar.set_postfix({'val_loss': loss.item()})
    
    # Compute metrics
    preds = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute depth metrics
    depth_metrics = compute_all_metrics(preds, targets)
    
    # Compute RADI
    radi = RADI()
    # Need to collect reliability and d0 predictions as well
    # For simplicity, we'll compute with available data
    
    metrics = {
        'val_loss': total_loss / num_batches,
        **depth_metrics,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train R3DC')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP')
    args = parser.parse_args()
    
    # Load config
    from utils.config import load_config, dict_to_namespace, update_config
    config_dict = load_config(args.config)
    config = dict_to_namespace(config_dict)
    
    # Set device
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{args.local_rank}')
        config.training.use_ddp = True
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        config.training.use_ddp = False
    
    # Set seed
    set_seed(config.training.seed)
    
    # Setup logger
    logger = Logger(config.training.log_dir)
    logger.log("=" * 50)
    logger.log(f"Training R3DC on {config.dataset.dataset_name}")
    logger.log(f"Device: {device}")
    logger.log("=" * 50)
    
    # Setup wandb
    if config.training.use_wandb and (not config.training.use_ddp or args.local_rank <= 0):
        wandb.init(
            project="r3dc",
            name=f"{config.dataset.dataset_name}_{config.model.base_width}",
            config=config_dict,
        )
    
    # Create dataset
    DatasetClass = get_dataset(config)
    
    if config.training.use_ddp:
        train_sampler = DistributedSampler(DatasetClass)
        val_sampler = DistributedSampler(DatasetClass, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_dataset = DatasetClass(
        data_root=config.dataset.data_root,
        split='train',
        transform=None,  # Add transforms here
        target_transform=None,
    )
    
    val_dataset = DatasetClass(
        data_root=config.dataset.data_root,
        split='val',
        transform=None,
        target_transform=None,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=(not config.training.use_ddp),
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        sampler=train_sampler,
        drop_last=config.dataset.drop_last,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        sampler=val_sampler,
    )
    
    # Create model
    model = R3DC(config)
    model = model.to(device)
    
    if config.training.use_ddp:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Log model info
    if not config.training.use_ddp or args.local_rank <= 0:
        logger.log_model_info(model, config_dict)
    
    # Create loss
    criterion = R3DCLoss(config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )
    
    # Create scheduler
    if config.optimizer.scheduler == 'cosine_warm_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.optimizer.t0,
            T_mult=config.optimizer.t_mult,
            eta_min=config.optimizer.eta_min,
        )
    elif config.optimizer.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.optimizer.eta_min,
        )
    elif config.optimizer.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.optimizer.step_size,
            gamma=config.optimizer.gamma,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # Create scaler for mixed precision
    scaler = GradScaler(enabled=config.training.use_amp)
    
    # Resume from checkpoint
    start_epoch = 0
    best_metric = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', float('inf'))
        logger.log(f"Resumed from checkpoint {args.resume} at epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config.training.epochs):
        if config.training.use_ddp:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, epoch, config, logger, device
        )
        
        # Validate
        if (epoch + 1) % config.training.eval_interval == 0:
            val_metrics = validate(
                model, val_loader, criterion, epoch, config, logger, device
            )
            
            logger.log_metrics(val_metrics, epoch, epoch)
            
            # Save checkpoint
            if (epoch + 1) % config.training.save_interval == 0:
                is_best = val_metrics.get('rmse', float('inf')) < best_metric
                if is_best:
                    best_metric = val_metrics.get('rmse', float('inf'))
                
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'config': config_dict,
                }
                
                if hasattr(model, 'ema_params'):
                    checkpoint['ema_state_dict'] = model.get_ema_state_dict()
                
                save_path = os.path.join(
                    config.training.checkpoint_dir,
                    f'checkpoint_epoch_{epoch}.pth.tar'
                )
                save_checkpoint(checkpoint, is_best, save_path)
    
    logger.log("Training completed!")
    
    if config.training.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()