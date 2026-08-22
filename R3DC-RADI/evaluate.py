"""
Evaluation script for R3DC.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np

from models.r3dc import R3DC
from metrics.depth_metrics import compute_all_metrics
from metrics.radi import RADI
from datasets.kitti import KITTIDepthDataset
from datasets.nyu import NYUDepthDataset
from utils.config import load_config, dict_to_namespace, set_seed, get_device
from utils.checkpoint import load_ema_checkpoint


def get_dataset(config, split='val'):
    """Get dataset based on config."""
    dataset_name = config.dataset.dataset_name
    
    if dataset_name == 'kitti':
        from datasets.kitti import KITTIDepthDataset
        return KITTIDepthDataset
    elif dataset_name == 'nyu':
        from datasets.nyu import NYUDepthDataset
        return NYUDepthDataset
    elif dataset_name == 'visdrone':
        from datasets.visdrone import VisDroneDataset
        return VisDroneDataset
    elif dataset_name == 'drone_videos':
        from datasets.drone_videos import DroneVideosDataset
        return DroneVideosDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate(config, checkpoint_path, device, save_results=False):
    """Evaluate the model."""
    # Set seed
    set_seed(config.training.seed)
    
    # Create dataset
    DatasetClass = get_dataset(config)
    
    dataset = DatasetClass(
        data_root=config.dataset.data_root,
        split='val',
        transform=None,
        target_transform=None,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )
    
    # Create model
    model = R3DC(config)
    model = model.to(device)
    
    # Load checkpoint (use EMA weights)
    model = load_ema_checkpoint(checkpoint_path, model, device)
    model.eval()
    
    # Initialize metrics
    radi = RADI()
    
    all_preds = []
    all_targets = []
    all_reliability = []
    all_rgbs = []
    
    # Evaluate
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            # Move to device
            rgb = batch['rgb'].to(device)
            sparse_depth = batch['sparse_depth'].to(device)
            sparse_mask = batch['sparse_mask'].to(device)
            target_depth = batch['depth'].to(device) if 'depth' in batch else None
            
            # Forward pass
            predictions = model(rgb, sparse_depth, sparse_mask)
            
            # Store predictions
            all_preds.append(predictions['d1_metric'].cpu())
            all_reliability.append(predictions['reliability'].cpu())
            all_rgbs.append(rgb.cpu())
            
            if target_depth is not None:
                all_targets.append(target_depth.cpu())
    
    # Concatenate predictions
    preds = torch.cat(all_preds, dim=0)
    reliability = torch.cat(all_reliability, dim=0)
    rgbs = torch.cat(all_rgbs, dim=0)
    targets = torch.cat(all_targets, dim=0) if all_targets else None
    
    # Compute metrics
    metrics = {}
    
    if targets is not None:
        # Depth metrics
        depth_metrics = compute_all_metrics(preds, targets)
        metrics.update(depth_metrics)
        
        # RADI metrics
        predictions_dict = {
            'reliability': reliability,
            'd0': preds,  # Using preds as d0 for simplicity
            'd1': preds,
            'd0_metric': preds,
            'd1_metric': preds,
        }
        targets_dict = {
            'depth': targets,
        }
        
        radi_results = radi.compute_radi(predictions_dict, targets_dict, rgbs)
        
        # Flatten radi results
        for region, region_results in radi_results.items():
            for k, v in region_results.items():
                metrics[f'radi_{region}_{k}'] = v
    
    # Print metrics
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.6f}")
    
    # Save results
    if save_results:
        output_dir = os.path.dirname(checkpoint_path)
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate R3DC')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_results', action='store_true', help='Save results to JSON')
    args = parser.parse_args()
    
    # Load config
    config_dict = load_config(args.config)
    config = dict_to_namespace(config_dict)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Evaluate
    evaluate(config, args.checkpoint, device, args.save_results)


if __name__ == '__main__':
    main()