"""
Inference script for R3DC.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from models.r3dc import R3DC
from utils.config import load_config, dict_to_namespace, get_device
from utils.checkpoint import load_ema_checkpoint


def load_image(image_path: str, size: tuple = None) -> torch.Tensor:
    """Load and preprocess image."""
    image = Image.open(image_path)
    
    if size is not None:
        image = image.resize(size, Image.BILINEAR)
    
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    
    return image.unsqueeze(0)  # Add batch dimension


def load_sparse_depth(depth_path: str, size: tuple = None) -> tuple:
    """Load sparse depth map and create mask."""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        raise FileNotFoundError(f"Sparse depth file not found: {depth_path}")
    
    # Convert to meters (KITTI format: depth / 256)
    depth = depth.astype(np.float32) / 256.0
    
    if size is not None:
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    
    mask = (depth > 0).astype(np.float32)
    
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # Add batch and channel
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    
    return depth, mask


def visualize_results(rgb: torch.Tensor, sparse_depth: torch.Tensor,
                      predictions: dict, save_path: str = None):
    """Visualize inference results."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Convert to numpy for visualization
    rgb_np = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sparse_depth_np = sparse_depth.squeeze(0).squeeze(0).cpu().numpy()
    
    d0_np = predictions['d0_metric'].squeeze(0).squeeze(0).cpu().numpy()
    d1_np = predictions['d1_metric'].squeeze(0).squeeze(0).cpu().numpy()
    reliability_np = predictions['reliability'].squeeze(0).squeeze(0).cpu().numpy()
    uncertainty_np = predictions['uncertainty'].squeeze(0).squeeze(0).cpu().numpy()
    
    # Plot
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title('RGB')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sparse_depth_np, cmap='viridis')
    axes[0, 1].set_title('Sparse Depth')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(d0_np, cmap='viridis')
    axes[0, 2].set_title('Coarse Depth')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(d1_np, cmap='viridis')
    axes[0, 3].set_title('Refined Depth')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(reliability_np, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title('Reliability')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(uncertainty_np, cmap='hot')
    axes[1, 1].set_title('Uncertainty')
    axes[1, 1].axis('off')
    
    # Error maps
    if 'depth' in predictions:
        error = torch.abs(predictions['d1_metric'] - predictions['depth'])
        error_np = error.squeeze(0).squeeze(0).cpu().numpy()
        axes[1, 2].imshow(error_np, cmap='hot')
        axes[1, 2].set_title('Absolute Error')
        axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def inference(config, checkpoint_path, image_path, sparse_path=None,
              output_dir='./outputs', device=None):
    """Run inference on a single image."""
    if device is None:
        device = get_device()
    
    # Create model
    model = R3DC(config)
    model = model.to(device)
    
    # Load checkpoint
    model = load_ema_checkpoint(checkpoint_path, model, device)
    model.eval()
    
    # Load image
    size = (config.dataset.input_width, config.dataset.input_height)
    rgb = load_image(image_path, size)
    rgb = rgb.to(device)
    
    # Load sparse depth
    if sparse_path is not None:
        sparse_depth, sparse_mask = load_sparse_depth(sparse_path, size)
        sparse_depth = sparse_depth.to(device)
        sparse_mask = sparse_mask.to(device)
    else:
        # No sparse depth provided (monocular inference)
        B, C, H, W = rgb.shape
        sparse_depth = torch.zeros(B, 1, H, W, device=device)
        sparse_mask = torch.zeros(B, 1, H, W, device=device)
    
    # Inference
    with torch.no_grad():
        predictions = model(rgb, sparse_depth, sparse_mask)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save depth maps
    d1_metric = predictions['d1_metric'].squeeze(0).squeeze(0).cpu().numpy()
    reliability = predictions['reliability'].squeeze(0).squeeze(0).cpu().numpy()
    uncertainty = predictions['uncertainty'].squeeze(0).squeeze(0).cpu().numpy()
    
    # Save as images
    def save_depth_as_image(depth, path):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_norm = (depth_norm * 255).astype(np.uint8)
        depth_norm = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        cv2.imwrite(path, depth_norm)
    
    save_depth_as_image(d1_metric, os.path.join(output_dir, f'{base_name}_depth.png'))
    save_depth_as_image(reliability, os.path.join(output_dir, f'{base_name}_reliability.png'))
    save_depth_as_image(uncertainty, os.path.join(output_dir, f'{base_name}_uncertainty.png'))
    
    # Save as numpy
    np.save(os.path.join(output_dir, f'{base_name}_depth.npy'), d1_metric)
    np.save(os.path.join(output_dir, f'{base_name}_reliability.npy'), reliability)
    np.save(os.path.join(output_dir, f'{base_name}_uncertainty.npy'), uncertainty)
    
    # Visualize
    visualize_results(rgb.cpu(), sparse_depth.cpu(), predictions,
                     os.path.join(output_dir, f'{base_name}_visualization.png'))
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Inference with R3DC')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--sparse', type=str, help='Path to sparse depth map')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load config
    config_dict = load_config(args.config)
    config = dict_to_namespace(config_dict)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Run inference
    predictions = inference(
        config, args.checkpoint, args.image, args.sparse,
        args.output_dir, device
    )
    
    print("Inference completed!")


if __name__ == '__main__':
    main()