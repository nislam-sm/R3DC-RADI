"""
Attention modules for R3DC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """Cross-modal attention between depth and RGB features."""
    
    def __init__(self, channels: int, num_heads: int = 4, max_tokens: int = 512, 
                 clamp_logits: float = 8.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.max_tokens = max_tokens
        self.clamp_logits = clamp_logits
        
        # Query projection for depth
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.q_norm = nn.GroupNorm(32, channels)
        
        # Key and value projections for RGB
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
    def _tokenize(self, x: torch.Tensor) -> tuple:
        """Convert spatial feature map to token sequence with pooling."""
        B, C, H, W = x.shape
        n_tokens = H * W
        
        if n_tokens <= self.max_tokens:
            # Flatten spatial dimensions
            tokens = x.view(B, C, -1).transpose(1, 2)  # B, N, C
            return tokens, (H, W)
        
        # Need to pool to max_tokens
        # Determine pooling factor
        pool_factor = math.ceil(math.sqrt(n_tokens / self.max_tokens))
        pooled_h = H // pool_factor
        pooled_w = W // pool_factor
        
        # Adaptive pooling
        x_pooled = F.adaptive_avg_pool2d(x, (pooled_h, pooled_w))
        tokens = x_pooled.view(B, C, -1).transpose(1, 2)  # B, N, C
        return tokens, (pooled_h, pooled_w)
    
    def _detokenize(self, tokens: torch.Tensor, spatial: tuple, original_size: tuple) -> torch.Tensor:
        """Convert tokens back to spatial feature map."""
        B, N, C = tokens.shape
        H, W = spatial
        
        if H * W == N:
            # No pooling was applied
            return tokens.transpose(1, 2).view(B, C, H, W)
        
        # Reshape and upsample back
        x = tokens.transpose(1, 2).view(B, C, H, W)
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x
    
    def forward(self, depth_feat: torch.Tensor, rgb_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = depth_feat.shape
        
        # Tokenize
        q_tokens, q_spatial = self._tokenize(depth_feat)
        kv_tokens, kv_spatial = self._tokenize(rgb_feat)
        
        # Project
        q = self.q_norm(q_tokens.transpose(1, 2)).transpose(1, 2)  # B, N, C
        k = self.k_proj(kv_tokens.transpose(1, 2)).transpose(1, 2)  # B, N, C
        v = self.v_proj(kv_tokens.transpose(1, 2)).transpose(1, 2)  # B, N, C
        
        # Multi-head attention
        head_dim = self.head_dim
        
        # Reshape for multi-head: B, N, num_heads, head_dim
        q = q.view(B, -1, self.num_heads, head_dim).transpose(1, 2)  # B, num_heads, N, head_dim
        k = k.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, -self.clamp_logits, self.clamp_logits)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # B, num_heads, N, head_dim
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, -1, C)  # B, N, C
        
        # Detokenize
        out = self._detokenize(out, q_spatial, (H, W))
        out = self.out_proj(out)
        
        # Residual connection
        return depth_feat + out


class TransformerBlock(nn.Module):
    """Transformer block with CBAM."""
    
    def __init__(self, channels: int, num_heads: int = 8, max_tokens: int = 512,
                 dropout: float = 0.1, dropout_path: float = 0.1):
        super().__init__()
        self.channels = channels
        self.max_tokens = max_tokens
        
        # Self-attention
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.drop_path1 = nn.Dropout(dropout_path) if dropout_path > 0 else nn.Identity()
        
        # MLP
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )
        self.drop_path2 = nn.Dropout(dropout_path) if dropout_path > 0 else nn.Identity()
        
        # CBAM
        self.cbam = CBAM(channels)
        
    def _tokenize(self, x: torch.Tensor) -> tuple:
        """Convert spatial feature map to token sequence."""
        B, C, H, W = x.shape
        n_tokens = H * W
        
        if n_tokens <= self.max_tokens:
            tokens = x.view(B, C, -1).transpose(1, 2)
            return tokens, (H, W)
        
        # Pool if too many tokens
        pool_factor = math.ceil(math.sqrt(n_tokens / self.max_tokens))
        pooled_h = H // pool_factor
        pooled_w = W // pool_factor
        x_pooled = F.adaptive_avg_pool2d(x, (pooled_h, pooled_w))
        tokens = x_pooled.view(B, C, -1).transpose(1, 2)
        return tokens, (pooled_h, pooled_w)
    
    def _detokenize(self, tokens: torch.Tensor, spatial: tuple, original_size: tuple) -> torch.Tensor:
        """Convert tokens back to spatial feature map."""
        B, N, C = tokens.shape
        H, W = spatial
        
        if H * W == N:
            return tokens.transpose(1, 2).view(B, C, H, W)
        
        x = tokens.transpose(1, 2).view(B, C, H, W)
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Tokenize
        tokens, spatial = self._tokenize(x)
        
        # Self-attention
        attn_out = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens))[0]
        tokens = tokens + self.drop_path1(attn_out)
        
        # MLP
        mlp_out = self.mlp(self.norm2(tokens))
        tokens = tokens + self.drop_path2(mlp_out)
        
        # Detokenize
        out = self._detokenize(tokens, spatial, (H, W))
        
        # CBAM
        out = self.cbam(out)
        
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.channel_mlp(self.channel_avg_pool(x))
        max_out = self.channel_mlp(self.channel_max_pool(x))
        channel_attn = torch.sigmoid(avg_out + max_out)
        x = x * channel_attn
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_conv(spatial_in))
        x = x * spatial_attn
        
        return x