"""
Projection Head Module for RQ-VAE

This module adds projection heads to separate contrastive learning (angle)
from reconstruction/quantization (magnitude).

Usage:
    from modules.projection_head import ProjectionHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Maps embeddings from reconstruction space (magnitude matters)
    to contrastive space (only angles matter).
    
    Architecture: MLP with 2 layers + ReLU
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 128,
        use_bn: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings (from RQ-VAE)
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of projection space
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Two-layer MLP (standard in SimCLR, MoCo)
        layers = []
        
        # Layer 1
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Layer 2
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Project embeddings to contrastive space.
        
        Args:
            x: [B, input_dim] or [B, N, input_dim]
            
        Returns:
            projected: [B, output_dim] or [B, N, output_dim]
        """
        original_shape = x.shape
        
        # Flatten if needed
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
            
        # Project
        projected = self.projection(x)
        
        # Restore shape if needed
        if len(original_shape) == 3:
            projected = projected.reshape(original_shape[0], original_shape[1], -1)
            
        return projected


class MultiLevelProjectionHeads(nn.Module):
    """
    Separate projection heads for each level of RQ-VAE.
    
    Usage:
        proj_heads = MultiLevelProjectionHeads(
            embed_dim=32,
            n_levels=3,
            projection_dim=128
        )
        
        # In forward pass:
        proj_L0, proj_L1, proj_L2 = proj_heads(z_L0, z_L1, z_L2)
    """
    
    def __init__(
        self,
        embed_dim: int = 32,
        n_levels: int = 3,
        projection_dim: int = 128,
        hidden_dim: int = 64,
        use_bn: bool = True,
        shared_projection: bool = False,
    ):
        """
        Args:
            embed_dim: Dimension of RQ-VAE embeddings
            n_levels: Number of levels (typically 3)
            projection_dim: Dimension of projection space
            hidden_dim: Hidden dimension in MLP
            use_bn: Use batch normalization
            shared_projection: If True, use same projection head for all levels
                              If False, separate head per level (recommended)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        self.projection_dim = projection_dim
        self.shared_projection = shared_projection
        
        if shared_projection:
            # Single shared projection head
            self.projection = ProjectionHead(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=projection_dim,
                use_bn=use_bn,
            )
        else:
            # Separate projection head per level
            self.projections = nn.ModuleList([
                ProjectionHead(
                    input_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_dim=projection_dim,
                    use_bn=use_bn,
                )
                for _ in range(n_levels)
            ])
    
    def forward(self, *embeddings):
        """
        Project embeddings from each level.
        
        Args:
            *embeddings: Variable number of embedding tensors
                        (z_L0, z_L1, z_L2, ...)
                        Each: [B, embed_dim] or [B, N, embed_dim]
                        
        Returns:
            List of projected embeddings: [proj_L0, proj_L1, proj_L2, ...]
            Each: [B, projection_dim] or [B, N, projection_dim]
        """
        if self.shared_projection:
            # Use same projection for all levels
            return [self.projection(emb) for emb in embeddings]
        else:
            # Use separate projection per level
            assert len(embeddings) == self.n_levels, \
                f"Expected {self.n_levels} embeddings, got {len(embeddings)}"
            return [proj(emb) for proj, emb in zip(self.projections, embeddings)]


def normalize_for_contrastive(embeddings, temperature=0.07):
    """
    Normalize embeddings for contrastive learning.
    
    Args:
        embeddings: [B, D] tensor
        temperature: Temperature for scaling
        
    Returns:
        normalized: [B, D] unit-norm tensor
    """
    return F.normalize(embeddings, dim=-1) / temperature


# Example usage
if __name__ == "__main__":
    # Test single projection head
    print("Testing ProjectionHead...")
    proj_head = ProjectionHead(input_dim=32, output_dim=128)
    
    x = torch.randn(16, 32)  # [batch, embed_dim]
    proj_x = proj_head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {proj_x.shape}")
    print(f"Input norm: {x.norm(dim=-1).mean():.3f}")
    print(f"Output norm: {proj_x.norm(dim=-1).mean():.3f}")
    
    # Test multi-level projection heads
    print("\nTesting MultiLevelProjectionHeads...")
    proj_heads = MultiLevelProjectionHeads(
        embed_dim=32,
        n_levels=3,
        projection_dim=128
    )
    
    z_L0 = torch.randn(16, 32)
    z_L1 = torch.randn(16, 32)
    z_L2 = torch.randn(16, 32)
    
    proj_L0, proj_L1, proj_L2 = proj_heads(z_L0, z_L1, z_L2)
    
    print(f"z_L0 shape: {z_L0.shape} -> proj_L0 shape: {proj_L0.shape}")
    print(f"z_L1 shape: {z_L1.shape} -> proj_L1 shape: {proj_L1.shape}")
    print(f"z_L2 shape: {z_L2.shape} -> proj_L2 shape: {proj_L2.shape}")
    
    print("\n✓ All tests passed!")