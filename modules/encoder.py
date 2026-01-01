import torch
from modules.normalize import L2NormalizationLayer
from typing import List
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout

        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]

        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i != len(dims) - 2:
                self.mlp.append(nn.SiLU())
                if dropout != 0:
                    self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.shape[-1] == self.input_dim
        ), f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)


class PatchEncoder(nn.Module):
    """
    Patch-based encoder that downsamples N token embeddings to K semantic patches.
    
    Input:  [B, N_tokens, token_dim]  (e.g., [B, 77, 768] for CLIP)
    Output: [B, K_patches, patch_dim]  (e.g., [B, 4, 32])
    
    Architecture:
    1. Token Projection: [token_dim] → [hidden_dim] (working dimension)
    2. Cross-Attention: K learnable queries attend to all tokens
    3. Patch Processing: [hidden_dim] → [...] → [patch_dim]
    """
    
    def __init__(
        self,
        token_dim: int = 768,
        num_patches: int = 4,
        patch_dim: int = 32,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        diversity_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.token_dim = token_dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.diversity_loss_weight = diversity_loss_weight
        
        # Learnable queries for patch selection
        self.patch_queries = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        nn.init.xavier_uniform_(self.patch_queries)
        
        # Token projection to working dimension
        self.token_proj = nn.Linear(token_dim, hidden_dim, bias=False)
        
        # Multi-head cross-attention for patch selection
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Patch processing MLP (similar depth to TIGER's encoder)
        self.patch_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, patch_dim, bias=False),
        )
        
        # Layer norms
        self.norm_tokens = nn.LayerNorm(hidden_dim)
        self.norm_patches = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        token_embeddings: Tensor, 
        attention_mask: Tensor = None
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            token_embeddings: [B, N_tokens, token_dim] - Token embeddings from pretrained model
            attention_mask: [B, N_tokens] - Mask for valid tokens (1 = valid, 0 = padding)
            
        Returns:
            patches: [B, num_patches, patch_dim] - Semantic patches
            diversity_loss: scalar - Loss to encourage diverse attention patterns
        """
        B, N, D = token_embeddings.shape
        assert D == self.token_dim, f"Expected token_dim={self.token_dim}, got {D}"
        
        # Project tokens to working dimension [B, N, hidden_dim]
        tokens = self.token_proj(token_embeddings)
        tokens = self.norm_tokens(tokens)
        
        # Expand learnable queries for batch [B, num_patches, hidden_dim]
        queries = self.patch_queries.expand(B, -1, -1)
        
        # Convert attention_mask to key_padding_mask format (True = ignore)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # Invert: 0 = valid, 1 = ignore
        else:
            key_padding_mask = None
        
        # Cross-attention: queries attend to all tokens
        # Output: [B, num_patches, hidden_dim]
        patches, attention_weights = self.cross_attention(
            query=queries,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # Get per-head weights for diversity loss
        )
        
        patches = self.norm_patches(patches)
        
        # Process patches through MLP [B, num_patches, patch_dim]
        patches = self.patch_processor(patches)
        
        # Compute diversity loss to encourage different patches to attend to different tokens
        diversity_loss = self._compute_diversity_loss(attention_weights, attention_mask)
        
        return patches, diversity_loss
    
    def _compute_diversity_loss(
        self, 
        attention_weights: Tensor, 
        attention_mask: Tensor = None
    ) -> Tensor:
        """
        Encourage different patches to attend to different tokens.
        
        Args:
            attention_weights: [B, num_heads, num_patches, N_tokens]
            attention_mask: [B, N_tokens]
            
        Returns:
            diversity_loss: scalar
        """
        # Average over heads: [B, num_patches, N_tokens]
        attn = attention_weights.mean(dim=1)
        
        # Mask out padding tokens if mask is provided
        if attention_mask is not None:
            # Expand mask: [B, 1, N_tokens]
            mask = attention_mask.unsqueeze(1).float()
            attn = attn * mask
            # Renormalize
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute pairwise cosine similarity between patch attention patterns
        # Normalize attention vectors
        attn_norm = F.normalize(attn, p=2, dim=-1)  # [B, num_patches, N_tokens]
        
        # Compute similarity matrix: [B, num_patches, num_patches]
        similarity = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
        
        # Take upper triangular part (excluding diagonal) to avoid double-counting
        mask = torch.triu(torch.ones_like(similarity[0]), diagonal=1).bool()
        
        # Average similarity across batch
        diversity_loss = similarity[:, mask].mean()
        
        # Apply weight
        diversity_loss = self.diversity_loss_weight * diversity_loss
        
        return diversity_loss


class HybridEncoder(nn.Module):
    """
    Hybrid encoder that combines patch-based text encoding with global image encoding.
    
    Text: PatchEncoder on tokens → [B, K_text, patch_dim]
    Image: MLP on global → [B, 1, patch_dim]
    Output: Concatenate → [B, K_text + 1, patch_dim]
    """
    
    def __init__(
        self,
        text_token_dim: int = 768,
        image_global_dim: int = 768,
        num_text_patches: int = 4,
        patch_dim: int = 32,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        diversity_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.num_text_patches = num_text_patches
        self.patch_dim = patch_dim
        
        # Text patch encoder
        self.text_patch_encoder = PatchEncoder(
            token_dim=text_token_dim,
            num_patches=num_text_patches,
            patch_dim=patch_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            diversity_loss_weight=diversity_loss_weight,
        )
        
        # Image global encoder (simple MLP to match patch_dim)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_global_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, patch_dim, bias=False),
        )
        
    def forward(
        self,
        text_token_embeddings: Tensor = None,
        text_attention_mask: Tensor = None,
        image_global_embedding: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            text_token_embeddings: [B, N_tokens, text_token_dim]
            text_attention_mask: [B, N_tokens]
            image_global_embedding: [B, image_global_dim]
            
        Returns:
            combined_patches: [B, K_text + K_image, patch_dim]
            diversity_loss: scalar
        """
        patches_list = []
        total_diversity_loss = 0.0
        
        # Process text if provided
        if text_token_embeddings is not None:
            text_patches, text_diversity_loss = self.text_patch_encoder(
                text_token_embeddings, text_attention_mask
            )
            patches_list.append(text_patches)
            total_diversity_loss += text_diversity_loss
        
        # Process image if provided
        if image_global_embedding is not None:
            # Encode image to patch_dim: [B, image_global_dim] → [B, patch_dim]
            image_patch = self.image_encoder(image_global_embedding)
            # Add patch dimension: [B, patch_dim] → [B, 1, patch_dim]
            image_patch = image_patch.unsqueeze(1)
            patches_list.append(image_patch)
        
        # Concatenate all patches along patch dimension
        if len(patches_list) > 0:
            combined_patches = torch.cat(patches_list, dim=1)
        else:
            raise ValueError("At least one of text or image must be provided")
        
        return combined_patches, total_diversity_loss