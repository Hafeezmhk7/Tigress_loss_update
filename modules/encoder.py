import torch
from modules.normalize import L2NormalizationLayer
from typing import List, Optional, Tuple
from torch import nn
from torch import Tensor
import torch.nn.functional as F


# =============================================================================
# LEGACY: MLP (needed by rqvae.py and other modules)
# =============================================================================

class MLP(nn.Module):
    """
    Legacy MLP encoder (for RQ-VAE and backward compatibility)
    """
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


# =============================================================================
# NEW: Hierarchical Patch Encoder for PQ-VAE
# =============================================================================

class HierarchicalPatchEncoder(nn.Module):
    """
    Hierarchical Patch Encoder for Semantic ID Generation
    
    Key Innovation: Sequential conditioning for hierarchical semantics
    
    Architecture (Based on VQ-VAE-2 + TIGER):
    1. Self-attention: Build contextual token representations
    2. Hierarchical extraction: Each semantic token conditioned on previous
       - Token 0: Category (most abstract)
       - Token 1: Subcategory (conditioned on token 0)
       - Token 2: Attributes (conditioned on tokens 0,1)
       - Token 3: Details (conditioned on tokens 0,1,2)
    3. Independent quantization: Each token gets own codebook
    
    This creates hierarchical codes perfect for:
    - Autoregressive generation (predict next item's codes sequentially)
    - Semantic understanding (each level captures different abstraction)
    - Product quantization (independent codebooks)
    
    Input:  [B, N_tokens, token_dim]      e.g., [B, 77, 1024]
    Output: [B, num_codebooks, token_embed_dim]  e.g., [B, 4, 192]
    """
    
    def __init__(
        self,
        token_dim: int = 1024,
        num_codebooks: int = 4,
        token_embed_dim: int = 192,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_diversity_loss: bool = True,
        diversity_weight: float = 0.01,
    ) -> None:
        super().__init__()
        
        self.token_dim = token_dim
        self.num_codebooks = num_codebooks
        self.token_embed_dim = token_embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_diversity_loss = use_diversity_loss
        self.diversity_weight = diversity_weight
        
        # Total output dimension
        self.output_dim = num_codebooks * token_embed_dim
        
        # Input projection: token_dim → hidden_dim
        self.input_proj = nn.Linear(token_dim, hidden_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Self-attention layers (process all tokens)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Hierarchical learnable queries (one per codebook level)
        self.semantic_queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, hidden_dim))
            for _ in range(num_codebooks)
        ])
        for query in self.semantic_queries:
            nn.init.xavier_uniform_(query)
        
        # Conditioning modules (project previous tokens for next query)
        self.conditioning_modules = nn.ModuleList([
            nn.Identity()  # Level 0: no conditioning
        ] + [
            nn.Linear(hidden_dim * i, hidden_dim, bias=False)
            for i in range(1, num_codebooks)
        ])
        
        # Cross-attention layers (one per level)
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_codebooks)
        ])
        
        # Layer norms for each level
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_codebooks)
        ])
        
        # Feed-forward for each semantic token
        self.token_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
            for _ in range(num_codebooks)
        ])
        self.token_ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_codebooks)
        ])
        
        # Final projections (to token_embed_dim)
        self.output_projs = nn.ModuleList([
            nn.Linear(hidden_dim, token_embed_dim, bias=False)
            for _ in range(num_codebooks)
        ])
        
    def forward(
        self, 
        token_embeddings: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Hierarchical encoding with sequential conditioning.
        
        Args:
            token_embeddings: [B, N, token_dim] - Patch embeddings
            attention_mask: [B, N] - Attention mask
            
        Returns:
            semantic_tokens: [B, num_codebooks, token_embed_dim]
            diversity_loss: scalar or None
        """
        B, N, D = token_embeddings.shape
        assert D == self.token_dim
        
        # Step 1: Project and self-attention
        x = self.input_proj(token_embeddings)
        x = self.input_norm(x)
        
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        
        for layer in self.self_attn_layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        # x: [B, N, hidden_dim] - contextual tokens
        
        # Step 2: Hierarchical token extraction
        semantic_tokens = []
        attention_weights_list = []
        
        for level in range(self.num_codebooks):
            # Get base query for this level
            query = self.semantic_queries[level].expand(B, 1, -1)
            
            # Condition on previous tokens (hierarchical dependency)
            if level > 0:
                # Concatenate all previous semantic tokens
                prev_tokens = torch.cat(semantic_tokens, dim=1)  # [B, level, hidden_dim]
                prev_tokens_flat = prev_tokens.reshape(B, -1)  # [B, level * hidden_dim]
                
                # Condition query on previous tokens
                conditioning = self.conditioning_modules[level](prev_tokens_flat)  # [B, hidden_dim]
                query = query + conditioning.unsqueeze(1)  # Additive conditioning
            
            # Cross-attention: conditioned query attends to all input tokens
            semantic_token, attn_weights = self.cross_attention_layers[level](
                query=query,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            # semantic_token: [B, 1, hidden_dim]
            # attn_weights: [B, num_heads, 1, N]
            
            semantic_token = self.cross_attn_norms[level](query + semantic_token)
            
            # Feed-forward
            semantic_token = self.token_ffn_norms[level](
                semantic_token + self.token_ffns[level](semantic_token)
            )
            
            semantic_tokens.append(semantic_token)
            attention_weights_list.append(attn_weights)
        
        # Step 3: Stack and project
        semantic_tokens = torch.cat(semantic_tokens, dim=1)  # [B, K, hidden_dim]
        
        # Project each token independently
        output_tokens = []
        for level in range(self.num_codebooks):
            token = semantic_tokens[:, level:level+1, :]  # [B, 1, hidden_dim]
            projected = self.output_projs[level](token)   # [B, 1, token_embed_dim]
            output_tokens.append(projected)
        
        output_tokens = torch.cat(output_tokens, dim=1)  # [B, K, token_embed_dim]
        
        # Step 4: Diversity loss
        diversity_loss = None
        if self.use_diversity_loss:
            diversity_loss = self._compute_diversity_loss(
                attention_weights_list, attention_mask
            )
        
        return output_tokens, diversity_loss
    
    def _compute_diversity_loss(
        self,
        attention_weights_list: List[Tensor],
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Encourage different semantic tokens to attend to different regions.
        """
        # Stack and average over heads
        attns = [a.mean(dim=1) for a in attention_weights_list]
        attn = torch.cat(attns, dim=1)  # [B, K, N]
        
        # Mask padding
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).float()
            attn = attn * mask
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute pairwise similarity
        attn_norm = F.normalize(attn, p=2, dim=-1)
        similarity = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
        
        # Upper triangular (exclude diagonal)
        mask = torch.triu(torch.ones_like(similarity[0]), diagonal=1).bool()
        
        diversity_loss = similarity[:, mask].mean()
        diversity_loss = self.diversity_weight * diversity_loss
        
        return diversity_loss


# =============================================================================
# NEW: Cross-Attention Decoder
# =============================================================================

class CrossAttentionDecoder(nn.Module):
    """
    Cross-Attention Decoder (Preserves Token Structure!)
    
    Based on VQ-VAE principle: Decoder operates on quantized latent structure.
    
    Architecture:
    - Global learnable query
    - Attends to all K quantized semantic tokens
    - Projects to target dimension (global embedding)
    
    Input:  [B, K, token_embed_dim]  e.g., [B, 4, 192]
    Output: [B, output_dim]          e.g., [B, 768]
    """
    
    def __init__(
        self,
        input_dim: int = 192,        # token_embed_dim
        num_tokens: int = 4,          # num_codebooks
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.total_dim = num_tokens * input_dim
        
        # Learnable global query
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.global_query)
        
        # Project tokens to hidden dimension
        self.token_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.token_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        
    def forward(self, quantized_tokens: Tensor) -> Tensor:
        """
        Decode from quantized semantic tokens.
        
        Args:
            quantized_tokens: [B, K, token_embed_dim]
            
        Returns:
            output: [B, output_dim] - Reconstructed global embedding
        """
        B, K, D = quantized_tokens.shape
        assert K == self.num_tokens
        assert D == self.input_dim
        
        # Project tokens to hidden dimension
        tokens = self.token_proj(quantized_tokens)
        tokens = self.token_norm(tokens)
        # tokens: [B, K, hidden_dim]
        
        # Expand global query
        query = self.global_query.expand(B, 1, -1)
        
        # Multiple cross-attention layers
        for i in range(len(self.cross_attn_layers)):
            # Cross-attention
            output, _ = self.cross_attn_layers[i](
                query=query,
                key=tokens,
                value=tokens,
                need_weights=False,
            )
            query = self.cross_attn_norms[i](query + output)
            
            # Feed-forward
            query = self.ffn_norms[i](query + self.ffns[i](query))
        
        # query: [B, 1, hidden_dim]
        
        # Project to output dimension
        output = self.output_proj(query.squeeze(1))
        # output: [B, output_dim]
        
        return output