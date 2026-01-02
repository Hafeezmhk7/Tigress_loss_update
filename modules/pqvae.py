import gin
import torch
from dataclasses import dataclass
from modules.encoder import HierarchicalPatchEncoder, CrossAttentionDecoder
from modules.quantize import ProductQuantize
from modules.loss import ReconstructionLoss, QuantizeLoss
from torch import nn, Tensor
from typing import Optional, List


@dataclass
class PqVaeOutput:
    """Output from PQ-VAE forward pass"""
    loss: Tensor
    reconstruction_loss: Tensor
    pqvae_loss: Tensor
    diversity_loss: Optional[Tensor]
    encoder_output: Tensor          # [B, K, D]
    quantized_output: Tensor        # [B, K, D]
    codes: Tensor                   # [B, K] - Semantic IDs!
    decoder_output: Tensor          # [B, output_dim]
    # embs_norm: Tensor
    # p_unique_ids: Tensor


@gin.configurable
class PqVae(nn.Module):
    """
    Hierarchical Product Quantization VAE for Semantic ID Generation
    
    Key Innovation: Combines hierarchical semantics with independent quantization
    
    Architecture (Based on VQ-VAE-2 + TIGER + PQ):
    1. Hierarchical Encoder:
       - Self-attention on patches
       - Sequential extraction of K semantic tokens
       - Each token conditioned on previous (hierarchy!)
    
    2. Product Quantization:
       - K independent codebooks
       - Each token quantized separately
       - Creates K-dimensional semantic ID
    
    3. Cross-Attention Decoder:
       - Preserves token structure (no flattening!)
       - Attends to all quantized tokens
       - Reconstructs global embedding
    
    Perfect for:
    - Autoregressive generation (predict codes sequentially)
    - Semantic understanding (hierarchical codes)
    - Efficient retrieval (product quantization)
    """
    
    def __init__(
        self,
        # Reconstruction target
        output_dim: int = 768,           # Reconstruct global embedding
        
        # Encoder
        use_patch_encoder: bool = False,
        patch_token_dim: int = 1024,
        patch_token_embed_dim: int = 192,
        patch_hidden_dim: int = 512,
        patch_num_heads: int = 8,
        patch_num_layers: int = 2,
        patch_dropout: float = 0.1,
        
        # Quantization
        num_codebooks: int = 4,
        codebook_size: int = 256,
        codebook_kmeans_init: bool = True,
        commitment_weight: float = 0.25,
        
        # Diversity
        use_diversity_loss: bool = True,
        diversity_weight: float = 0.01,
        
        # Hybrid mode (text + image) - NOT USED FOR NOW
        patch_hybrid_mode: bool = False,
        patch_num_text_codebooks: int = 3,
        patch_num_image_codebooks: int = 1,
        
        # Decoder
        decoder_hidden_dim: int = 512,
        decoder_num_layers: int = 2,
        decoder_num_heads: int = 8,
        decoder_dropout: float = 0.1,
        
        # Legacy parameters (for compatibility)
        input_dim: int = 768,
        embed_dim: int = 32,
        hidden_dims: list = [512, 256, 128],
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: str = "rotation_trick",
        n_cat_features: int = 0,
    ):
        super().__init__()
        
        self.use_patch_encoder = use_patch_encoder
        self.num_codebooks = num_codebooks
        self.patch_token_embed_dim = patch_token_embed_dim
        self.output_dim = output_dim
        self.use_diversity_loss = use_diversity_loss
        
        # Encoder
        if use_patch_encoder:
            self.encoder = HierarchicalPatchEncoder(
                token_dim=patch_token_dim,
                num_codebooks=num_codebooks,
                token_embed_dim=patch_token_embed_dim,
                hidden_dim=patch_hidden_dim,
                num_heads=patch_num_heads,
                num_layers=patch_num_layers,
                dropout=patch_dropout,
                use_diversity_loss=use_diversity_loss,
                diversity_weight=diversity_weight,
            )
        else:
            # Legacy: global MLP encoder
            from modules.encoder import MLP
            self.encoder = MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                out_dim=input_dim,
                dropout=0.0,
                normalize=False,
            )
        
        # Product Quantizer
        self.quantizer = ProductQuantize(
            embed_dim=patch_token_embed_dim if use_patch_encoder else input_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            do_kmeans_init=codebook_kmeans_init,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            forward_mode=codebook_mode,
        )
        
        # Decoder
        if use_patch_encoder:
            # Cross-attention decoder (preserves structure!)
            self.decoder = CrossAttentionDecoder(
                input_dim=patch_token_embed_dim,
                num_tokens=num_codebooks,
                hidden_dim=decoder_hidden_dim,
                output_dim=output_dim,
                num_heads=decoder_num_heads,
                num_layers=decoder_num_layers,
                dropout=decoder_dropout,
            )
        else:
            # Legacy decoder
            from modules.decoder import ItemDecoder
            self.decoder = ItemDecoder(
                input_dim=input_dim,
                hidden_dim=decoder_hidden_dim,
                output_dim=output_dim,
            )
        
        # Losses
        self.reconstruction_loss = ReconstructionLoss()
        self.quantize_loss = QuantizeLoss(commitment_weight=commitment_weight)
        
        # Config
        self.config = {
            'use_patch_encoder': use_patch_encoder,
            'num_codebooks': num_codebooks,
            'patch_token_dim': patch_token_dim,
            'patch_token_embed_dim': patch_token_embed_dim,
            'output_dim': output_dim,
        }
    
    def encode(
        self,
        x: Tensor,
        text_patches: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Hierarchical encoding.
        
        Args:
            x: [B, input_dim] - Original global embeddings (for legacy)
            text_patches: [B, N, token_dim] - Text patch embeddings
            text_mask: [B, N] - Attention mask
        
        Returns:
            semantic_tokens: [B, K, token_embed_dim]
            diversity_loss: scalar or None
        """
        if self.use_patch_encoder:
            if text_patches is None:
                raise ValueError("text_patches required for patch encoder mode")
            
            # Hierarchical extraction with sequential conditioning
            semantic_tokens, diversity_loss = self.encoder(text_patches, text_mask)
            # semantic_tokens: [B, K, token_embed_dim]
            # Each token captures different hierarchical level!
        else:
            # Legacy
            semantic_tokens = self.encoder(x)
            semantic_tokens = semantic_tokens.unsqueeze(1)
            diversity_loss = None
        
        return semantic_tokens, diversity_loss
    
    def quantize(
        self,
        semantic_tokens: Tensor,
        gumbel_t: float = 0.2,
    ) -> tuple[Tensor, Tensor]:
        """
        Product quantization (independent codebooks).
        
        Args:
            semantic_tokens: [B, K, token_embed_dim]
            gumbel_t: Gumbel temperature
        
        Returns:
            quantized: [B, K, token_embed_dim]
            codes: [B, K] - Discrete codes (Semantic IDs!)
        """
        # Quantize each token independently with its own codebook
        quantized, codes,_ = self.quantizer(semantic_tokens, gumbel_t)
        # quantized: [B, K, token_embed_dim]
        
        # Extract discrete codes
        # codes = self.quantizer.get_codes(semantic_tokens)
        # codes: [B, K] - Each element in [0, codebook_size)
        
        return quantized, codes
    
    def decode(
        self,
        quantized_tokens: Tensor,
    ) -> Tensor:
        """
        Decode from quantized tokens.
        
        Args:
            quantized_tokens: [B, K, token_embed_dim]
        
        Returns:
            reconstructed: [B, output_dim] - Global embedding
        """
        if self.use_patch_encoder:
            # Cross-attention decoder (preserves structure!)
            reconstructed = self.decoder(quantized_tokens)
            # reconstructed: [B, output_dim]
        else:
            # Legacy: flatten and decode
            B, K, D = quantized_tokens.shape
            quantized_flat = quantized_tokens.reshape(B, K * D)
            reconstructed = self.decoder(quantized_flat)
        
        return reconstructed
    
    def forward(
        self,
        batch,
        gumbel_t: float = 0.2,
        text_patches: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> PqVaeOutput:
        """
        Forward pass with hierarchical encoding and structured decoding.
        
        Args:
            batch: SeqBatch with .x field [B, output_dim]
            gumbel_t: Gumbel temperature
            text_patches: [B, N, token_dim] - Text patches
            text_mask: [B, N] - Attention mask
        """
        # Target: Global embedding
        x_target = batch.x  # [B, output_dim]
        
        # Encode (hierarchical)
        semantic_tokens, diversity_loss = self.encode(
            x_target, text_patches, text_mask
        )
        # semantic_tokens: [B, K, token_embed_dim]
        
        # Quantize (independent)
        quantized_tokens, codes = self.quantize(semantic_tokens, gumbel_t)
        # quantized_tokens: [B, K, token_embed_dim]
        # codes: [B, K] - Semantic IDs!
        
        # Decode (preserves structure)
        x_hat = self.decode(quantized_tokens)
        # x_hat: [B, output_dim]
        
        # Compute losses
        reconstruction_loss = self.reconstruction_loss(x_hat, x_target).mean()
        
        # VQ loss (per token)
        pqvae_loss = 0.0
        for k in range(self.num_codebooks):
            pqvae_loss += self.quantize_loss(
                semantic_tokens[:, k, :],
                quantized_tokens[:, k, :]
            ).mean()
        pqvae_loss = pqvae_loss / self.num_codebooks
        
        # Total loss
        total_loss = reconstruction_loss + pqvae_loss
        if diversity_loss is not None and self.use_diversity_loss:
            total_loss = total_loss + diversity_loss
        
        # Metrics
        # embs_norm = self.quantizer.get_codebook_norms()
        # p_unique_ids = self.quantizer.get_unique_code_usage()
        
        return PqVaeOutput(
            loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            pqvae_loss=pqvae_loss,
            diversity_loss=diversity_loss if diversity_loss is not None else torch.tensor(0.0),
            encoder_output=semantic_tokens,
            quantized_output=quantized_tokens,
            codes=codes,  # Semantic IDs!
            decoder_output=x_hat,
            # embs_norm=embs_norm,
            # p_unique_ids=p_unique_ids,
        )
    
    def get_semantic_ids(
        self,
        batch,
        text_patches: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Extract semantic IDs for items (for downstream task).
        
        Args:
            batch: SeqBatch with items
            text_patches: [B, N, token_dim]
            text_mask: [B, N]
        
        Returns:
            codes: [B, K] - Hierarchical semantic IDs
                           Each row is [code_0, code_1, code_2, code_3]
                           code_0: Category (most abstract)
                           code_3: Details (most specific)
        """
        with torch.no_grad():
            x_target = batch.x
            semantic_tokens, _ = self.encode(x_target, text_patches, text_mask)
            _, codes = self.quantize(semantic_tokens, gumbel_t=0.0)
        return codes
    
    def reconstruct_from_codes(
        self,
        codes: Tensor,
    ) -> Tensor:
        """
        Reconstruct global embedding from semantic IDs.
        
        Args:
            codes: [B, K] - Discrete codes
        
        Returns:
            reconstructed: [B, output_dim]
        """
        with torch.no_grad():
            # Lookup quantized tokens from codes
            quantized_tokens = self.quantizer.decode_codes(codes)
            # quantized_tokens: [B, K, token_embed_dim]
            
            # Decode
            reconstructed = self.decode(quantized_tokens)
        return reconstructed
    
    def load_pretrained(self, path: str):
        """Load pretrained weights"""
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(state['model'], strict=False)