import gin
import torch
from dataclasses import dataclass
from modules.encoder import HierarchicalPatchEncoder, CrossAttentionDecoder, PatchReconstructionDecoder
from modules.quantize import ProductQuantize
from modules.loss import ReconstructionLoss, QuantizeLoss
from torch import nn, Tensor
from typing import Optional, List


@dataclass
class PqVaeOutput:
    """Output from PQ-VAE forward pass"""
    loss: Tensor
    patch_reconstruction_loss: Tensor
    global_reconstruction_loss: Tensor
    pqvae_loss: Tensor
    diversity_loss: Optional[Tensor]
    encoder_output: Tensor          # [B, K, D]
    quantized_output: Tensor        # [B, K, D]
    codes: Tensor                   # [B, K] - Semantic IDs!
    patch_decoder_output: Tensor    # [B, N, patch_dim]
    global_decoder_output: Tensor   # [B, output_dim]
    embs_norm: Tensor
    p_unique_ids: Tensor


@dataclass
class SemanticIdsOutput:
    """Output from get_semantic_ids (for tokenizer compatibility)"""
    sem_ids: Tensor  # [B, K] - Hierarchical semantic IDs


@gin.configurable
class PqVae(nn.Module):
    """
    Hierarchical Product Quantization VAE with Dual Reconstruction
    
    **Dual Reconstruction Strategy:**
    - **Main Task**: Patch reconstruction (preserves fine-grained structure)
    - **Auxiliary Task**: Global reconstruction (preserves overall semantics)
    
    This ensures semantic IDs capture both:
    1. Fine-grained details (patch-level)
    2. Global semantics (item-level)
    """
    
    def __init__(
        self,
        # Reconstruction targets
        patch_dim: int = 1024,           # Patch dimension (e.g., 1024 from sentence-t5-xl)
        num_patches: int = 256,          # Number of patches (max_seq_length)
        global_dim: int = 768,           # Global embedding dimension
        
        # Encoder
        use_patch_encoder: bool = True,
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
        
        # Decoders
        decoder_hidden_dim: int = 512,
        decoder_num_layers: int = 2,
        decoder_num_heads: int = 8,
        decoder_dropout: float = 0.1,
        
        # Loss weights
        patch_recon_weight: float = 1.0,      # Main task
        global_recon_weight: float = 0.2,     # Auxiliary task (low weight)
        
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
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.global_dim = global_dim
        self.use_diversity_loss = use_diversity_loss
        
        self.patch_recon_weight = patch_recon_weight
        self.global_recon_weight = global_recon_weight
        
        # Encoder
        if use_patch_encoder:
            self.encoder = HierarchicalPatchEncoder(
                token_dim=patch_dim,
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
            num_codebooks=num_codebooks,
            embed_dim=patch_token_embed_dim if use_patch_encoder else input_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            do_kmeans_init=codebook_kmeans_init,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            forward_mode=codebook_mode,
        )
        
        # Decoder 1: Patch Reconstruction (MAIN TASK)
        if use_patch_encoder:
            self.patch_decoder = PatchReconstructionDecoder(
                num_tokens=num_codebooks,
                token_embed_dim=patch_token_embed_dim,
                num_patches=num_patches,
                patch_dim=patch_dim,
                hidden_dim=decoder_hidden_dim,
                num_heads=decoder_num_heads,
                num_layers=decoder_num_layers,
                dropout=decoder_dropout,
            )
        else:
            self.patch_decoder = None
        
        # Decoder 2: Global Reconstruction (AUXILIARY TASK)
        if use_patch_encoder:
            self.global_decoder = CrossAttentionDecoder(
                input_dim=patch_token_embed_dim,
                num_tokens=num_codebooks,
                hidden_dim=decoder_hidden_dim,
                output_dim=global_dim,
                num_heads=decoder_num_heads,
                num_layers=decoder_num_layers,
                dropout=decoder_dropout,
            )
        else:
            # Legacy decoder
            from modules.encoder import MLP
            self.global_decoder = MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims[::-1],
                out_dim=global_dim,
                normalize=True,
            )
        
        # Losses
        self.patch_reconstruction_loss = ReconstructionLoss()
        self.global_reconstruction_loss = ReconstructionLoss()
        self.quantize_loss = QuantizeLoss(commitment_weight=commitment_weight)
        
        # Config
        self.config = {
            'use_patch_encoder': use_patch_encoder,
            'num_codebooks': num_codebooks,
            'patch_token_dim': patch_dim,
            'patch_token_embed_dim': patch_token_embed_dim,
            'global_dim': global_dim,
            'patch_recon_weight': patch_recon_weight,
            'global_recon_weight': global_recon_weight,
        }
    
    @property
    def device(self) -> torch.device:
        """Get device from model parameters"""
        return next(self.parameters()).device
    
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
        result = self.quantizer(semantic_tokens, gumbel_t)
        quantized = result.embeddings  # [B, K, token_embed_dim]
        codes = result.ids             # [B, K]
        return quantized, codes
    
    def decode_patches(
        self,
        quantized_tokens: Tensor,
    ) -> Tensor:
        """
        Decode to patches (MAIN TASK).
        
        Args:
            quantized_tokens: [B, K, token_embed_dim]
        
        Returns:
            reconstructed_patches: [B, N, patch_dim]
        """
        if self.patch_decoder is None:
            raise ValueError("Patch decoder not available in legacy mode")
        return self.patch_decoder(quantized_tokens)
    
    def decode_global(
        self,
        quantized_tokens: Tensor,
    ) -> Tensor:
        """
        Decode to global (AUXILIARY TASK).
        
        Args:
            quantized_tokens: [B, K, token_embed_dim]
        
        Returns:
            reconstructed_global: [B, output_dim]
        """
        if self.use_patch_encoder:
            return self.global_decoder(quantized_tokens)
        else:
            # Legacy: flatten and decode
            B, K, D = quantized_tokens.shape
            quantized_flat = quantized_tokens.reshape(B, K * D)
            return self.global_decoder(quantized_flat)
    
    def forward(
        self,
        batch,
        gumbel_t: float = 0.2,
        text_patches: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> PqVaeOutput:
        """
        Forward pass with dual reconstruction.
        
        Args:
            batch: SeqBatch with .x field [B, output_dim]
            gumbel_t: Gumbel temperature
            text_patches: [B, N, token_dim] - Text patches (MAIN reconstruction target)
            text_mask: [B, N] - Attention mask
        """
        # Targets
        patches_target = text_patches  # [B, N, patch_dim] - MAIN
        global_target = batch.x        # [B, global_dim] - AUXILIARY
        
        # Encode (hierarchical)
        semantic_tokens, diversity_loss = self.encode(
            global_target, text_patches, text_mask
        )
        # semantic_tokens: [B, K, token_embed_dim]
        
        # Quantize (independent)
        quantized_tokens, codes = self.quantize(semantic_tokens, gumbel_t)
        # quantized_tokens: [B, K, token_embed_dim]
        # codes: [B, K] - Semantic IDs!
        
        # Decode patches (MAIN TASK)
        patches_hat = self.decode_patches(quantized_tokens)
        # patches_hat: [B, N, patch_dim]
        
        # Decode global (AUXILIARY TASK)
        global_hat = self.decode_global(quantized_tokens)
        # global_hat: [B, global_dim]
        
        # Compute losses
        # 1. Patch reconstruction (MAIN)
        patch_recon_loss = self.patch_reconstruction_loss(
            patches_hat, patches_target
        ).mean()
        
        # 2. Global reconstruction (AUXILIARY, low weight)
        global_recon_loss = self.global_reconstruction_loss(
            global_hat, global_target
        ).mean()
        
        # 3. VQ loss (per token)
        pqvae_loss = 0.0
        for k in range(self.num_codebooks):
            pqvae_loss += self.quantize_loss(
                semantic_tokens[:, k, :],
                quantized_tokens[:, k, :]
            ).mean()
        pqvae_loss = pqvae_loss / self.num_codebooks
        
        # Total loss
        total_loss = (
            self.patch_recon_weight * patch_recon_loss +
            self.global_recon_weight * global_recon_loss +
            pqvae_loss
        )
        if diversity_loss is not None and self.use_diversity_loss:
            total_loss = total_loss + diversity_loss
        
        # Metrics
        embs_norm = self.quantizer.get_codebook_norms()
        p_unique_ids = self.quantizer.get_unique_code_usage()
        
        return PqVaeOutput(
            loss=total_loss,
            patch_reconstruction_loss=patch_recon_loss,
            global_reconstruction_loss=global_recon_loss,
            pqvae_loss=pqvae_loss,
            diversity_loss=diversity_loss if diversity_loss is not None else torch.tensor(0.0),
            encoder_output=semantic_tokens,
            quantized_output=quantized_tokens,
            codes=codes,  # Semantic IDs!
            patch_decoder_output=patches_hat,
            global_decoder_output=global_hat,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )
    
    def get_semantic_ids(
        self,
        batch,
        text_patches: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> "SemanticIdsOutput":
        """
        Extract semantic IDs for items (for downstream task).
        
        Args:
            batch: SeqBatch with items OR Tensor [B, input_dim] directly
            text_patches: [B, N, token_dim] (optional, auto-extracted if batch has it)
            text_mask: [B, N] (optional, auto-extracted if batch has it)
        
        Returns:
            SemanticIdsOutput with:
                sem_ids: [B, K] - Hierarchical semantic IDs
        """
        with torch.no_grad():
            # Handle both batch object and direct tensor input
            if isinstance(batch, Tensor):
                x_target = batch
            else:
                x_target = batch.x
                # Auto-extract patches if available and not explicitly provided
                if text_patches is None and hasattr(batch, 'text_patches'):
                    text_patches = batch.text_patches
                if text_mask is None and hasattr(batch, 'text_masks'):
                    text_mask = batch.text_masks
            
            semantic_tokens, _ = self.encode(x_target, text_patches, text_mask)
            _, codes = self.quantize(semantic_tokens, gumbel_t=0.0)
        
        return SemanticIdsOutput(sem_ids=codes)
    
    def reconstruct_from_codes(
        self,
        codes: Tensor,
        target: str = "global",
    ) -> Tensor:
        """
        Reconstruct from semantic IDs.
        
        Args:
            codes: [B, K] - Discrete codes
            target: "patches" or "global"
        
        Returns:
            reconstructed: [B, N, patch_dim] or [B, output_dim]
        """
        with torch.no_grad():
            # Lookup quantized tokens from codes
            quantized_tokens = self.quantizer.decode_codes(codes)
            
            # Decode
            if target == "patches":
                reconstructed = self.decode_patches(quantized_tokens)
            elif target == "global":
                reconstructed = self.decode_global(quantized_tokens)
            else:
                raise ValueError(f"Unknown target: {target}")
        
        return reconstructed
    
    def load_pretrained(self, path: str):
        """Load pretrained weights"""
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(state['model'], strict=False)