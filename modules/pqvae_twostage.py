import torch
from dataclasses import dataclass
from modules.encoder import HierarchicalPatchEncoder, CrossAttentionDecoder, PatchReconstructionDecoder
from modules.quantize import ProductQuantize
from modules.loss import ReconstructionLoss, QuantizeLoss, SequenceContrastiveLoss
from torch import nn, Tensor
from typing import Optional, List


@dataclass
class PqVaeOutput:
    """Output from PQ-VAE forward pass"""
    loss: Tensor
    patch_reconstruction_loss: Tensor
    global_reconstruction_loss: Tensor
    pqvae_loss: Tensor
    sequence_contrastive_loss: Tensor
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
    """Output from get_semantic_ids"""
    sem_ids: Tensor  # [B, K]


class PqVaeTwoStage(nn.Module):
    """
    Two-Stage Hierarchical PQ-VAE for Semantic ID Generation
    
    **Stage 1: Item-Level Training**
    - Patch reconstruction (main task)
    - Global reconstruction (auxiliary task)
    - Codebook quantization
    - Diversity loss
    
    **Stage 2: Sequence-Level Training**
    - Continue Stage 1 losses
    - Add sequence contrastive loss
    - Enforce behavioral patterns
    """
    
    def __init__(
        self,
        # Reconstruction targets
        patch_dim: int = 1024,           # Patch dimension
        num_patches: int = 256,          # Number of patches
        global_dim: int = 768,           # Global embedding dimension
        
        # Encoder
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
        patch_recon_weight: float = 1.0,
        global_recon_weight: float = 0.2,
        sequence_contrastive_weight: float = 0.5,
        
        # Sequence contrastive
        contrastive_temperature: float = 0.1,
        num_negatives: int = 16,
        
        # Legacy
        **kwargs
    ):
        super().__init__()
        
        self.num_codebooks = num_codebooks
        self.patch_token_embed_dim = patch_token_embed_dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.global_dim = global_dim
        self.use_diversity_loss = use_diversity_loss
        
        self.patch_recon_weight = patch_recon_weight
        self.global_recon_weight = global_recon_weight
        self.sequence_contrastive_weight = sequence_contrastive_weight
        
        # Training stage
        self.current_stage = 1  # 1 or 2
        
        # Encoder
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
        
        # Product Quantizer
        self.quantizer = ProductQuantize(
            num_codebooks=num_codebooks,
            embed_dim=patch_token_embed_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            do_kmeans_init=codebook_kmeans_init,
        )
        
        # Decoder 1: Patch Reconstruction (MAIN)
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
        
        # Decoder 2: Global Reconstruction (AUXILIARY)
        self.global_decoder = CrossAttentionDecoder(
            input_dim=patch_token_embed_dim,
            num_tokens=num_codebooks,
            hidden_dim=decoder_hidden_dim,
            output_dim=global_dim,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
        )
        
        # Losses
        self.patch_reconstruction_loss = ReconstructionLoss()
        self.global_reconstruction_loss = ReconstructionLoss()
        self.quantize_loss = QuantizeLoss(commitment_weight=commitment_weight)
        self.sequence_contrastive_loss = SequenceContrastiveLoss(
            temperature=contrastive_temperature,
            num_negatives=num_negatives,
        )
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def set_stage(self, stage: int):
        """Set training stage (1 or 2)"""
        assert stage in [1, 2]
        self.current_stage = stage
        print(f"✅ Switched to Stage {stage}")
    
    def encode(
        self,
        text_patches: Tensor,
        text_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Hierarchical encoding"""
        semantic_tokens, diversity_loss = self.encoder(text_patches, text_mask)
        return semantic_tokens, diversity_loss
    
    def quantize(
        self,
        semantic_tokens: Tensor,
        gumbel_t: float = 0.2,
    ) -> tuple[Tensor, Tensor]:
        """Product quantization"""
        result = self.quantizer(semantic_tokens, gumbel_t)
        quantized = result.embeddings
        codes = result.ids
        return quantized, codes
    
    def decode_patches(
        self,
        quantized_tokens: Tensor,
    ) -> Tensor:
        """Decode to patches (MAIN)"""
        return self.patch_decoder(quantized_tokens)
    
    def decode_global(
        self,
        quantized_tokens: Tensor,
    ) -> Tensor:
        """Decode to global (AUXILIARY)"""
        return self.global_decoder(quantized_tokens)
    
    def forward(
        self,
        batch,
        gumbel_t: float = 0.2,
        text_patches: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        # Stage 2 specific
        anchor_codes: Optional[Tensor] = None,
        positive_codes: Optional[Tensor] = None,
        negative_codes: Optional[Tensor] = None,
    ) -> PqVaeOutput:
        """
        Forward pass with stage-aware loss computation.
        
        Stage 1: Patch + global reconstruction
        Stage 2: Stage 1 + sequence contrastive
        """
        # Targets
        patches_target = text_patches  # [B, N, patch_dim]
        global_target = batch.x        # [B, global_dim]
        
        # Encode
        semantic_tokens, diversity_loss = self.encode(patches_target, text_mask)
        
        # Quantize
        quantized_tokens, codes = self.quantize(semantic_tokens, gumbel_t)
        
        # Decode
        patches_hat = self.decode_patches(quantized_tokens)
        global_hat = self.decode_global(quantized_tokens)
        
        # ===== STAGE 1 LOSSES =====
        
        # Patch reconstruction (MAIN)
        patch_recon_loss = self.patch_reconstruction_loss(
            patches_hat, patches_target
        ).mean()
        
        # Global reconstruction (AUXILIARY)
        global_recon_loss = self.global_reconstruction_loss(
            global_hat, global_target
        ).mean()
        
        # VQ loss (per token)
        pqvae_loss = 0.0
        for k in range(self.num_codebooks):
            pqvae_loss += self.quantize_loss(
                semantic_tokens[:, k, :],
                quantized_tokens[:, k, :]
            ).mean()
        pqvae_loss = pqvae_loss / self.num_codebooks
        
        # Total Stage 1 loss
        stage1_loss = (
            self.patch_recon_weight * patch_recon_loss +
            self.global_recon_weight * global_recon_loss +
            pqvae_loss
        )
        if diversity_loss is not None and self.use_diversity_loss:
            stage1_loss = stage1_loss + diversity_loss
        
        # ===== STAGE 2 LOSS =====
        
        seq_contrastive_loss = torch.tensor(0.0, device=self.device)
        
        if self.current_stage == 2 and anchor_codes is not None:
            # Get embeddings from codes
            anchor_emb = self.quantizer.decode_codes(anchor_codes)  # [B, K, D]
            positive_emb = self.quantizer.decode_codes(positive_codes)
            negative_emb = self.quantizer.decode_codes(negative_codes)  # [B, num_neg, K, D]
            
            # Flatten to [B, K*D]
            B = anchor_emb.shape[0]
            anchor_flat = anchor_emb.reshape(B, -1)
            positive_flat = positive_emb.reshape(B, -1)
            negative_flat = negative_emb.reshape(B, negative_emb.shape[1], -1)
            
            # Compute contrastive loss
            seq_contrastive_loss = self.sequence_contrastive_loss(
                anchor_flat, positive_flat, negative_flat
            )
        
        # Total loss
        total_loss = stage1_loss
        if self.current_stage == 2:
            total_loss = total_loss + self.sequence_contrastive_weight * seq_contrastive_loss
        
        # Metrics
        embs_norm = self.quantizer.get_codebook_norms()
        p_unique_ids = self.quantizer.get_unique_code_usage()
        
        return PqVaeOutput(
            loss=total_loss,
            patch_reconstruction_loss=patch_recon_loss,
            global_reconstruction_loss=global_recon_loss,
            pqvae_loss=pqvae_loss,
            sequence_contrastive_loss=seq_contrastive_loss,
            diversity_loss=diversity_loss if diversity_loss is not None else torch.tensor(0.0),
            encoder_output=semantic_tokens,
            quantized_output=quantized_tokens,
            codes=codes,
            patch_decoder_output=patches_hat,
            global_decoder_output=global_hat,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )
    
    def get_semantic_ids(
        self,
        batch,
        text_patches: Tensor,
        text_mask: Optional[Tensor] = None,
    ) -> SemanticIdsOutput:
        """Extract semantic IDs"""
        with torch.no_grad():
            semantic_tokens, _ = self.encode(text_patches, text_mask)
            _, codes = self.quantize(semantic_tokens, gumbel_t=0.0)
        return SemanticIdsOutput(sem_ids=codes)
    
    def load_pretrained(self, path: str):
        """Load pretrained weights"""
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(state['model'], strict=False)