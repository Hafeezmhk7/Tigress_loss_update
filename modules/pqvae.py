import torch
import logging
from data.schemas import SeqBatch
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP, PatchEncoder, HybridEncoder
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.normalize import l2norm
from modules.quantize import ProductQuantize, QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from typing import NamedTuple
from torch import nn
from torch import Tensor

# fetch logger
logger = logging.getLogger("recsys_logger")
torch.set_float32_matmul_precision("high")


class PqVaeOutput(NamedTuple):
    """Output from Product Quantized VAE."""
    embeddings: Tensor  # [B, num_patches, embed_dim]
    sem_ids: Tensor     # [B, num_patches]
    quantize_loss: Tensor
    diversity_loss: Tensor = None


class PqVaeComputedLosses(NamedTuple):
    """Computed losses for training."""
    loss: Tensor
    reconstruction_loss: Tensor
    pqvae_loss: Tensor
    diversity_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


class PqVae(nn.Module, PyTorchModelHubMixin):
    """
    Product Quantized VAE (PQ-VAE) for patch-based semantic encoding.
    
    Instead of Residual Quantization (RQ):
        Sequential: z₀ → z₁ = VQ(x - z₀) → z₂ = VQ(x - z₀ - z₁)
    
    We use Product Quantization (PQ):
        Parallel: z₀ = VQ₀(patch₀), z₁ = VQ₁(patch₁), z₂ = VQ₂(patch₂), ...
    
    Benefits:
    - Natural fit for patch-based encoding (each patch = semantic aspect)
    - Parallel optimization (no sequential dependencies)
    - True compositional semantics (no forced hierarchy)
    - Simpler training (no residual accumulation)
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.ROTATION_TRICK,
        num_codebooks: int = 4,  # Number of patches/codebooks
        commitment_weight: float = 0.25,
        n_cat_features: int = 0,
        # Patch-based parameters
        use_patch_encoder: bool = False,
        patch_num_patches: int = 4,
        patch_hidden_dim: int = 256,
        patch_num_heads: int = 4,
        patch_dropout: float = 0.1,
        patch_diversity_weight: float = 0.1,
        patch_hybrid_mode: bool = False,
    ) -> None:
        self._config = locals()

        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features
        
        # Patch-based parameters
        self.use_patch_encoder = use_patch_encoder
        self.patch_num_patches = patch_num_patches
        self.patch_hybrid_mode = patch_hybrid_mode

        # Product Quantizer (independent codebooks for each patch)
        self.quantizer = ProductQuantize(
            num_codebooks=num_codebooks,
            embed_dim=embed_dim,
            codebook_size=codebook_size,
            do_kmeans_init=codebook_kmeans_init,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            commitment_weight=commitment_weight,
            forward_mode=codebook_mode,
        )

        # Encoder: Choose between patch-based or global
        if use_patch_encoder:
            if patch_hybrid_mode:
                # Hybrid: text patches + image global
                logger.info("Using HybridEncoder (text patches + image global)")
                self.encoder = HybridEncoder(
                    text_token_dim=input_dim,
                    image_global_dim=input_dim,
                    num_text_patches=patch_num_patches,
                    patch_dim=embed_dim,
                    hidden_dim=patch_hidden_dim,
                    num_heads=patch_num_heads,
                    dropout=patch_dropout,
                    diversity_loss_weight=patch_diversity_weight,
                )
                self.total_patches = patch_num_patches + 1
            else:
                # Text-only patches
                logger.info(f"Using PatchEncoder with {patch_num_patches} patches")
                self.encoder = PatchEncoder(
                    token_dim=input_dim,
                    num_patches=patch_num_patches,
                    patch_dim=embed_dim,
                    hidden_dim=patch_hidden_dim,
                    num_heads=patch_num_heads,
                    dropout=patch_dropout,
                    diversity_loss_weight=patch_diversity_weight,
                )
                self.total_patches = patch_num_patches
        else:
            # Original global MLP encoder
            logger.info("Using original MLP encoder (global)")
            # For global mode, create multiple patches from global embedding
            self.encoder = MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                out_dim=embed_dim * num_codebooks,  # Output for all patches
                normalize=codebook_normalize,
            )
            self.total_patches = num_codebooks

        # Decoder: Reconstruct from sum of patches
        self.decoder = MLP(
            input_dim=embed_dim,  # Sum of all patches
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=True,
        )

        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features)
            if n_cat_features != 0
            else ReconstructionLoss()
        )

    @cached_property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        logger.info(f"Loaded PQ-VAE Model | Iteration {state['iteration']}")

    def encode(self, x: Tensor, text_patches: Tensor = None, text_mask: Tensor = None) -> tuple:
        """
        Encode input to patch representations.
        
        Args:
            x: [B, input_dim] - Global features (for backward compatibility)
            text_patches: [B, N_tokens, token_dim] - Token embeddings (for patch encoder)
            text_mask: [B, N_tokens] - Attention mask (for patch encoder)
            
        Returns:
            patches: [B, num_codebooks, embed_dim]
            diversity_loss: scalar or None
        """
        if self.use_patch_encoder:
            if self.patch_hybrid_mode:
                # Hybrid mode: text patches + image global
                patches, diversity_loss = self.encoder(
                    text_token_embeddings=text_patches,
                    text_attention_mask=text_mask,
                    image_global_embedding=x,
                )
                return patches, diversity_loss
            else:
                # Text-only patches
                if text_patches is None:
                    raise ValueError("text_patches required when use_patch_encoder=True")
                patches, diversity_loss = self.encoder(text_patches, text_mask)
                return patches, diversity_loss
        else:
            # Original global encoding - reshape to patches
            encoded = self.encoder(x)  # [B, embed_dim * num_codebooks]
            patches = encoded.reshape(-1, self.num_codebooks, self.embed_dim)
            return patches, None

    def decode(self, patches_sum: Tensor) -> Tensor:
        """
        Decode from sum of all patches.
        
        Args:
            patches_sum: [B, embed_dim] - Sum of all quantized patches
        
        Returns:
            decoded: [B, input_dim]
        """
        return self.decoder(patches_sum)

    def get_semantic_ids(
        self, 
        x: Tensor, 
        gumbel_t: float = 0.001,
        text_patches: Tensor = None,
        text_mask: Tensor = None,
    ) -> PqVaeOutput:
        """
        Get semantic IDs through product quantization.
        
        Each patch is quantized independently (no residuals).
        """
        # Encode: [B, num_codebooks, embed_dim]
        patches, diversity_loss = self.encode(x, text_patches, text_mask)
        
        # Product quantization (parallel, independent)
        quantized = self.quantizer(patches, temperature=gumbel_t)
        
        return PqVaeOutput(
            embeddings=quantized.embeddings,  # [B, num_codebooks, embed_dim]
            sem_ids=quantized.ids,            # [B, num_codebooks]
            quantize_loss=quantized.loss,
            diversity_loss=diversity_loss,
        )

    @torch.compile(mode="reduce-overhead")
    def forward(
        self, 
        batch: SeqBatch, 
        gumbel_t: float,
        text_patches: Tensor = None,
        text_mask: Tensor = None,
    ) -> PqVaeComputedLosses:
        """
        Forward pass with loss computation.
        """
        x = batch.x
        quantized = self.get_semantic_ids(x, gumbel_t, text_patches, text_mask)
        
        # Sum all patches for reconstruction
        # [B, num_codebooks, embed_dim] → [B, embed_dim]
        patches_sum = quantized.embeddings.sum(dim=1)
        
        # Decode
        x_hat = self.decode(patches_sum)
        x_hat = torch.cat(
            [l2norm(x_hat[..., : -self.n_cat_feats]), x_hat[..., -self.n_cat_feats :]],
            axis=-1,
        )

        # Compute losses
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        pqvae_loss = quantized.quantize_loss
        
        # Add diversity loss if using patch encoder
        diversity_loss_value = quantized.diversity_loss if quantized.diversity_loss is not None else torch.tensor(0.0, device=x.device)
        
        loss = (reconstruction_loss + pqvae_loss).mean() + diversity_loss_value

        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = quantized.embeddings.norm(dim=-1)  # [B, num_codebooks]
            
            # Check uniqueness of semantic IDs
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(quantized.sem_ids, "b d -> b 1 d")
                        == rearrange(quantized.sem_ids, "b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return PqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss.mean(),
            pqvae_loss=pqvae_loss.mean(),
            diversity_loss=diversity_loss_value,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )


# Backward compatibility alias
RqVae = PqVae  # Allow old code to work