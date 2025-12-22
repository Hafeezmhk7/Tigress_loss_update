import torch
import torch.nn.functional as F
import logging
from data.schemas import SeqBatch
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss, ReconstructionLoss
from modules.normalize import l2norm
from modules.quantize import Quantize, QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List, NamedTuple, Optional
from torch import nn
from torch import Tensor
from modules.transformer.attention import MultiHeadAttention
import torch._dynamo
torch._dynamo.config.suppress_errors = True

logger = logging.getLogger("recsys_logger")
torch.set_float32_matmul_precision("high")


class ContrastiveRqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor
    text_emb: Optional[Tensor] = None  # For contrastive loss
    image_emb: Optional[Tensor] = None  # For contrastive loss


class ContrastiveRqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    contrastive_loss: Tensor
    text_image_alignment: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


class ContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for multimodal alignment
    Based on CLIP - learns that text and image of same item should be similar
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, text_emb: Tensor, image_emb: Tensor) -> Tensor:
        """
        text_emb: [batch, dim]
        image_emb: [batch, dim]
        """
        # Normalize embeddings
        text_emb = F.normalize(text_emb, dim=-1)
        image_emb = F.normalize(image_emb, dim=-1)
        
        # Compute similarity matrix
        logits = (text_emb @ image_emb.T) / self.temperature
        
        # Labels are diagonal (each text matches its corresponding image)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Symmetric loss (text->image and image->text)
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)
        
        return (loss_t2i + loss_i2t) / 2


class ReconstructionContrastiveLoss(nn.Module):
    """
    Contrastive loss on reconstructed embeddings (CoST paper)
    Reconstructed vectors should be closer to their input than to other items
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, input_emb: Tensor, recon_emb: Tensor) -> Tensor:
        """
        input_emb: [batch, dim] - original embeddings
        recon_emb: [batch, dim] - reconstructed embeddings
        """
        # Normalize
        input_emb = F.normalize(input_emb, dim=-1)
        recon_emb = F.normalize(recon_emb, dim=-1)
        
        # Similarity matrix: [batch, batch]
        logits = (recon_emb @ input_emb.T) / self.temperature
        
        # Each reconstruction should match its own input
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class ContrastiveMultimodalRqVae(nn.Module, PyTorchModelHubMixin):
    """
    Contrastive RQ-VAE - separates fusion strategy from contrastive learning
    
    Key innovations:
    1. INDEPENDENT encoders for text and image (even in sum/concat modes)
    2. CLIP-style contrastive loss (text <-> image alignment)
    3. CoST-style reconstruction contrastive loss (neighborhood preservation)
    4. Multiple fusion strategies (sum/concat/cross-attn)
    
    Contrastive learning works with ANY fusion mode!
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
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        use_contrastive: bool = True,
        contrastive_weight: float = 0.5,
        recon_contrastive_weight: float = 0.3,
        n_cat_features: int = 18,
        use_cross_attn: bool = False,  
        attn_heads: int = 8,
        mixed_precision_type: str = "fp16",
    ) -> None:
        self._config = locals()
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.recon_contrastive_weight = recon_contrastive_weight
        self.n_cat_feats = n_cat_features
        self.use_cross_attn = use_cross_attn
        
        # SEPARATE encoders for text and image (for contrastive learning)
        # Even in sum/concat modes, we process them separately first
        self.text_encoder = MLP(
            input_dim=input_dim,  # 768 for T5
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )
        
        self.image_encoder = MLP(
            input_dim=768,  # CLIP image features
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )
        
        # Fusion layer (strategy depends on use_cross_attn flag)
        if use_cross_attn:
            # Cross-attention fusion
            self.img_proj = nn.Linear(768, embed_dim)
            self.cross_attn = MultiHeadAttention(
                d_in=embed_dim,
                d_out=embed_dim,
                num_heads=attn_heads,
                cross_attn=True,
                dropout=0.1,
            )
            self.fusion_norm = nn.LayerNorm(embed_dim)
        else:
            # Simple fusion (sum or gated)
            self.fusion_gate = nn.Linear(embed_dim * 2, 1)  # Learn to weight text vs image
        
        # Residual quantization layers
        self.layers = nn.ModuleList([
            Quantize(
                embed_dim=embed_dim,
                n_embed=codebook_size,
                forward_mode=codebook_mode,
                do_kmeans_init=codebook_kmeans_init,
                codebook_normalize=i == 0 and codebook_normalize,
                sim_vq=codebook_sim_vq,
                commitment_weight=commitment_weight,
            )
            for i in range(n_layers)
        ])
        
        # Decoder (reconstructs text embeddings)
        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[::-1],
            out_dim=input_dim,  # Reconstruct to text space
            normalize=True,
        )
        
        # Contrastive losses
        if use_contrastive:
            self.cross_modal_contrastive = ContrastiveLoss(temperature=0.07)
            self.recon_contrastive = ReconstructionContrastiveLoss(temperature=0.1)
        
        # Reconstruction loss
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
        return next(self.text_encoder.parameters()).device
    
    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        logger.info(f"Loaded Contrastive RQ-VAE Model | Iteration {state['iteration']}")
    
    def encode(
        self, 
        x: Tensor, 
        x_cross: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Encode with separate text/image processing for contrastive learning
        
        Args:
            x: text embeddings [batch, 768]
            x_cross: image embeddings [batch, 768] (optional)
            
        Returns:
            (fused_embedding, text_embedding, image_embedding)
        """
        # Always encode text
        text_emb = self.text_encoder(x)
        
        # If no image, just return text
        if x_cross is None:
            return text_emb, None, None
        
        # Encode image separately
        x_cross = x_cross.to(text_emb.device, dtype=text_emb.dtype)
        image_emb = self.image_encoder(x_cross)
        
        # Fuse based on strategy
        if self.use_cross_attn:
            # Cross-attention fusion
            text_emb_proj = self.img_proj.to(dtype=text_emb.dtype)(text_emb)
            fused = text_emb_proj + self.cross_attn(
                x=text_emb_proj.unsqueeze(1),
                x_kv=image_emb.unsqueeze(1),
                jagged=False
            ).squeeze(1)
            fused = self.fusion_norm(fused)
        else:
            # Learned gating: how much to weight text vs image
            gate = torch.sigmoid(self.fusion_gate(torch.cat([text_emb, image_emb], dim=-1)))
            fused = gate * text_emb + (1 - gate) * image_emb
        
        return fused, text_emb, image_emb
    
    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)
    
    def get_semantic_ids(
        self,
        x: Tensor,
        x_image: Optional[Tensor] = None,
        gumbel_t: float = 0.001
    ) -> ContrastiveRqVaeOutput:
        """Get semantic IDs with contrastive embeddings"""
        # Encode with separate text/image embeddings
        fused_emb, text_emb, image_emb = self.encode(x, x_cross=x_image)
        
        res = fused_emb
        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []
        
        # Residual quantization
        for layer in self.layers:
            residuals.append(res)
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss += quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb
            sem_ids.append(id)
            embs.append(emb)
        
        return ContrastiveRqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss,
            text_emb=text_emb,
            image_emb=image_emb,
        )
    
    @torch.compile(mode="reduce-overhead")
    def forward(
        self, 
        batch: SeqBatch, 
        gumbel_t: float = 0.001
    ) -> ContrastiveRqVaeComputedLosses:
        x, x_image = batch.x, batch.x_image
        
        # Get quantized representations
        quantized = self.get_semantic_ids(x, x_image=x_image, gumbel_t=gumbel_t)
        embs = quantized.embeddings
        
        # Decode
        embeddings_sum = embs.sum(axis=-1)
        x_hat = self.decode(embeddings_sum)
        x_hat = torch.cat(
            [l2norm(x_hat[..., :-self.n_cat_feats]), x_hat[..., -self.n_cat_feats:]],
            axis=-1,
        )
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(x_hat, x)
        
        # RQ-VAE quantization loss
        rqvae_loss = quantized.quantize_loss
        
        # Contrastive losses
        contrastive_loss = torch.tensor(0.0, device=x.device)
        text_image_alignment = torch.tensor(0.0, device=x.device)
        
        if self.use_contrastive and x_image is not None and quantized.text_emb is not None:
            # 1. Cross-modal contrastive (CLIP-style)
            # Learn that text and image of same item should be similar
            cross_modal_loss = self.cross_modal_contrastive(
                quantized.text_emb, 
                quantized.image_emb
            )
            
            # 2. Reconstruction contrastive (CoST-style)
            # Learn that reconstruction should match input, not other items
            fused_input, _, _ = self.encode(x, x_cross=x_image)
            recon_contrastive_loss = self.recon_contrastive(
                fused_input,
                embeddings_sum
            )
            
            contrastive_loss = (
                self.contrastive_weight * cross_modal_loss +
                self.recon_contrastive_weight * recon_contrastive_loss
            )
            
            # Compute alignment metric
            with torch.no_grad():
                text_norm = F.normalize(quantized.text_emb, dim=-1)
                image_norm = F.normalize(quantized.image_emb, dim=-1)
                text_image_alignment = (text_norm * image_norm).sum(dim=-1).mean()
        
        # Total loss
        total_loss = (recon_loss + rqvae_loss + contrastive_loss).mean()
        
        with torch.no_grad():
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(quantized.sem_ids, "b d -> b 1 d")
                        == rearrange(quantized.sem_ids, "b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]
        
        return ContrastiveRqVaeComputedLosses(
            loss=total_loss,
            reconstruction_loss=recon_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            contrastive_loss=contrastive_loss,
            text_image_alignment=text_image_alignment,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )