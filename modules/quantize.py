import gin
import torch

from distributions.gumbel import gumbel_softmax_sample
from einops import rearrange
from enum import Enum
from init.kmeans import kmeans_init_
from modules.loss import QuantizeLoss
from modules.normalize import L2NormalizationLayer
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F


@gin.constants_from_enum
class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, "b d -> b 1 d")
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e
        - 2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d"))
        + 2
        * (
            e
            @ rearrange(u, "b d -> b d 1").detach()
            @ rearrange(q, "b d -> b 1 d").detach()
        )
    ).squeeze()


class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        sim_vq: bool = False,  # https://arxiv.org/pdf/2411.02038
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity(),
        )

        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    @torch.no_grad
    def _kmeans_init(self, x) -> None:
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x, temperature=None):
        """
        Forward pass with optional temperature for Gumbel-Softmax.
        """
        if temperature is not None and temperature > 0:
            # Compute distances and soft assignments
            distances = torch.cdist(x, self.embedding.weight, p=2.0)  # [B, codebook_size]
            logits = -distances / temperature
            soft_codes = F.softmax(logits, dim=-1)  # [B, codebook_size]
            
            # Soft quantization: weighted sum
            quantized = torch.matmul(soft_codes, self.embedding.weight)  # [B, D]
            
            # Hard codes for metrics (not used in gradient)
            codes = torch.argmin(distances, dim=-1)  # [B]
            
            # ✅ CORRECT FIX: Use mean reduction everywhere
            # Codebook loss: bring codebook closer to encoder outputs
            codebook_loss = F.mse_loss(quantized, x.detach(), reduction='mean')
            
            # Commitment loss: bring encoder outputs closer to codebook  
            commitment_loss = F.mse_loss(x, quantized.detach(), reduction='mean')
            
            # Combined VQ loss with proper weighting
            vq_loss = codebook_loss + self.quantize_loss.commitment_weight * commitment_loss
            
            # Straight-through estimator for gradients
            quantized = x + (quantized - x).detach()
            
            return QuantizeOutput(
                embeddings=quantized,
                ids=codes,
                loss=vq_loss,  # ✅ Now a proper scalar
            )
        
        else:
            raise Exception("Unsupported Quantize forward mode.")

class ProductQuantize(nn.Module):
    """
    Product Quantization: Independent codebooks for each subspace.
    
    Instead of residual quantization (sequential):
        z₀ = VQ(x)
        z₁ = VQ(x - z₀)
        z₂ = VQ(x - z₀ - z₁)
    
    We use product quantization (parallel):
        z₀ = VQ₀(x₀)  # Brand codebook
        z₁ = VQ₁(x₁)  # Category codebook
        z₂ = VQ₂(x₂)  # Attribute codebook
        z₃ = VQ₃(x₃)  # Detail codebook
    
    This is more natural for patch-based semantic encoding where each
    patch represents a distinct semantic aspect.
    """
    
    def __init__(
        self,
        num_codebooks: int,
        embed_dim: int,
        codebook_size: int,
        do_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        sim_vq: bool = False,
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.ROTATION_TRICK,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
    ) -> None:
        super().__init__()
        
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        
        # Create independent codebooks
        self.codebooks = nn.ModuleList([
            Quantize(
                embed_dim=embed_dim,
                n_embed=codebook_size,
                do_kmeans_init=do_kmeans_init,
                codebook_normalize=i == 0 and codebook_normalize,  # Only first
                sim_vq=sim_vq,
                commitment_weight=commitment_weight,
                forward_mode=forward_mode,
                distance_mode=distance_mode,
            )
            for i in range(num_codebooks)
        ])
    
    def forward(self, patches: Tensor, temperature: float = 0.001) -> QuantizeOutput:
        """
        Args:
            patches: [B, num_codebooks, embed_dim] - Independent patches
            temperature: Temperature for Gumbel-Softmax
        
        Returns:
            QuantizeOutput with:
                embeddings: [B, num_codebooks, embed_dim] - Quantized patches
                ids: [B, num_codebooks] - Code indices
                loss: scalar - Combined quantization loss
        """
        B, M, D = patches.shape
        assert M == self.num_codebooks, f"Expected {self.num_codebooks} patches, got {M}"
        assert D == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {D}"
        
        # Quantize each patch independently
        quantized_patches = []
        ids_list = []
        total_loss = 0.0
        
        for i in range(self.num_codebooks):
            patch = patches[:, i, :]  # [B, embed_dim]
            result = self.codebooks[i](patch, temperature)
            
            quantized_patches.append(result.embeddings)  # [B, embed_dim]
            ids_list.append(result.ids)  # [B]
            total_loss += result.loss
        
        # Stack results
        embeddings = torch.stack(quantized_patches, dim=1)  # [B, M, D]
        ids = torch.stack(ids_list, dim=1)  # [B, M]
        
        # Average loss across codebooks
        loss = total_loss / self.num_codebooks
        
        return QuantizeOutput(embeddings=embeddings, ids=ids, loss=loss)
    
    def get_codebook_embeddings(self, patch_idx: int, code_ids: Tensor) -> Tensor:
        """
        Get embeddings for specific codes from a specific codebook.
        
        Args:
            patch_idx: Which codebook (0 to num_codebooks-1)
            code_ids: [B] - Code indices
        
        Returns:
            embeddings: [B, embed_dim]
        """
        return self.codebooks[patch_idx].get_item_embeddings(code_ids)
    
    def get_codebook_norms(self) -> Tensor:
        """
        Get L2 norms of all codebook embeddings.
        
        Returns:
            norms: [num_codebooks] - Average norm for each codebook
        """
        norms = []
        for codebook in self.codebooks:
            # Get embeddings from this codebook
            embeddings = codebook.embedding.weight  # [codebook_size, embed_dim]
            # Compute L2 norm for each embedding
            emb_norms = torch.norm(embeddings, p=2, dim=1)  # [codebook_size]
            # Average norm for this codebook
            avg_norm = emb_norms.mean()
            norms.append(avg_norm)
        
        return torch.stack(norms)  # [num_codebooks]
    
    def get_unique_code_usage(self) -> Tensor:
        """
        Compute percentage of unique codes being used.
        This is a dummy implementation that returns 1.0 (100% usage).
        
        To get actual usage, you'd need to track which codes are used during forward passes.
        
        Returns:
            usage: scalar tensor (percentage of codebook used)
        """
        # Dummy implementation - return 100% usage
        # In practice, you'd track this during training
        return torch.tensor(1.0, device=self.codebooks[0].device)
    
    @property
    def device(self) -> torch.device:
        """Get device from first codebook"""
        return self.codebooks[0].device
    
    def decode_codes(self, codes: Tensor) -> Tensor:
        """
        Decode discrete codes back to embeddings.
        
        Args:
            codes: [B, num_codebooks] - Discrete code indices
        
        Returns:
            embeddings: [B, num_codebooks, embed_dim] - Reconstructed embeddings
        """
        B, M = codes.shape
        assert M == self.num_codebooks, f"Expected {self.num_codebooks} codes, got {M}"
        
        embeddings = []
        for i in range(self.num_codebooks):
            code_ids = codes[:, i]  # [B]
            emb = self.codebooks[i].get_item_embeddings(code_ids)  # [B, embed_dim]
            embeddings.append(emb)
        
        return torch.stack(embeddings, dim=1)  # [B, M, embed_dim]