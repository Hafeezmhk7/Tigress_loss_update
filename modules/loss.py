from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F
from typing import List


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        return ((x_hat - x) ** 2).sum(axis=-1)


class CategoricalReconstuctionLoss(nn.Module):
    def __init__(self, n_cat_feats: int) -> None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        reconstr = self.reconstruction_loss(
            x_hat[:, : -self.n_cat_feats], x[:, : -self.n_cat_feats]
        )
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats :],
                x[:, -self.n_cat_feats :],
                reduction="none",
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        emb_loss = ((query.detach() - value) ** 2).sum(axis=[-1])
        query_loss = ((query - value.detach()) ** 2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss


# ===== TRIPLE-STAGE CONTRASTIVE LEARNING =====

class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE contrastive loss using cross-entropy formulation.
    
    Reference: van den Oord et al. (2018). Representation Learning with 
               Contrastive Predictive Coding.
    """
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, q: Tensor, k: Tensor) -> Tensor:
        """
        Args:
            q: [N, d] query embeddings
            k: [N, d] key embeddings
        Returns:
            InfoNCE loss scalar
        """
        batch_size = q.shape[0]
        
        # Normalize embeddings to unit sphere
        q = F.normalize(q, dim=1, eps=1e-8)
        k = F.normalize(k, dim=1, eps=1e-8)
        
        # Compute similarity matrix [batch_size, batch_size]
        logits = torch.mm(q, k.t()) / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=q.device)
        
        # Symmetric contrastive loss (q->k and k->q)
        loss = (F.cross_entropy(logits, labels) + 
                F.cross_entropy(logits.t(), labels)) / 2.0
        
        return loss


class EncoderInfoNCELoss(nn.Module):
    """
    Encoder-level InfoNCE to prevent latent collapse at Level 0.
    
    Uses dropout augmentation to create two views of the same item.
    Ensures encoder outputs are well-distributed before quantization.
    
    Novel Contribution: First explicit treatment of Level-0 collapse
    in semantic ID generation for recommender systems.
    """
    def __init__(
        self, 
        temperature: float = 0.07,
        dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout_rate)
        self.infonce = InfoNCELoss(temperature=temperature)
    
    def forward(self, encoder_output: Tensor) -> Tensor:
        """
        Args:
            encoder_output: [N, d] encoder embeddings BEFORE quantization
        Returns:
            Encoder InfoNCE loss scalar
        """
        # Create two augmented views via dropout
        # View 1: Apply dropout
        view1 = self.dropout(encoder_output)
        
        # View 2: Apply dropout again (different mask)
        view2 = self.dropout(encoder_output)
        
        # Contrastive loss between two views
        # Enforces: same item should have similar encoding despite dropout
        loss = self.infonce(view1, view2)
        
        return loss


class MultiScaleInfoNCELoss(nn.Module):
    """
    Multi-scale residual InfoNCE for per-level distinctiveness.
    
    Applies InfoNCE loss at each residual quantization level independently.
    
    Novel Contribution: First per-level InfoNCE supervision in 
    hierarchical RQ-VAE for semantic IDs.
    """
    def __init__(
        self, 
        n_levels: int, 
        temperature: float = 0.07, 
        level_weights: List[float] = None
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.infonce_losses = nn.ModuleList([
            InfoNCELoss(temperature) for _ in range(n_levels)
        ])
        # Allow different weights per level (e.g., emphasize higher levels)
        self.level_weights = level_weights if level_weights else [1.0] * n_levels
    
    def forward(
        self, 
        quantized_list: List[Tensor], 
        residual_list: List[Tensor]
    ) -> Tensor:
        """
        Args:
            quantized_list: List of [N, d] quantized embeddings per level
            residual_list: List of [N, d] residuals BEFORE quantization per level
        Returns:
            Weighted sum of InfoNCE losses across all levels
        """
        total_loss = 0
        for level, (q, k, weight) in enumerate(
            zip(quantized_list, residual_list, self.level_weights)
        ):
            level_loss = self.infonce_losses[level](q, k)
            total_loss += weight * level_loss
        
        return total_loss


class CoSTInfoNCELoss(nn.Module):
    """
    CoST (Contrastive Semantic Tokenization) reconstruction-level InfoNCE.
    
    Ensures reconstructed embeddings preserve semantic relationships
    from original embeddings. Critical for downstream LLM performance.
    
    Reference: Zhu et al. (2024). CoST: Contrastive Quantization based 
               Semantic Tokenization for Generative Recommendation.
    
    Improvement over baseline: 43% on MIND dataset (RecSys 2024).
    """
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.infonce = InfoNCELoss(temperature=temperature)
    
    def forward(
        self, 
        reconstructed: Tensor, 
        original: Tensor
    ) -> Tensor:
        """
        Args:
            reconstructed: [N, d] decoder output (sum of quantized embeddings)
            original: [N, d] original encoder output BEFORE quantization
        Returns:
            CoST InfoNCE loss scalar
        """
        # Contrastive loss between reconstructed and original
        # Enforces: reconstructed should match original semantics
        loss = self.infonce(reconstructed, original)
        
        return loss