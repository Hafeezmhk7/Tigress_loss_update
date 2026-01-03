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
    """
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, q: Tensor, k: Tensor) -> Tensor:
        batch_size = q.shape[0]
        
        # Normalize inputs
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
        # Create two augmented views via dropout
        view1 = self.dropout(encoder_output)
        view2 = self.dropout(encoder_output)
        
        loss = self.infonce(view1, view2)
        return loss


class MultiScaleInfoNCELoss(nn.Module):
    """
    Hierarchical Multi-scale InfoNCE Loss.
    
    Normalizes codebook vectors and residuals before computing contrastive loss.
    """
    def __init__(
        self, 
        n_levels: int, 
        temperature: float = 0.07, 
        level_weights: List[float] = None,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        
        # Create InfoNCE loss for each level
        self.infonce_losses = nn.ModuleList([
            InfoNCELoss(temperature) for _ in range(n_levels)
        ])
        
        # Use raw weights (NO NORMALIZATION)
        if level_weights is None:
            self.level_weights = [1.0] * n_levels
        else:
            assert len(level_weights) == n_levels, \
                f"level_weights must have length {n_levels}, got {len(level_weights)}"
            self.level_weights = level_weights
        
        # Store for logging
        self.per_level_losses = {}
        
        # Log configuration
        import logging
        logger = logging.getLogger("recsys_logger")
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"Hierarchical Multi-Scale InfoNCE Configuration:")
        logger.info(f"{'='*70}")
        logger.info(f"  Levels: {n_levels}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Level Weights (RAW): {self.level_weights}")
        logger.info(f"{'='*70}")
        logger.info(f"")
    
    def forward(
        self, 
        quantized_list: List[Tensor], 
        residual_list: List[Tensor]
    ) -> Tensor:
        assert len(quantized_list) == self.n_levels
        assert len(residual_list) == self.n_levels
        
        total_loss = 0
        self.per_level_losses = {}
        
        for level, (q, k, weight) in enumerate(
            zip(quantized_list, residual_list, self.level_weights)
        ):
            # Compute InfoNCE at this level (normalization happens inside InfoNCELoss)
            level_loss = self.infonce_losses[level](q, k)
            
            # Apply hierarchical weight (RAW, not normalized)
            weighted_loss = weight * level_loss
            total_loss += weighted_loss
            
            # Store for logging
            self.per_level_losses[f'ms_loss_L{level}'] = level_loss.item()
            self.per_level_losses[f'ms_weight_L{level}'] = weight
            self.per_level_losses[f'ms_weighted_L{level}'] = weighted_loss.item()
        
        # Return total (NOT averaged by n_levels)
        self.per_level_losses['ms_total'] = total_loss.item()
        
        return total_loss


class CoSTInfoNCELoss(nn.Module):
    """
    CoST (Contrastive Semantic Tokenization) reconstruction-level InfoNCE.
    """
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.infonce = InfoNCELoss(temperature=temperature)
    
    def forward(
        self, 
        reconstructed: Tensor, 
        original: Tensor
    ) -> Tensor:
        loss = self.infonce(reconstructed, original)
        return loss


class SequenceContrastiveLoss(nn.Module):
    """
    Sequence-Aware Contrastive Loss (Stage 2)
    
    Enforces that items appearing close in user sequences have similar semantic IDs.
    
    Uses InfoNCE with:
    - Anchor: item i in sequence
    - Positive: item i+1 (next item in same sequence)
    - Negatives: random items from other sequences
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        num_negatives: int = 16,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
    
    def forward(
        self,
        anchor: Tensor,      # [B, D] - Current item encoding
        positive: Tensor,    # [B, D] - Next item in sequence
        negatives: Tensor,   # [B, num_neg, D] - Random items
    ) -> Tensor:
        """
        InfoNCE loss for sequence pairs.
        
        Args:
            anchor: [B, D] - Embeddings of anchor items
            positive: [B, D] - Embeddings of positive items (i+1)
            negatives: [B, num_neg, D] - Embeddings of negative items
            
        Returns:
            loss: scalar
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # Positive similarity: [B]
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        
        # Negative similarities: [B, num_neg]
        neg_sim = torch.bmm(
            anchor.unsqueeze(1),           # [B, 1, D]
            negatives.transpose(1, 2)      # [B, D, num_neg]
        ).squeeze(1) / self.temperature    # [B, num_neg]
        
        # Logits: [B, 1 + num_neg]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # Labels: positive is always first (index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


def extract_sequence_pairs(
    sequences: List[List[int]],
    num_negatives: int = 16,
) -> List[Tuple[int, int, List[int]]]:
    """
    Extract (anchor, positive, negatives) triplets from user sequences.
    
    Args:
        sequences: List of user sequences [[item1, item2, ...], ...]
        num_negatives: Number of negative samples
        
    Returns:
        triplets: List of (anchor_idx, positive_idx, [neg_idx1, neg_idx2, ...])
    """
    import random
    
    # Collect all items for negative sampling
    all_items = []
    for seq in sequences:
        all_items.extend(seq)
    all_items = list(set(all_items))
    
    triplets = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        
        for i in range(len(seq) - 1):
            anchor = seq[i]
            positive = seq[i + 1]
            
            # Sample negatives (avoid anchor and positive)
            negatives = random.sample(
                [item for item in all_items if item != anchor and item != positive],
                min(num_negatives, len(all_items) - 2)
            )
            
            triplets.append((anchor, positive, negatives))
    
    return triplets