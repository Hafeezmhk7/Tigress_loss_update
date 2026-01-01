from typing import NamedTuple, Optional
from torch import Tensor

FUT_SUFFIX = "_fut"


class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut_brand_id: Tensor
    x_fut: Tensor
    x_brand_id: Tensor
    seq_mask: Tensor
    x_image: Optional[Tensor] = None
    # NEW: Patch embeddings for PQ-VAE
    text_patches: Optional[Tensor] = None  # [B, N_tokens, hidden_dim]
    text_masks: Optional[Tensor] = None     # [B, N_tokens]


class TokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor
    x_image: Optional[Tensor] = None