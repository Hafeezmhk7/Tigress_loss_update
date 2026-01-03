"""
PqVaeTokenizer: Semantic ID tokenizer for hierarchical PQ-VAE

Simplified version that creates PqVae and loads checkpoint.
The checkpoint contains the full architecture.
"""

import math
import torch

from data.processed import ItemData
from data.schemas import SeqBatch, TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange, pack
from modules.utils import eval_mode
from modules.pqvae import PqVae
from typing import Optional
from torch import nn, Tensor
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

BATCH_SIZE = 16


class PqVaeTokenizer(nn.Module):
    """
    Tokenizes item features into hierarchical semantic IDs using PQ-VAE.
    
    Semantic IDs are hierarchical: [code_0, code_1, code_2, code_3]
    """

    def __init__(
        self,
        pqvae_weights_path: Optional[str] = None,
        codebook_size: int = 256,
        num_codebooks: int = 4,
        **kwargs  # Ignore extra parameters from decoder config
    ) -> None:
        super().__init__()

        # Create PqVae with minimal parameters
        # Full architecture will be restored from checkpoint
        self.pq_vae = PqVae(
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            use_patch_encoder=True,  # Your checkpoint uses patches
        )

        if pqvae_weights_path is not None:
            self.pq_vae.load_pretrained(pqvae_weights_path)
            print(f"✅ Loaded PQ-VAE checkpoint from {pqvae_weights_path}")

        self.pq_vae.eval()

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.reset()

        # Map semantic IDs to categories (for analysis)
        self.map_to_category = {}

    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(
            axis=-1
        )

    def reset(self):
        self.cached_ids = None

    @property
    def sem_ids_dim(self):
        """Dimension of semantic IDs (num_codebooks + 1 for dedup)"""
        return self.num_codebooks + 1

    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        """
        Precompute semantic IDs for entire corpus.
        """
        cached_ids = None
        dedup_dim = []
        
        # Determine safe dataset size
        dataset_size = len(movie_dataset)
        
        # If using patch embeddings, check their actual size
        if hasattr(movie_dataset, 'patch_processor') and movie_dataset.patch_processor is not None:
            if hasattr(movie_dataset.patch_processor, '_embeddings') and movie_dataset.patch_processor._embeddings is not None:
                actual_embeddings = movie_dataset.patch_processor._embeddings.shape[0]
                if actual_embeddings < dataset_size:
                    print(f"⚠️  Warning: Dataset reports {dataset_size} items but only {actual_embeddings} patch embeddings exist")
                    print(f"   Using {actual_embeddings} items to avoid IndexError")
                    dataset_size = actual_embeddings
        
        sampler = BatchSampler(
            SequentialSampler(range(dataset_size)),
            batch_size=512,
            drop_last=False,
        )
        dataloader = DataLoader(
            movie_dataset,
            sampler=sampler,
            shuffle=False,
            collate_fn=lambda batch: batch[0],
        )
        
        from tqdm import tqdm
        print(f"Computing corpus semantic IDs for {dataset_size} items...")
        for batch in tqdm(dataloader, desc="Computing corpus IDs"):
            try:
                output = batch_to(batch, self.pq_vae.device)
                
                # Extract semantic IDs from PQ-VAE
                batch_ids = self.forward(output).sem_ids
                
                # Map semantic IDs to categories
                for idx, item in enumerate(batch_ids):
                    if str(item.tolist()) not in self.map_to_category:
                        self.map_to_category[str(item.tolist())] = output.x_brand_id[
                            idx
                        ].item()
                
                # Detect in-batch duplicates
                is_hit = self._get_hits(batch_ids, batch_ids)
                hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
                assert hits.min() >= 0
                
                if cached_ids is None:
                    cached_ids = batch_ids.clone()
                else:
                    # Detect batch-cache duplicates
                    is_hit = self._get_hits(batch_ids, cached_ids)
                    hits += is_hit.sum(axis=-1)
                    cached_ids = pack([cached_ids, batch_ids], "* d")[0]
                dedup_dim.append(hits)
                
            except Exception as e:
                print(f"⚠️  Warning: Skipping batch due to error: {e}")
                continue
        
        # Concatenate dedup column
        dedup_dim_tensor = pack(dedup_dim, "*")[0]
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]
        print(f"✅ Corpus IDs computed: {self.cached_ids.shape}")
        return self.cached_ids

    @torch.no_grad
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        """Check if semantic ID prefix exists in cached corpus IDs"""
        if self.cached_ids is None:
            raise Exception("No match can be found in empty cache.")

        prefix_length = sem_id_prefix.shape[-1]
        prefix_cache = self.cached_ids[:, :prefix_length]
        out = torch.zeros(
            *sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device
        )

        # Batch prefixes matching to avoid OOM
        batches = math.ceil(sem_id_prefix.shape[0] // BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, ...]
            matches = (
                (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3))
                .all(axis=-1)
                .any(axis=-1)
            )
            out[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, ...] = matches

        return out

    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        """Convert item IDs to semantic IDs using cached corpus IDs"""
        return rearrange(
            self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1]
        )

    @torch.no_grad
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        """
        Tokenize a batch of sequences into semantic IDs.
        """
        # Check if we need to compute or use cached IDs
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            
            # Extract hierarchical semantic IDs from PQ-VAE
            # PqVae.get_semantic_ids expects patches if use_patch_encoder=True
            sem_ids_output = self.pq_vae.get_semantic_ids(
                batch,
                text_patches=batch.text_patches if hasattr(batch, 'text_patches') else None,
                text_mask=batch.text_masks if hasattr(batch, 'text_masks') else None,
            )
            sem_ids = sem_ids_output.sem_ids
            D = sem_ids.shape[-1]
            seq_mask, sem_ids_fut = None, None
        else:
            # Use cached semantic IDs
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1

            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)

        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)
        
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut,
            x_image=batch.x_image,
        )