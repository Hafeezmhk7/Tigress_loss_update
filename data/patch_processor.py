"""
Patch Embedding Processor
--------------------------
This module provides utilities to extract and cache token-level embeddings
for patch-based semantic encoding. Integrates with TIGRESS data processing pipeline.

Usage:
    1. Add PatchEmbeddingProcessor to your data/processed.py
    2. Call process_patch_embeddings() once to precompute and cache embeddings
    3. Use get_patch_embeddings() in your dataset to retrieve embeddings
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
import logging

logger = logging.getLogger("recsys_logger")


class PatchEmbeddingProcessor:
    """
    Handles extraction and caching of token-level embeddings for patch encoding.
    
    Integrates with existing TIGRESS data processing by:
    - Following same caching pattern as other processed data
    - Using same directory structure (processed_dir/)
    - Lazy loading to save memory
    """
    
    def __init__(
        self,
        processed_dir: str,
        dataset_split: str,
        model_name: str = "sentence-transformers/sentence-t5-xl",
        max_seq_length: int = 77,
        batch_size: int = 32,
    ):
        """
        Args:
            processed_dir: Directory for processed data (e.g., "dataset/amazon/2014/processed")
            dataset_split: Dataset split name (e.g., "beauty", "sports")
            model_name: HuggingFace model for text embeddings
            max_seq_length: Maximum sequence length for tokenization
            batch_size: Batch size for embedding extraction
        """
        self.processed_dir = Path(processed_dir)
        self.dataset_split = dataset_split
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        
        # Cache file paths
        self.embeddings_path = self.processed_dir / f"{dataset_split}_text_patches.npy"
        self.masks_path = self.processed_dir / f"{dataset_split}_text_masks.npy"
        self.metadata_path = self.processed_dir / f"{dataset_split}_patch_metadata.pt"
        
        # Loaded embeddings (lazy loading)
        self._embeddings = None
        self._masks = None
    
    @property
    def is_processed(self) -> bool:
        """Check if patch embeddings are already processed."""
        return (
            self.embeddings_path.exists() and
            self.masks_path.exists() and
            self.metadata_path.exists()
        )
    
    def process(
        self,
        item_titles: list,
        force_reprocess: bool = False,
    ) -> None:
        """
        Extract and cache token-level embeddings for all items.
        
        Args:
            item_titles: List of item titles/descriptions
            force_reprocess: Force reprocessing even if cache exists
        """
        if self.is_processed and not force_reprocess:
            logger.info(f"Patch embeddings already processed for {self.dataset_split}")
            return
        
        logger.info(f"Processing patch embeddings for {self.dataset_split}...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Items: {len(item_titles)}")
        
        # Load model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        model.eval()
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Extract embeddings
        all_embeddings = []
        all_masks = []
        
        for i in tqdm(range(0, len(item_titles), self.batch_size), desc="Extracting embeddings"):
            batch_titles = item_titles[i:i + self.batch_size]
            
            # Tokenize
            features = model.tokenize(batch_titles)
            
            # Get token embeddings (not pooled sentence embeddings)
            with torch.no_grad():
                # Move to device
                input_ids = features['input_ids'].to(device)
                attention_mask = features['attention_mask'].to(device)
                
                # Get embeddings from transformer
                outputs = model[0].auto_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                
                # Get last hidden state [batch, seq_len, hidden_dim]
                token_embeddings = outputs.last_hidden_state.cpu()
                
                # Pad/truncate to max_seq_length
                seq_len = token_embeddings.shape[1]
                if seq_len < self.max_seq_length:
                    # Pad
                    padding = torch.zeros(
                        token_embeddings.shape[0],
                        self.max_seq_length - seq_len,
                        token_embeddings.shape[2]
                    )
                    token_embeddings = torch.cat([token_embeddings, padding], dim=1)
                    
                    # Pad mask
                    mask_padding = torch.zeros(
                        attention_mask.shape[0],
                        self.max_seq_length - seq_len,
                        dtype=torch.bool
                    )
                    attention_mask = torch.cat([attention_mask.cpu(), mask_padding], dim=1)
                else:
                    # Truncate
                    token_embeddings = token_embeddings[:, :self.max_seq_length, :]
                    attention_mask = attention_mask[:, :self.max_seq_length]
                
                all_embeddings.append(token_embeddings)
                all_masks.append(attention_mask.cpu())
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, max_seq_len, hidden_dim]
        all_masks = torch.cat(all_masks, dim=0)  # [N, max_seq_len]
        
        logger.info(f"Extracted embeddings: {all_embeddings.shape}")
        logger.info(f"Extracted masks: {all_masks.shape}")
        
        # Save to disk
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to {self.embeddings_path}")
        np.save(self.embeddings_path, all_embeddings.numpy())
        
        logger.info(f"Saving to {self.masks_path}")
        np.save(self.masks_path, all_masks.numpy())
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'max_seq_length': self.max_seq_length,
            'num_items': len(item_titles),
            'embedding_dim': all_embeddings.shape[-1],
        }
        torch.save(metadata, self.metadata_path)
        logger.info(f"Saved metadata to {self.metadata_path}")
        
        logger.info("✅ Patch embedding processing complete!")
    
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load cached patch embeddings and masks.
        
        Returns:
            embeddings: [N, max_seq_len, hidden_dim]
            masks: [N, max_seq_len]
        """
        if self._embeddings is None:
            if not self.is_processed:
                raise FileNotFoundError(
                    f"Patch embeddings not found. Run process() first.\n"
                    f"Expected: {self.embeddings_path}"
                )
            
            logger.info(f"Loading patch embeddings from {self.embeddings_path}")
            self._embeddings = torch.from_numpy(np.load(self.embeddings_path)).float()
            
            logger.info(f"Loading attention masks from {self.masks_path}")
            self._masks = torch.from_numpy(np.load(self.masks_path)).bool()
            
            logger.info(f"Loaded: {self._embeddings.shape}, {self._masks.shape}")
        
        return self._embeddings, self._masks
    
    def get_patch_embeddings(
        self,
        indices: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get patch embeddings for specific indices.
        
        Args:
            indices: Item indices [batch_size] or [batch_size, 1]
            device: Device to move tensors to
            
        Returns:
            embeddings: [batch_size, max_seq_len, hidden_dim]
            masks: [batch_size, max_seq_len]
        """
        # Load if not already loaded
        if self._embeddings is None:
            self.load()
        
        # Handle different index shapes
        if indices.dim() > 1:
            indices = indices.flatten()
        
        # Index embeddings
        embeddings = self._embeddings[indices]
        masks = self._masks[indices]
        
        # Move to device if specified
        if device is not None:
            embeddings = embeddings.to(device)
            masks = masks.to(device)
        
        return embeddings, masks


def process_patch_embeddings_for_dataset(
    dataset_folder: str,
    dataset_split: str,
    model_name: str = "sentence-transformers/sentence-t5-xl",
    max_seq_length: int = 77,
    batch_size: int = 32,
    force_reprocess: bool = False,
) -> None:
    """
    Standalone function to process patch embeddings for a dataset.
    
    This can be called independently or integrated into your data processing pipeline.
    
    Args:
        dataset_folder: Path to dataset (e.g., "dataset/amazon/2014")
        dataset_split: Split name (e.g., "beauty", "sports")
        model_name: HuggingFace model for embeddings
        max_seq_length: Maximum sequence length
        batch_size: Batch size for processing
        force_reprocess: Force reprocessing even if cache exists
    """
    import pandas as pd
    
    # Determine paths
    processed_dir = os.path.join(dataset_folder, "processed")
    metadata_path = os.path.join(dataset_folder, f"meta_{dataset_split}.json.gz")
    
    # Load metadata to get item titles
    logger.info(f"Loading metadata from {metadata_path}")
    meta_df = pd.read_json(metadata_path, lines=True)
    
    # Get titles (combine title and description if available)
    if 'title' in meta_df.columns and 'description' in meta_df.columns:
        item_titles = (
            meta_df['title'].fillna('') + ' ' + meta_df['description'].fillna('')
        ).str.strip().tolist()
    elif 'title' in meta_df.columns:
        item_titles = meta_df['title'].fillna('').tolist()
    else:
        raise ValueError("Metadata must contain 'title' column")
    
    # Create processor
    processor = PatchEmbeddingProcessor(
        processed_dir=processed_dir,
        dataset_split=dataset_split,
        model_name=model_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )
    
    # Process
    processor.process(item_titles, force_reprocess=force_reprocess)
    
    logger.info(f"✅ Patch embeddings saved to {processed_dir}")


if __name__ == "__main__":
    """
    Standalone script to preprocess patch embeddings.
    
    Usage:
        python -m data.patch_processor dataset/amazon/2014 beauty
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Process patch embeddings for TIGRESS")
    parser.add_argument("dataset_folder", type=str, help="Path to dataset folder")
    parser.add_argument("dataset_split", type=str, help="Dataset split (e.g., beauty)")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/sentence-t5-xl")
    parser.add_argument("--max_seq_length", type=int, default=77)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="Force reprocessing")
    
    args = parser.parse_args()
    
    process_patch_embeddings_for_dataset(
        dataset_folder=args.dataset_folder,
        dataset_split=args.dataset_split,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        force_reprocess=args.force,
    )