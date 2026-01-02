#!/usr/bin/env python
"""
Regenerate patch embeddings for ALL items (train + eval = 12101 items)
"""

import torch
from data.processed import ItemData, RecDataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_patch_embeddings():
    """Generate patch embeddings for the complete dataset (train + eval)"""
    
    logger.info("="*70)
    logger.info("Generating Patch Embeddings for Beauty Dataset")
    logger.info("="*70)
    logger.info("")
    
    # Delete old patch embeddings to force regeneration
    import os
    patch_file = 'dataset/amazon/2014/processed/beauty_text_patches.npy'
    mask_file = 'dataset/amazon/2014/processed/beauty_text_masks.npy'
    
    if os.path.exists(patch_file):
        logger.info(f"Removing old patch embeddings: {patch_file}")
        os.remove(patch_file)
    
    if os.path.exists(mask_file):
        logger.info(f"Removing old attention masks: {mask_file}")
        os.remove(mask_file)
    
    logger.info("")
    
    # Generate patch embeddings for ALL split (train + eval)
    logger.info("Generating patch embeddings for 'all' split (train + eval)...")
    logger.info("This will take a few minutes...")
    logger.info("")
    
    dataset = ItemData(
        root='dataset/amazon/2014',
        dataset=RecDataset.AMAZON,
        train_test_split='all',  # ← Generate for ALL items (12101)
        split='beauty',
        use_patch_embeddings=True,
        patch_model_name='sentence-transformers/sentence-t5-xl',
        patch_max_seq_length=77,
        force_process=True,  # ← Force regeneration even if files exist
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    logger.info("")
    logger.info("="*70)
    logger.info("✅ GENERATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Dataset items: {len(dataset)}")
    logger.info(f"Patch embeddings shape: {dataset.patch_processor._embeddings.shape}")
    logger.info(f"Attention masks shape: {dataset.patch_processor._attention_masks.shape}")
    logger.info("")
    logger.info(f"Files created:")
    logger.info(f"  - {patch_file}")
    logger.info(f"  - {mask_file}")
    logger.info("")
    logger.info("You can now run training successfully!")
    logger.info("="*70)

if __name__ == "__main__":
    generate_patch_embeddings()