"""
Script to precompute and save token embeddings from pretrained models (CLIP/SentenceTransformer).

Usage:
    python scripts/precompute_patch_embeddings.py \
        --dataset_folder dataset/amazon/2014 \
        --dataset_split beauty \
        --model_name sentence-transformers/sentence-t5-xl \
        --output_dir data/processed
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_items_metadata(dataset_folder: str, dataset_split: str):
    """
    Load item metadata (titles) from the dataset.
    
    Returns:
        List of item titles/descriptions
    """
    import pandas as pd
    
    # Construct path to metadata file
    metadata_path = os.path.join(dataset_folder, f"{dataset_split}.csv")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    logger.info(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Assuming the metadata has 'title' column
    # Adjust this based on your actual data schema
    if 'title' in df.columns:
        texts = df['title'].fillna('').tolist()
    elif 'text' in df.columns:
        texts = df['text'].fillna('').tolist()
    else:
        # Fallback: concatenate available text columns
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        texts = df[text_cols].fillna('').agg(' '.join, axis=1).tolist()
    
    logger.info(f"Loaded {len(texts)} items")
    return texts


def extract_token_embeddings(
    texts: list,
    model_name: str = "sentence-transformers/sentence-t5-xl",
    batch_size: int = 32,
    max_seq_length: int = 77,
    device: str = "cuda",
):
    """
    Extract token embeddings from pretrained model.
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model identifier
        batch_size: Batch size for processing
        max_seq_length: Maximum sequence length
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        token_embeddings: np.array of shape [N, max_seq_length, hidden_dim]
        attention_masks: np.array of shape [N, max_seq_length]
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    
    # Get hidden dimension
    hidden_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model hidden dimension: {hidden_dim}")
    
    # Initialize arrays
    n_items = len(texts)
    token_embeddings_list = []
    attention_masks_list = []
    
    # Process in batches
    logger.info(f"Extracting token embeddings for {n_items} items...")
    for i in tqdm(range(0, n_items, batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encoded = model.tokenize(batch_texts)
        
        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get token embeddings (not pooled)
        with torch.no_grad():
            # Forward through transformer to get hidden states
            outputs = model[0].auto_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Get last hidden state: [batch_size, seq_len, hidden_dim]
            token_embeds = outputs.last_hidden_state
        
        # Pad or truncate to max_seq_length
        curr_seq_len = token_embeds.shape[1]
        if curr_seq_len < max_seq_length:
            # Pad
            padding = torch.zeros(
                token_embeds.shape[0],
                max_seq_length - curr_seq_len,
                token_embeds.shape[2],
                device=device
            )
            token_embeds = torch.cat([token_embeds, padding], dim=1)
            
            # Pad mask
            mask_padding = torch.zeros(
                attention_mask.shape[0],
                max_seq_length - curr_seq_len,
                device=device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
        elif curr_seq_len > max_seq_length:
            # Truncate
            token_embeds = token_embeds[:, :max_seq_length, :]
            attention_mask = attention_mask[:, :max_seq_length]
        
        token_embeddings_list.append(token_embeds.cpu().numpy())
        attention_masks_list.append(attention_mask.cpu().numpy())
    
    # Concatenate all batches
    token_embeddings = np.concatenate(token_embeddings_list, axis=0)
    attention_masks = np.concatenate(attention_masks_list, axis=0)
    
    logger.info(f"Token embeddings shape: {token_embeddings.shape}")
    logger.info(f"Attention masks shape: {attention_masks.shape}")
    
    return token_embeddings, attention_masks


def main():
    parser = argparse.ArgumentParser(
        description="Precompute token embeddings for patch-based semantic IDs"
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to dataset folder (e.g., dataset/amazon/2014)"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        required=True,
        help="Dataset split name (e.g., beauty, sports, toys)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/sentence-t5-xl",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for saving embeddings"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=77,
        help="Maximum sequence length (CLIP default is 77)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load texts
    texts = load_items_metadata(args.dataset_folder, args.dataset_split)
    
    # Extract embeddings
    token_embeddings, attention_masks = extract_token_embeddings(
        texts=texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )
    
    # Save to disk
    output_embeddings_path = os.path.join(
        args.output_dir,
        f"{args.dataset_split}_text_patches.npy"
    )
    output_masks_path = os.path.join(
        args.output_dir,
        f"{args.dataset_split}_text_masks.npy"
    )
    
    logger.info(f"Saving token embeddings to {output_embeddings_path}")
    np.save(output_embeddings_path, token_embeddings)
    
    logger.info(f"Saving attention masks to {output_masks_path}")
    np.save(output_masks_path, attention_masks)
    
    logger.info("Done!")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset_folder}/{args.dataset_split}")
    print(f"Model: {args.model_name}")
    print(f"Number of items: {len(texts)}")
    print(f"Token embeddings shape: {token_embeddings.shape}")
    print(f"Attention masks shape: {attention_masks.shape}")
    print(f"Output files:")
    print(f"  - {output_embeddings_path}")
    print(f"  - {output_masks_path}")
    print("="*50)


if __name__ == "__main__":
    main()