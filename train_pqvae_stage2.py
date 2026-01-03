"""
Stage 2 Training: Sequence-Level Contrastive Learning
=====================================================

Builds on Stage 1 by adding:
- Sequence contrastive loss (items in sequences share latent space)
- Continues patch + global reconstruction
- Enforces behavioral patterns
"""

import gin
import os
import torch
import numpy as np
import wandb
import time
from accelerate import Accelerator
from data.processed import ItemData, SeqData, RecDataset
from data.utils import batch_to, cycle, next_batch
from modules.pqvae_twostage import PqVaeTwoStage
from modules.loss import extract_sequence_pairs
from modules.utils import parse_config, display_args, display_metrics, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.logging import RichHandler
import logging

# Setup logging
logger = logging.getLogger("recsys_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = RichHandler(show_path=False)
    logger.addHandler(handler)
    logger.propagate = False


def sample_sequence_triplets(seq_dataloader, model, device, num_samples=100):
    """
    Sample (anchor, positive, negative) triplets from user sequences
    
    Returns:
        anchor_codes: [B, K] - Anchor item semantic IDs
        positive_codes: [B, K] - Next item semantic IDs (positive)
        negative_codes: [B, num_neg, K] - Random item semantic IDs (negatives)
    """
    sequences = []
    
    # Collect sequences
    for i, batch in enumerate(seq_dataloader):
        if i >= num_samples:
            break
        
        data = batch_to(batch, device)
        
        # Extract item IDs from sequence
        for seq_idx in range(data.ids.shape[0]):
            seq_mask = data.seq_mask[seq_idx]
            item_ids = data.ids[seq_idx][seq_mask].tolist()
            
            if len(item_ids) >= 2:
                sequences.append(item_ids)
    
    if not sequences:
        return None, None, None
    
    # Extract triplets (anchor, positive, negatives)
    triplets = extract_sequence_pairs(sequences, num_negatives=16)
    
    if not triplets:
        return None, None, None
    
    # Separate into anchors, positives, negatives
    anchors = torch.tensor([t[0] for t in triplets], device=device)
    positives = torch.tensor([t[1] for t in triplets], device=device)
    negatives_list = [t[2] for t in triplets]
    
    # Pad negatives to same length
    max_neg_len = max(len(negs) for negs in negatives_list)
    negatives = torch.zeros(len(triplets), max_neg_len, dtype=torch.long, device=device)
    for i, negs in enumerate(negatives_list):
        negatives[i, :len(negs)] = torch.tensor(negs, device=device)
    
    return anchors, positives, negatives


def get_codes_for_items(model, item_dataset, item_indices, device, batch_size=256):
    """
    Get semantic IDs for specific items
    
    Args:
        model: PqVaeTwoStage model
        item_dataset: ItemData dataset
        item_indices: [N] - Item indices to get codes for
        
    Returns:
        codes: [N, K] - Semantic IDs
    """
    model.eval()
    
    all_codes = []
    
    with torch.no_grad():
        for i in range(0, len(item_indices), batch_size):
            batch_indices = item_indices[i:i+batch_size]
            
            # Get item data
            batch_data = [item_dataset[idx.item()] for idx in batch_indices]
            
            # Stack into batch
            x = torch.stack([b.x for b in batch_data]).to(device)
            text_patches = torch.stack([b.text_patches for b in batch_data]).to(device)
            text_masks = torch.stack([b.text_masks for b in batch_data]).to(device)
            
            # Get semantic IDs
            from data.schemas import SeqBatch
            dummy_batch = SeqBatch(
                user_ids=torch.zeros(len(batch_indices), dtype=torch.long, device=device),
                ids=batch_indices.unsqueeze(1),
                ids_fut=torch.zeros(len(batch_indices), 1, dtype=torch.long, device=device),
                x=x,
                x_image=None,
                x_brand_id=torch.zeros(len(batch_indices), dtype=torch.long, device=device),
                x_fut=torch.zeros(len(batch_indices), 768, device=device),
                x_fut_brand_id=torch.zeros(len(batch_indices), dtype=torch.long, device=device),
                seq_mask=torch.ones(len(batch_indices), 1, dtype=torch.bool, device=device),
                text_patches=text_patches,
                text_masks=text_masks,
            )
            
            codes = model.get_semantic_ids(
                dummy_batch, text_patches, text_masks
            ).sem_ids
            
            all_codes.append(codes)
    
    return torch.cat(all_codes, dim=0)


def train_iteration(
    model, optimizer, train_dataloader, seq_dataloader, item_dataset,
    device, accelerator, iteration, t=0.2, 
    log_every=1000, wandb_logging=False,
):
    """Single training iteration for Stage 2"""
    model.train()
    model.set_stage(2)
    
    # Get item batch for reconstruction
    data = next_batch(train_dataloader, device)
    
    # Sample sequence triplets
    anchors, positives, negatives = sample_sequence_triplets(
        seq_dataloader, model, device, num_samples=10
    )
    
    # Get semantic IDs for triplets
    if anchors is not None:
        anchor_codes = get_codes_for_items(model, item_dataset, anchors, device)
        positive_codes = get_codes_for_items(model, item_dataset, positives, device)
        
        # Get codes for all negatives
        negative_codes_list = []
        for neg_batch in negatives:
            neg_codes = get_codes_for_items(model, item_dataset, neg_batch, device)
            negative_codes_list.append(neg_codes)
        negative_codes = torch.stack(negative_codes_list, dim=0)  # [B, num_neg, K]
    else:
        anchor_codes = None
        positive_codes = None
        negative_codes = None
    
    # Forward pass
    optimizer.zero_grad()
    
    with accelerator.autocast():
        output = model(
            data,
            gumbel_t=t,
            text_patches=data.text_patches,
            text_mask=data.text_masks,
            anchor_codes=anchor_codes,
            positive_codes=positive_codes,
            negative_codes=negative_codes,
        )
        loss = output.loss
    
    # Backward pass
    accelerator.backward(loss)
    optimizer.step()
    
    # Logging
    if (iteration + 1) % log_every == 0:
        metrics = {
            'train/total_loss': loss.cpu().item(),
            'train/patch_recon': output.patch_reconstruction_loss.cpu().item(),
            'train/global_recon': output.global_reconstruction_loss.cpu().item(),
            'train/pqvae_loss': output.pqvae_loss.cpu().item(),
            'train/seq_contrastive': output.sequence_contrastive_loss.cpu().item(),
            'train/diversity_loss': output.diversity_loss.cpu().item(),
        }
        
        display_metrics(metrics, title=f"Stage 2 Training (Iter {iteration+1})")
        
        if wandb_logging:
            wandb.log(metrics, step=iteration+1)
    
    return output


@gin.configurable
def train(
    # Training
    iterations=30000,
    batch_size=512,
    learning_rate=5e-4,  # Lower LR for fine-tuning
    weight_decay=0.01,
    
    # Dataset
    dataset_folder="dataset/amazon/2014",
    dataset_split="beauty",
    dataset=RecDataset.AMAZON,
    
    # Model checkpoint
    stage1_checkpoint_path=None,  # Required!
    
    # Model architecture (must match Stage 1)
    use_patch_embeddings=True,
    patch_model_name="sentence-transformers/sentence-t5-xl",
    patch_max_seq_length=77,
    num_codebooks=4,
    codebook_size=256,
    patch_token_embed_dim=192,
    
    # Loss weights
    patch_recon_weight=1.0,
    global_recon_weight=0.2,
    sequence_contrastive_weight=0.5,  # NEW: sequence-level loss
    commitment_weight=0.25,
    use_diversity_loss=True,
    diversity_weight=0.01,
    
    # Sequence sampling
    sequence_sample_size=10,
    num_negatives=16,
    contrastive_temperature=0.1,
    
    # Logging
    log_dir="logdir/pqvae/stage2",
    save_model_every=10000,
    eval_every=5000,
    log_every=1000,
    wandb_logging=True,
    run_prefix="Stage2-SeqContrastive",
    
    # Advanced
    gumbel_temperature=0.2,
    mixed_precision_type="fp16",
    **kwargs
):
    """
    Stage 2 Training: Sequence-Level Contrastive Learning
    """
    
    # Validation
    if stage1_checkpoint_path is None:
        raise ValueError("stage1_checkpoint_path is required for Stage 2 training!")
    
    if not os.path.exists(stage1_checkpoint_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint_path}")
    
    # Setup
    uid = str(int(time.time()))
    logger.info(f"\n{'='*70}")
    logger.info(f"=== Stage 2: Sequence-Level Training ===")
    logger.info(f"{'='*70}")
    logger.info(f"Session UID: {uid}")
    logger.info(f"Dataset: {dataset_folder} | Split: {dataset_split}")
    logger.info(f"Loading from Stage 1: {stage1_checkpoint_path}")
    logger.info(f"{'='*70}\n")
    
    log_dir = os.path.join(os.path.expanduser("~"), log_dir, dataset_split, uid)
    os.makedirs(log_dir, exist_ok=True)
    
    # Accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision_type)
    device = accelerator.device
    
    # Display config
    display_args(locals(), title="Stage 2 Training Configuration")
    
    # WandB
    if wandb_logging and accelerator.is_main_process:
        run_name = f"{run_prefix}-{dataset_split}/{uid}"
        wandb.init(
            entity="RecSys-UvA",
            project="PQ-VAE-TwoStage",
            name=run_name,
            config=locals(),
        )
    
    # Load datasets
    logger.info("Loading datasets...")
    
    # Item dataset (for getting codes)
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        train_test_split="all",
        split=dataset_split,
        force_process=False,
        use_patch_embeddings=use_patch_embeddings,
        patch_model_name=patch_model_name,
        patch_max_seq_length=patch_max_seq_length,
        device=device,
    )
    
    # Sequence dataset (for contrastive learning)
    seq_train_data = SeqData(
        root=dataset_folder,
        dataset=dataset,
        data_split="train",
        split=dataset_split,
        subsample=False,
        device=device,
    )
    
    train_dataloader = DataLoader(
        item_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    train_dataloader = cycle(train_dataloader)
    
    seq_dataloader = DataLoader(
        seq_train_data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    seq_dataloader = cycle(seq_dataloader)
    
    train_dataloader, seq_dataloader = accelerator.prepare(
        train_dataloader, seq_dataloader
    )
    
    logger.info(f"✓ Item dataset: {len(item_dataset)} items")
    logger.info(f"✓ Sequence dataset: {len(seq_train_data)} sequences")
    
    # Model
    logger.info("Initializing model...")
    
    model = PqVaeTwoStage(
        # Reconstruction targets
        patch_dim=1024,
        num_patches=patch_max_seq_length,
        global_dim=768,
        
        # Encoder
        patch_token_embed_dim=patch_token_embed_dim,
        patch_hidden_dim=512,
        patch_num_heads=8,
        patch_num_layers=2,
        patch_dropout=0.1,
        
        # Quantization
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        codebook_kmeans_init=True,
        commitment_weight=commitment_weight,
        
        # Diversity
        use_diversity_loss=use_diversity_loss,
        diversity_weight=diversity_weight,
        
        # Decoders
        decoder_hidden_dim=512,
        decoder_num_layers=2,
        decoder_num_heads=8,
        decoder_dropout=0.1,
        
        # Loss weights
        patch_recon_weight=patch_recon_weight,
        global_recon_weight=global_recon_weight,
        sequence_contrastive_weight=sequence_contrastive_weight,
        
        # Sequence contrastive
        contrastive_temperature=contrastive_temperature,
        num_negatives=num_negatives,
    )
    
    # Load Stage 1 checkpoint
    logger.info(f"Loading Stage 1 checkpoint: {stage1_checkpoint_path}")
    model.load_pretrained(stage1_checkpoint_path)
    logger.info("✓ Stage 1 weights loaded")
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    model, optimizer = accelerator.prepare(model, optimizer)
    
    logger.info(f"✓ Model initialized")
    logger.info(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting Stage 2 Training: {iterations} iterations")
    logger.info(f"{'='*70}\n")
    
    with tqdm(total=iterations, desc="Stage 2") as pbar:
        for iter_ in range(iterations):
            # Train
            output = train_iteration(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                seq_dataloader=seq_dataloader,
                item_dataset=item_dataset,
                device=device,
                accelerator=accelerator,
                iteration=iter_,
                t=gumbel_temperature,
                log_every=log_every,
                wandb_logging=wandb_logging,
            )
            
            # Save checkpoint
            if (iter_ + 1) % save_model_every == 0 or (iter_ + 1) == iterations:
                if accelerator.is_main_process:
                    checkpoint_path = f"{log_dir}/checkpoint_{iter_+1}.pt"
                    torch.save({
                        'iteration': iter_ + 1,
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, checkpoint_path)
                    logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{output.loss.item():.4f}",
                'seq_cont': f"{output.sequence_contrastive_loss.item():.4f}",
            })
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Stage 2 Training Complete!")
    logger.info(f"Final checkpoint: {log_dir}/checkpoint_{iterations}.pt")
    logger.info(f"{'='*70}\n")
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    parse_config()
    train()