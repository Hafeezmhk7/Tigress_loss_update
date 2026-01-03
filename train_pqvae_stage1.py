"""
Stage 1 Training: Item-Level Patch Reconstruction
=================================================

Trains PQ-VAE with:
- Patch reconstruction (main task)
- Global reconstruction (auxiliary task)  
- Codebook quantization loss
- Diversity loss
"""

import gin
import os
import torch
import numpy as np
import wandb
import time
from accelerate import Accelerator
from data.processed import ItemData, RecDataset
from data.utils import batch_to, cycle, next_batch
from modules.pqvae import PqVae
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


def train_iteration(
    model, optimizer, train_dataloader, device, accelerator, 
    iteration, t=0.2, log_every=1000, wandb_logging=False
):
    """Single training iteration"""
    model.train()
    
    # Get batch
    data = next_batch(train_dataloader, device)
    
    # Forward pass
    optimizer.zero_grad()
    
    with accelerator.autocast():
        output = model(
            data,
            gumbel_t=t,
            text_patches=data.text_patches,
            text_mask=data.text_masks,
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
            'train/diversity_loss': output.diversity_loss.cpu().item(),
            'train/embs_norm': output.embs_norm.mean().cpu().item(),
            'train/p_unique_ids': output.p_unique_ids.cpu().item(),
        }
        
        display_metrics(metrics, title=f"Training Metrics (Iter {iteration+1})")
        
        if wandb_logging:
            wandb.log(metrics, step=iteration+1)
    
    return output


def evaluate(model, eval_dataloader, device, t=0.2):
    """Evaluate model"""
    model.eval()
    
    losses = []
    patch_losses = []
    global_losses = []
    pqvae_losses = []
    diversity_losses = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            data = batch_to(batch, device)
            
            output = model(
                data,
                gumbel_t=t,
                text_patches=data.text_patches,
                text_mask=data.text_masks,
            )
            
            losses.append(output.loss.cpu().item())
            patch_losses.append(output.patch_reconstruction_loss.cpu().item())
            global_losses.append(output.global_reconstruction_loss.cpu().item())
            pqvae_losses.append(output.pqvae_loss.cpu().item())
            diversity_losses.append(output.diversity_loss.cpu().item())
    
    return {
        'eval/total_loss': np.mean(losses),
        'eval/patch_recon': np.mean(patch_losses),
        'eval/global_recon': np.mean(global_losses),
        'eval/pqvae_loss': np.mean(pqvae_losses),
        'eval/diversity_loss': np.mean(diversity_losses),
    }


@gin.configurable
def train(
    # Training
    iterations=50000,
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=0.01,
    
    # Dataset
    dataset_folder="dataset/amazon/2014",
    dataset_split="beauty",
    dataset=RecDataset.AMAZON,
    force_dataset_process=False,
    
    # Model architecture
    use_patch_embeddings=True,
    patch_model_name="sentence-transformers/sentence-t5-xl",
    patch_max_seq_length=77,
    num_codebooks=4,
    codebook_size=256,
    patch_token_embed_dim=192,
    
    # Loss weights
    patch_recon_weight=1.0,
    global_recon_weight=0.2,
    commitment_weight=0.25,
    use_diversity_loss=True,
    diversity_weight=0.01,
    
    # Logging
    log_dir="logdir/pqvae/stage1",
    save_model_every=10000,
    eval_every=5000,
    log_every=1000,
    wandb_logging=True,
    run_prefix="Stage1-PatchRecon",
    
    # Advanced
    gumbel_temperature=0.2,
    mixed_precision_type="fp16",
    **kwargs
):
    """
    Stage 1 Training: Item-Level Reconstruction
    """
    
    # Setup
    uid = str(int(time.time()))
    logger.info(f"=== Stage 1: Item-Level Training ===")
    logger.info(f"Session UID: {uid}")
    logger.info(f"Dataset: {dataset_folder} | Split: {dataset_split}")
    
    log_dir = os.path.join(os.path.expanduser("~"), log_dir, dataset_split, uid)
    os.makedirs(log_dir, exist_ok=True)
    
    # Accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision_type)
    device = accelerator.device
    
    # Display config
    display_args(locals(), title="Stage 1 Training Configuration")
    
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
    
    train_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        train_test_split="train",
        split=dataset_split,
        force_process=force_dataset_process,
        use_patch_embeddings=use_patch_embeddings,
        patch_model_name=patch_model_name,
        patch_max_seq_length=patch_max_seq_length,
        device=device,
    )
    
    eval_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        train_test_split="eval",
        split=dataset_split,
        force_process=False,  # Already processed
        use_patch_embeddings=use_patch_embeddings,
        patch_model_name=patch_model_name,
        patch_max_seq_length=patch_max_seq_length,
        device=device,
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    train_dataloader = cycle(train_dataloader)
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )
    
    logger.info(f"✓ Train items: {len(train_dataset)}")
    logger.info(f"✓ Eval items: {len(eval_dataset)}")
    
    # Model
    logger.info("Initializing model...")
    
    model = PqVae(
        # Reconstruction targets
        patch_dim=1024,  # sentence-t5-xl hidden dim
        num_patches=patch_max_seq_length,
        global_dim=768,  # Target global embedding dim
        
        # Encoder
        use_patch_encoder=True,
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
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    model, optimizer = accelerator.prepare(model, optimizer)
    
    logger.info(f"✓ Model initialized")
    logger.info(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting Stage 1 Training: {iterations} iterations")
    logger.info(f"{'='*70}\n")
    
    with tqdm(total=iterations, desc="Stage 1") as pbar:
        for iter_ in range(iterations):
            # Train
            output = train_iteration(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                accelerator=accelerator,
                iteration=iter_,
                t=gumbel_temperature,
                log_every=log_every,
                wandb_logging=wandb_logging,
            )
            
            # Evaluate
            if (iter_ + 1) % eval_every == 0 or (iter_ + 1) == iterations:
                eval_metrics = evaluate(model, eval_dataloader, device, gumbel_temperature)
                display_metrics(eval_metrics, title=f"Evaluation (Iter {iter_+1})")
                
                if wandb_logging:
                    wandb.log(eval_metrics, step=iter_+1)
            
            # Save checkpoint
            if (iter_ + 1) % save_model_every == 0 or (iter_ + 1) == iterations:
                if accelerator.is_main_process:
                    checkpoint_path = f"{log_dir}/checkpoint_{iter_+1}.pt"
                    torch.save({
                        'iteration': iter_ + 1,
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': model.config,
                    }, checkpoint_path)
                    logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{output.loss.item():.4f}",
                'patch': f"{output.patch_reconstruction_loss.item():.4f}",
                'global': f"{output.global_reconstruction_loss.item():.4f}",
            })
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Stage 1 Training Complete!")
    logger.info(f"Final checkpoint: {log_dir}/checkpoint_{iterations}.pt")
    logger.info(f"{'='*70}\n")
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    parse_config()
    train()