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
from modules.utils import parse_config, display_args, display_metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logger = logging.getLogger("recsys_logger")


def sample_sequence_pairs(seq_data, num_samples=1000):
    """Sample sequence pairs for contrastive learning"""
    sequences = []
    for i in range(min(num_samples, len(seq_data))):
        batch = seq_data[i]
        item_ids = batch.ids[batch.seq_mask].tolist()
        if len(item_ids) >= 2:
            sequences.append(item_ids)
    
    return extract_sequence_pairs(sequences, num_negatives=16)


def train_iteration_stage1(
    model, optimizer, train_dataloader, device, accelerator, iteration, t=0.2
):
    """Stage 1: Item-level training"""
    model.train()
    model.set_stage(1)
    
    data = next_batch(train_dataloader, device)
    
    optimizer.zero_grad()
    
    with accelerator.autocast():
        output = model(
            data,
            gumbel_t=t,
            text_patches=data.text_patches,
            text_mask=data.text_masks,
        )
        loss = output.loss
    
    accelerator.backward(loss)
    optimizer.step()
    
    return output


def train_iteration_stage2(
    model, optimizer, train_dataloader, seq_data, device, accelerator, iteration, t=0.2
):
    """Stage 2: Sequence-level training"""
    model.train()
    model.set_stage(2)
    
    # Get item batch
    data = next_batch(train_dataloader, device)
    
    # Sample sequence triplets
    triplets = sample_sequence_pairs(seq_data, num_samples=100)
    
    # Convert to tensors
    if triplets:
        anchors = torch.tensor([t[0] for t in triplets], device=device)
        positives = torch.tensor([t[1] for t in triplets], device=device)
        negatives = torch.tensor([t[2] for t in triplets], device=device)
        
        # Get semantic IDs for triplets
        with torch.no_grad():
            anchor_batch = data.__class__(
                **{k: v[anchors] if k in ['x', 'ids'] else v for k, v in data._asdict().items()}
            )
            positive_batch = data.__class__(
                **{k: v[positives] if k in ['x', 'ids'] else v for k, v in data._asdict().items()}
            )
            
            anchor_codes = model.get_semantic_ids(
                anchor_batch, anchor_batch.text_patches, anchor_batch.text_masks
            ).sem_ids
            
            positive_codes = model.get_semantic_ids(
                positive_batch, positive_batch.text_patches, positive_batch.text_masks
            ).sem_ids
            
            # Negatives: [B, num_neg]
            negative_codes_list = []
            for neg_list in negatives:
                neg_codes = []
                for neg_id in neg_list:
                    neg_batch = data.__class__(
                        **{k: v[neg_id:neg_id+1] if k in ['x', 'ids'] else v for k, v in data._asdict().items()}
                    )
                    neg_code = model.get_semantic_ids(
                        neg_batch, neg_batch.text_patches, neg_batch.text_masks
                    ).sem_ids
                    neg_codes.append(neg_code)
                negative_codes_list.append(torch.cat(neg_codes, dim=0))
            negative_codes = torch.stack(negative_codes_list, dim=0)
    else:
        anchor_codes = None
        positive_codes = None
        negative_codes = None
    
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
    
    accelerator.backward(loss)
    optimizer.step()
    
    return output


def evaluate_stage1(model, eval_dataloader, device, t=0.2):
    """Evaluate Stage 1"""
    model.eval()
    model.set_stage(1)
    
    losses = []
    patch_losses = []
    global_losses = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
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
    
    return {
        'eval/loss': np.mean(losses),
        'eval/patch_recon': np.mean(patch_losses),
        'eval/global_recon': np.mean(global_losses),
    }


@gin.configurable
def train(
    iterations=50000,
    batch_size=512,
    learning_rate=1e-3,
    dataset_folder="dataset/amazon/2014",
    dataset_split="beauty",
    log_dir="logdir/pqvae_twostage",
    stage1_iterations=50000,
    stage2_iterations=30000,
    stage1_checkpoint_path=None,
    use_patch_embeddings=True,
    patch_model_name="sentence-transformers/sentence-t5-xl",
    wandb_logging=True,
    save_model_every=10000,
    eval_every=5000,
    log_every=1000,
    **kwargs
):
    """Two-stage training"""
    
    uid = str(int(time.time()))
    log_dir = os.path.join(os.path.expanduser("~"), log_dir, dataset_split, f"stage1_{uid}")
    os.makedirs(log_dir, exist_ok=True)
    
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load datasets
    train_dataset = ItemData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        train_test_split="train",
        split=dataset_split,
        use_patch_embeddings=use_patch_embeddings,
        patch_model_name=patch_model_name,
        device=device,
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)
    
    # For Stage 2: sequence data
    seq_train_data = SeqData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        data_split="train",
        split=dataset_split,
        device=device,
    )
    
    # Model
    model = PqVaeTwoStage(**kwargs)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    if stage1_checkpoint_path:
        model.load_pretrained(stage1_checkpoint_path)
        logger.info(f"Loaded Stage 1 checkpoint from {stage1_checkpoint_path}")
    
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # STAGE 1 TRAINING
    logger.info("=" * 70)
    logger.info("STAGE 1: Item-Level Training (Patch + Global Reconstruction)")
    logger.info("=" * 70)
    
    with tqdm(total=stage1_iterations) as pbar:
        for iter_ in range(stage1_iterations):
            output = train_iteration_stage1(
                model, optimizer, train_dataloader, device, accelerator, iter_
            )
            
            if (iter_ + 1) % log_every == 0:
                metrics = {
                    'stage1/loss': output.loss.cpu().item(),
                    'stage1/patch_recon': output.patch_reconstruction_loss.cpu().item(),
                    'stage1/global_recon': output.global_reconstruction_loss.cpu().item(),
                }
                if wandb_logging:
                    wandb.log(metrics, step=iter_)
            
            if (iter_ + 1) % save_model_every == 0:
                torch.save({
                    'iteration': iter_ + 1,
                    'model': model.state_dict(),
                }, f"{log_dir}/stage1_checkpoint_{iter_+1}.pt")
            
            pbar.update(1)
    
    # STAGE 2 TRAINING
    logger.info("=" * 70)
    logger.info("STAGE 2: Sequence-Level Training (+ Contrastive)")
    logger.info("=" * 70)
    
    log_dir_stage2 = log_dir.replace("stage1", "stage2")
    os.makedirs(log_dir_stage2, exist_ok=True)
    
    with tqdm(total=stage2_iterations) as pbar:
        for iter_ in range(stage2_iterations):
            output = train_iteration_stage2(
                model, optimizer, train_dataloader, seq_train_data, device, accelerator, iter_
            )
            
            if (iter_ + 1) % log_every == 0:
                metrics = {
                    'stage2/loss': output.loss.cpu().item(),
                    'stage2/seq_contrastive': output.sequence_contrastive_loss.cpu().item(),
                }
                if wandb_logging:
                    wandb.log(metrics, step=stage1_iterations + iter_)
            
            if (iter_ + 1) % save_model_every == 0:
                torch.save({
                    'iteration': iter_ + 1,
                    'model': model.state_dict(),
                }, f"{log_dir_stage2}/stage2_checkpoint_{iter_+1}.pt")
            
            pbar.update(1)
    
    logger.info("✅ Two-stage training complete!")


if __name__ == "__main__":
    parse_config()
    train()