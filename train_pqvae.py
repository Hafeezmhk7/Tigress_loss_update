import gin
import os
import torch
import numpy as np
import wandb
import time
from accelerate import Accelerator
from data.processed import ItemData, RecDataset
from data.utils import batch_to, cycle, next_batch, describe_dataloader
from modules.pqvae import PqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config, display_args, display_metrics, display_model_summary
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from tqdm import tqdm
from rich.logging import RichHandler
import logging


# logging
os.environ["WANDB__SERVICE_WAIT"] = "300"

# create logger
logger = logging.getLogger("recsys_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = RichHandler(show_path=False)
    logger.addHandler(handler)
    logger.propagate = False


def load_patch_embeddings(data_dir: str, dataset_split: str, device: str):
    """
    Load precomputed patch embeddings and masks.
    
    Returns:
        text_patches: torch.Tensor [N, max_seq_len, hidden_dim]
        text_masks: torch.Tensor [N, max_seq_len]
    """
    embeddings_path = os.path.join(data_dir, f"{dataset_split}_text_patches.npy")
    masks_path = os.path.join(data_dir, f"{dataset_split}_text_masks.npy")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Patch embeddings not found at {embeddings_path}. "
            f"Please run scripts/precompute_patch_embeddings.py first."
        )
    
    logger.info(f"Loading patch embeddings from {embeddings_path}")
    text_patches = np.load(embeddings_path)
    text_patches = torch.from_numpy(text_patches).float().to(device)
    
    logger.info(f"Loading attention masks from {masks_path}")
    text_masks = np.load(masks_path)
    text_masks = torch.from_numpy(text_masks).bool().to(device)
    
    logger.info(f"Loaded patch embeddings: {text_patches.shape}")
    logger.info(f"Loaded attention masks: {text_masks.shape}")
    
    return text_patches, text_masks


def train_iteration(
        model,
        optimizer,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        index_dataset,
        vae_codebook_size,
        accelerator,
        gradient_accumulate_every,
        device,
        iteration,
        iterations,
        losses,
        use_kmeans_init,
        train_dataset,
        num_codebooks,
        wandb_logging,
        log_every,
        eval_every,
        save_model_every,
        log_dir,
        pbar,
        # Patch-related arguments
        use_patch_encoder,
        text_patches_all,
        text_masks_all,
        t=0.2,
    ): 
    # set model to training mode
    model.train()
    # set variables
    total_loss = 0
    eval_log = {}

    if iteration == 0 and use_kmeans_init:
        logger.info("Using KMeans initialization for codebooks.")
        kmeans_init_data = batch_to(
            train_dataset[torch.arange(min(20000, len(train_dataset)))], device
        )
        if use_patch_encoder:
            # Get indices for this batch
            indices = kmeans_init_data.ids if hasattr(kmeans_init_data, 'ids') else torch.arange(len(kmeans_init_data.x))
            batch_text_patches = text_patches_all[indices] if text_patches_all is not None else None
            batch_text_masks = text_masks_all[indices] if text_masks_all is not None else None
            model(kmeans_init_data, t, batch_text_patches, batch_text_masks)
        else:
            model(kmeans_init_data, t)

    optimizer.zero_grad()
    for _ in range(gradient_accumulate_every):
        data = next_batch(train_dataloader, device)
        
        # Get patch embeddings for this batch if using patch encoder
        if use_patch_encoder:
            if hasattr(data, 'ids'):
                batch_indices = data.ids.flatten()
            else:
                batch_size = data.x.shape[0]
                batch_indices = torch.arange(batch_size, device=device)
            
            batch_text_patches = text_patches_all[batch_indices] if text_patches_all is not None else None
            batch_text_masks = text_masks_all[batch_indices] if text_masks_all is not None else None
        else:
            batch_text_patches = None
            batch_text_masks = None
        
        with accelerator.autocast():
            model_output = model(data, gumbel_t=t, text_patches=batch_text_patches, text_mask=batch_text_masks)
            loss = model_output.loss / gradient_accumulate_every
            total_loss += loss

    # autograd stuff
    accelerator.backward(total_loss)
    optimizer.step()
    accelerator.wait_for_everyone()

    # pbar metrics update
    losses[0].append(total_loss.cpu().item())
    losses[1].append(model_output.reconstruction_loss.cpu().item())
    losses[2].append(model_output.pqvae_loss.cpu().item())
    if use_patch_encoder:
        losses[3].append(model_output.diversity_loss.cpu().item())
    losses[0] = losses[0][-1000:]
    losses[1] = losses[1][-1000:]
    losses[2] = losses[2][-1000:]
    if use_patch_encoder:
        losses[3] = losses[3][-1000:]
    
    if iteration % 100 == 0:
        print_loss = np.mean(losses[0])
        print_rec_loss = np.mean(losses[1])
        print_pq_loss = np.mean(losses[2])
        
        if use_patch_encoder:
            print_div_loss = np.mean(losses[3])
            pbar.set_description(
                f"loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, pq: {print_pq_loss:.4f}, dl: {print_div_loss:.4f}"
            )
        else:
            pbar.set_description(
                f"loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, pq: {print_pq_loss:.4f}"
            )

    # autograd
    accelerator.wait_for_everyone()
    optimizer.step()
    accelerator.wait_for_everyone()

    # compute logs
    if accelerator.is_main_process:
        if ((iteration + 1) % log_every == 0 or iteration + 1 == iterations):
            emb_norms_avg = model_output.embs_norm.mean(axis=0)
            emb_norms_avg_log = {
                f"train/emb_avg_norm_{i}": emb_norms_avg[i].cpu().item()
                for i in range(num_codebooks)
            }
            train_log = {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/total_loss": total_loss.cpu().item(),
                "train/reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                "train/pqvae_loss": model_output.pqvae_loss.cpu().item(),
                "train/temperature": t,
                "train/p_unique_ids": model_output.p_unique_ids.cpu().item(),
                **emb_norms_avg_log,
            }
            
            if use_patch_encoder:
                train_log["train/diversity_loss"] = model_output.diversity_loss.cpu().item()
            
            # print train metrics
            display_metrics(metrics=train_log, title="Training Metrics")
            
            # log metrics
            if wandb_logging:
                wandb.log(train_log, step=iteration+1)
    
    # evaluate model
    if accelerator.is_main_process:
        # save model checkpoint
        if ((iteration + 1) % save_model_every == 0 or (iteration + 1) == iterations):
            state = {
                "iteration": iteration+1,
                "model": model.state_dict(),
                "model_config": model.config,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, f"{log_dir}/checkpoint_{iteration+1}.pt")
            logger.info(f'Iteration {iteration}: Model saved.')
            
        if ((iteration + 1) % eval_every == 0 or (iteration + 1) == iterations):
            # start evaluation 
            logger.info('Evaluation Started!')
            eval_log = evaluate(
                model, eval_dataloader, device,
                use_patch_encoder=use_patch_encoder,
                text_patches_all=text_patches_all,
                text_masks_all=text_masks_all,
                t=t
            )

            # log entropy and duplicates
            tokenizer.reset()
            corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
            max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]
            _, counts = torch.unique(
                corpus_ids[:, :-1], dim=0, return_counts=True
            )
            p = counts / corpus_ids.shape[0]
            pqvae_entropy = -(p * torch.log(p)).sum()

            for cid in range(num_codebooks):
                _, counts = torch.unique(corpus_ids[:, cid], return_counts=True)
                eval_log[f"eval/codebook_usage_{cid}"] = (
                    len(counts) / vae_codebook_size
                )
            eval_log["eval/pqvae_entropy"] = pqvae_entropy.cpu().item()
            eval_log["eval/max_id_duplicates"] = max_duplicates.cpu().item()
            
            # print eval metrics
            display_metrics(metrics=eval_log, title="Evaluation Metrics")
            
            # log metrics
            if wandb_logging:
                wandb.log(eval_log, step=iteration+1)
                
            model.train()  # switch back to training mode after validation

    return


def evaluate(model, eval_dataloader, device, use_patch_encoder=False, 
             text_patches_all=None, text_masks_all=None, t=0.2):
    model.eval()
    eval_losses = [[], [], [], []] if use_patch_encoder else [[], [], []]
    pbar = tqdm(eval_dataloader, desc=f"Eval")
    with torch.no_grad():
        for batch in pbar:
            data = batch_to(batch, device)
            
            # Get patch embeddings for this batch if using patch encoder
            if use_patch_encoder:
                if hasattr(data, 'ids'):
                    batch_indices = data.ids.flatten()
                else:
                    batch_size = data.x.shape[0]
                    batch_indices = torch.arange(batch_size, device=device)
                
                batch_text_patches = text_patches_all[batch_indices] if text_patches_all is not None else None
                batch_text_masks = text_masks_all[batch_indices] if text_masks_all is not None else None
            else:
                batch_text_patches = None
                batch_text_masks = None
            
            model_output = model(data, gumbel_t=t, text_patches=batch_text_patches, text_mask=batch_text_masks)
            eval_losses[0].append(model_output.loss.cpu().item())
            eval_losses[1].append(model_output.reconstruction_loss.cpu().item())
            eval_losses[2].append(model_output.pqvae_loss.cpu().item())
            if use_patch_encoder:
                eval_losses[3].append(model_output.diversity_loss.cpu().item())

    eval_losses = np.array(eval_losses).mean(axis=-1)
    
    eval_log = {
        "eval/total_loss": eval_losses[0],
        "eval/reconstruction_loss": eval_losses[1],
        "eval/pqvae_loss": eval_losses[2],
    }
    
    if use_patch_encoder:
        eval_log["eval/diversity_loss"] = eval_losses[3]
    
    return eval_log


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    dataset=RecDataset.ML_1M,
    pretrained_pqvae_path=None,
    log_dir="logdir/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    log_every=1000,
    commitment_weight=0.25,
    vae_n_cat_feats=0,
    vae_input_dim=768,
    vae_embed_dim=32,
    vae_hidden_dims=[512, 256, 128],
    vae_codebook_size=256,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.ROTATION_TRICK,
    vae_sim_vq=False,
    num_codebooks=4,  # Number of independent codebooks (patches)
    dataset_split="beauty",
    use_image_features=False,
    feature_combination_mode="sum",
    run_prefix="",
    debug=False,
    # Patch-based parameters
    use_patch_encoder=False,
    patch_num_patches=4,
    patch_hidden_dim=256,
    patch_num_heads=4,
    patch_dropout=0.1,
    patch_diversity_weight=0.1,
    patch_hybrid_mode=False,
    patch_embeddings_dir="data/processed",
):

    # create logdir if not exists
    uid = str(int(time.time()))
    logger.info(f"Session Started with UID '{uid}' | Dataset '{dataset_folder}' | Split '{dataset_split}'")
    if use_patch_encoder:
        logger.info(f"[PATCH MODE] Using Product Quantization with {num_codebooks} independent codebooks")
    log_dir = os.path.join(os.path.expanduser("~"), log_dir, dataset_split, uid)
    os.makedirs(log_dir, exist_ok=True)
    
    # setup accelerator and device
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device
    display_args(locals())

    # Load patch embeddings if using patch encoder
    text_patches_all = None
    text_masks_all = None
    if use_patch_encoder:
        text_patches_all, text_masks_all = load_patch_embeddings(
            data_dir=patch_embeddings_dir,
            dataset_split=dataset_split,
            device=device
        )

    # logging
    if wandb_logging and accelerator.is_main_process:
        params = locals()
        mode_suffix = "-pq" if use_patch_encoder else ""
        run_name = f"pq-vae{mode_suffix}-{dataset.name.lower()}-{dataset_split}" + "/" + uid
        if run_prefix:
            run_name = f"{run_prefix}-{run_name}"
        run = wandb.init(entity="RecSys-UvA",
                         name=run_name,
                         project="pq-vae-training", 
                         config=params)
        
    # load train dataset
    train_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        train_test_split="train",
        split=dataset_split,
        use_image_features=use_image_features,
        feature_combination_mode=feature_combination_mode,
        device=device,
    )
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=None,
        collate_fn=lambda batch: batch,
    )
    describe_dataloader(train_dataloader, train_sampler, title="Train DataLoader Summary")
    train_dataloader = cycle(train_dataloader)
    train_dataloader = accelerator.prepare(train_dataloader)

    # load eval dataset
    eval_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=False,
        train_test_split="eval",
        split=dataset_split,
        use_image_features=use_image_features,
        feature_combination_mode=feature_combination_mode,
        device=device,
    )
    eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=None,
        collate_fn=lambda batch: batch,
    )
    describe_dataloader(eval_dataloader, eval_sampler, title="Eval DataLoader Summary")
    
    # load all the items dataset
    index_dataset = (
        ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=False,
            train_test_split="all",
            split=dataset_split,
            use_image_features=use_image_features,
            feature_combination_mode=feature_combination_mode,
            device=device,
        )
    )

    # load model
    if use_image_features and feature_combination_mode == "concat":
        vae_input_dim = vae_input_dim * 2
    
    model = PqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_pqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        num_codebooks=num_codebooks,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
        # Patch parameters
        use_patch_encoder=use_patch_encoder,
        patch_num_patches=patch_num_patches,
        patch_hidden_dim=patch_hidden_dim,
        patch_num_heads=patch_num_heads,
        patch_dropout=patch_dropout,
        patch_diversity_weight=patch_diversity_weight,
        patch_hybrid_mode=patch_hybrid_mode,
    )
    display_model_summary(model, device)

    # setup optimizer
    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    start_iter = 0
    if pretrained_pqvae_path is not None:
        logger.info(f"[Resume Training] Loading pretrained PQ-VAE from {pretrained_pqvae_path}.")
        model.load_pretrained(pretrained_pqvae_path)
        state = torch.load(
            pretrained_pqvae_path, map_location=device, weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iteration"] + 1

    model, optimizer = accelerator.prepare(model, optimizer)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=num_codebooks,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_pqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer.rq_vae = model
    
    # starting the training
    logger.info(f"Training Started! - Debugging: {debug}")

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        losses = [[], [], [], []] if use_patch_encoder else [[], [], []]
        for iter_ in range(start_iter, start_iter + 1 + iterations):
            t = 0.2
            train_iteration(
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                index_dataset=index_dataset,
                vae_codebook_size=vae_codebook_size,
                accelerator=accelerator,
                gradient_accumulate_every=gradient_accumulate_every,
                device=device,
                iteration=iter_,
                iterations=iterations,
                losses=losses,
                use_kmeans_init=use_kmeans_init,
                train_dataset=train_dataset,
                num_codebooks=num_codebooks,
                wandb_logging=wandb_logging,
                log_every=log_every,
                eval_every=eval_every,
                save_model_every=save_model_every,
                pbar=pbar,
                log_dir=log_dir,
                # Patch parameters
                use_patch_encoder=use_patch_encoder,
                text_patches_all=text_patches_all,
                text_masks_all=text_masks_all,
                t=t,
            )
            pbar.update(1)

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()