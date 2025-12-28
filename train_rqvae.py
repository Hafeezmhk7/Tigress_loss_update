import gin
import os
import torch
import numpy as np
import wandb
import time
from accelerate import Accelerator
from data.processed import ItemData, RecDataset
from data.utils import batch_to, cycle, next_batch, describe_dataloader
from modules.rqvae import RqVae
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
        vae_n_layers,
        wandb_logging,
        log_every,
        eval_every,
        save_model_every,
        log_dir,
        pbar,
    ): 
    # set model to training mode
    model.train()
    # set variables
    total_loss = 0
    eval_log = {}

    # Temperature decay for Gumbel-Softmax
    decay_steps = max(1, iteration // 1000)
    t = max(0.5 * (0.95 ** decay_steps), 0.01)

    if iteration == 0 and use_kmeans_init:
        logger.info("Using KMeans initialization for codebooks.")
        kmeans_init_data = batch_to(
            train_dataset[torch.arange(min(20000, len(train_dataset)))], device
        )
        model(kmeans_init_data, t)

    optimizer.zero_grad()
    for _ in range(gradient_accumulate_every):
        data = next_batch(train_dataloader, device)
        with accelerator.autocast():
            model_output = model(data, gumbel_t=t)
            loss = model_output.loss / gradient_accumulate_every
            total_loss += loss

    # autograd stuff
    accelerator.backward(total_loss)
    optimizer.step()
    accelerator.wait_for_everyone()

    # ===== TRIPLE-STAGE CONTRASTIVE: Track 6 losses =====
    losses[0].append(total_loss.cpu().item())
    losses[1].append(model_output.reconstruction_loss.cpu().item())
    losses[2].append(model_output.rqvae_loss.cpu().item())
    losses[3].append(model_output.encoder_infonce_loss.cpu().item())
    losses[4].append(model_output.multiscale_infonce_loss.cpu().item())
    losses[5].append(model_output.cost_infonce_loss.cpu().item())
    
    losses[0] = losses[0][-1000:]
    losses[1] = losses[1][-1000:]
    losses[2] = losses[2][-1000:]
    losses[3] = losses[3][-1000:]
    losses[4] = losses[4][-1000:]
    losses[5] = losses[5][-1000:]
    # ===== END =====
    
    if iteration % 100 == 0:
        print_loss = np.mean(losses[0])
        print_rec_loss = np.mean(losses[1])
        print_vae_loss = np.mean(losses[2])
        # ===== TRIPLE-STAGE CONTRASTIVE: Display all InfoNCE losses =====
        print_enc_infonce = np.mean(losses[3])
        print_ms_infonce = np.mean(losses[4])
        print_cost_infonce = np.mean(losses[5])
        
        pbar.set_description(
            f"loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, "
            f"vl: {print_vae_loss:.4f}, e_il: {print_enc_infonce:.4f}, "
            f"ms_il: {print_ms_infonce:.4f}, c_il: {print_cost_infonce:.4f}, t: {t:.3f}"
        )
        # ===== END =====

    # autograd
    accelerator.wait_for_everyone()
    optimizer.step()
    accelerator.wait_for_everyone()

    # compute logs depending on training model_output here to avoid cuda graph overwrite from eval graph.
    if accelerator.is_main_process:
        if ((iteration + 1) % log_every == 0 or iteration + 1 == iterations):
            emb_norms_avg = model_output.embs_norm.mean(axis=0)
            emb_norms_avg_log = {
                f"train/emb_avg_norm_{i}": emb_norms_avg[i].cpu().item()
                for i in range(vae_n_layers)
            }
            
            # ===== TRIPLE-STAGE CONTRASTIVE: Log all losses =====
            train_log = {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/total_loss": total_loss.cpu().item(),
                "train/reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                "train/rqvae_loss": model_output.rqvae_loss.cpu().item(),
                "train/encoder_infonce_loss": model_output.encoder_infonce_loss.cpu().item(),
                "train/multiscale_infonce_loss": model_output.multiscale_infonce_loss.cpu().item(),
                "train/cost_infonce_loss": model_output.cost_infonce_loss.cpu().item(),
                "train/temperature": t,
                "train/p_unique_ids": model_output.p_unique_ids.cpu().item(),
                **emb_norms_avg_log,
            }
            # ===== END =====
            
            # ===== Log gradient norms for diagnostics =====
            if iteration % (log_every * 10) == 0:  # Log less frequently to avoid overhead
                try:
                    grad_norms = {}
                    for i, layer in enumerate(model.layers):
                        if hasattr(layer, 'embedding') and layer.embedding.weight.grad is not None:
                            grad_norm = layer.embedding.weight.grad.norm().item()
                            grad_norms[f"diagnostics/grad_norm_level_{i}"] = grad_norm
                    
                    if grad_norms:
                        train_log.update(grad_norms)
                        # Compute gradient uniformity ratio
                        grad_values = list(grad_norms.values())
                        if len(grad_values) > 1:
                            grad_ratio = max(grad_values) / (min(grad_values) + 1e-8)
                            train_log["diagnostics/grad_uniformity_ratio"] = grad_ratio
                            
                            if grad_ratio > 100:
                                logger.warning(f"⚠️  Large gradient imbalance: {grad_ratio:.1f}x")
                except Exception as e:
                    # Don't fail training if gradient logging fails
                    logger.debug(f"Could not log gradients: {e}")
            # ===== END =====
            
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
            logger.info(f'Iteration {iteration+1}: Model saved.')
            
        if ((iteration + 1) % eval_every == 0 or (iteration + 1) == iterations):
            # start evaluation 
            logger.info('Evaluation Started!')
            eval_log = evaluate(
                model=model,
                eval_dataloader=eval_dataloader,
                device=device,
                tokenizer=tokenizer,
                index_dataset=index_dataset,
                vae_codebook_size=vae_codebook_size,
                vae_n_layers=vae_n_layers,
                t=t
            )
            
            # print eval metrics
            display_metrics(metrics=eval_log, title="Evaluation Metrics")
            
            # log metrics
            if wandb_logging:
                wandb.log(eval_log, step=iteration+1)
                
            model.train()  # switch back to training mode after validation

    return


def evaluate(model, eval_dataloader, device, tokenizer, index_dataset, 
             vae_codebook_size, vae_n_layers, t=0.2):
    """
    Enhanced evaluation with per-level codebook entropy tracking.
    """
    model.eval()
    eval_losses = [[], [], [], [], [], []]
    
    pbar = tqdm(eval_dataloader, desc=f"Eval")
    with torch.no_grad():
        for batch in pbar:
            data = batch_to(batch, device)
            model_output = model(data, gumbel_t=t)
            eval_losses[0].append(model_output.loss.cpu().item())
            eval_losses[1].append(model_output.reconstruction_loss.cpu().item())
            eval_losses[2].append(model_output.rqvae_loss.cpu().item())
            eval_losses[3].append(model_output.encoder_infonce_loss.cpu().item())
            eval_losses[4].append(model_output.multiscale_infonce_loss.cpu().item())
            eval_losses[5].append(model_output.cost_infonce_loss.cpu().item())

    eval_losses = np.array(eval_losses).mean(axis=-1)
    
    # Compute semantic IDs for entire corpus
    logger.info("Computing corpus semantic IDs for codebook analysis...")
    tokenizer.reset()
    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
    
    eval_log = {}
    
    # Per-level codebook analysis
    logger.info("Analyzing per-level codebook statistics...")
    for cid in range(vae_n_layers):
        level_ids = corpus_ids[:, cid]
        unique_codes, counts = torch.unique(level_ids, return_counts=True)
        
        # Sort by frequency (descending)
        sorted_indices = torch.argsort(counts, descending=True)
        sorted_codes = unique_codes[sorted_indices]
        sorted_counts = counts[sorted_indices]
        
        # 1. Codebook usage (percentage of codes used)
        usage = len(counts) / vae_codebook_size
        eval_log[f"eval/codebook_usage_{cid}"] = usage
        
        # 2. Codebook entropy (distribution uniformity)
        probs = counts.float() / counts.sum()
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(float(vae_codebook_size)))
        normalized_entropy = entropy / max_entropy
        
        eval_log[f"eval/codebook_entropy_{cid}"] = entropy.item()
        eval_log[f"eval/codebook_entropy_normalized_{cid}"] = normalized_entropy.item()
        
        # 3. Most frequent code percentage (concentration metric)
        max_count = sorted_counts[0].item()
        max_freq_percentage = max_count / corpus_ids.shape[0]
        eval_log[f"eval/codebook_max_freq_{cid}"] = max_freq_percentage
        
        # ===== NEW: EXPLICIT LEVEL-0 COLLAPSE METRICS =====
        if cid == 0:
            # Number of items using the most common Level-0 code
            eval_log[f"eval/level0_max_code_count"] = max_count
            eval_log[f"eval/level0_max_code_id"] = sorted_codes[0].item()
            
            # Top-5 most frequent codes
            top_k = min(5, len(sorted_codes))
            for k in range(top_k):
                code_id = sorted_codes[k].item()
                code_count = sorted_counts[k].item()
                code_percentage = code_count / corpus_ids.shape[0]
                eval_log[f"eval/level0_top{k+1}_code_id"] = code_id
                eval_log[f"eval/level0_top{k+1}_count"] = code_count
                eval_log[f"eval/level0_top{k+1}_percentage"] = code_percentage
            
            # Percentage of items covered by top-10 codes (concentration)
            top_10_k = min(10, len(sorted_counts))
            top_10_coverage = sorted_counts[:top_10_k].sum().item() / corpus_ids.shape[0]
            eval_log[f"eval/level0_top10_coverage"] = top_10_coverage
            
            # Percentage of items covered by top-20 codes
            top_20_k = min(20, len(sorted_counts))
            top_20_coverage = sorted_counts[:top_20_k].sum().item() / corpus_ids.shape[0]
            eval_log[f"eval/level0_top20_coverage"] = top_20_coverage
            
            # Effective number of codes (inverse of Herfindahl index)
            # If all items use 1 code: eff_num = 1
            # If perfectly uniform: eff_num = codebook_size
            herfindahl = (probs ** 2).sum()
            effective_num_codes = 1.0 / herfindahl.item()
            eval_log[f"eval/level0_effective_num_codes"] = effective_num_codes
            
            logger.info(f"")
            logger.info(f"{'='*70}")
            logger.info(f"Level-0 Collapse Analysis:")
            logger.info(f"{'='*70}")
            logger.info(f"  Most common code: {sorted_codes[0].item()} used by {max_count} items ({max_freq_percentage*100:.2f}%)")
            logger.info(f"  Top-5 codes cover: {sorted_counts[:top_k].sum().item()} items ({sorted_counts[:top_k].sum().item()/corpus_ids.shape[0]*100:.2f}%)")
            logger.info(f"  Top-10 codes cover: {top_10_coverage*100:.2f}% of all items")
            logger.info(f"  Top-20 codes cover: {top_20_coverage*100:.2f}% of all items")
            logger.info(f"  Effective # codes: {effective_num_codes:.1f} / {vae_codebook_size} ({effective_num_codes/vae_codebook_size*100:.1f}%)")
            logger.info(f"  Entropy (normalized): {normalized_entropy:.3f} (1.0 = perfect uniform)")
            logger.info(f"")
            
            # Print top-5 code distribution
            logger.info(f"  Top-5 Level-0 Code Distribution:")
            for k in range(top_k):
                code_id = sorted_codes[k].item()
                code_count = sorted_counts[k].item()
                code_percentage = code_count / corpus_ids.shape[0] * 100
                bar_length = int(code_percentage / 2)  # Scale to fit terminal
                bar = "█" * bar_length
                logger.info(f"    Code {code_id:3d}: {bar} {code_count:5d} items ({code_percentage:5.2f}%)")
            logger.info(f"{'='*70}")
            logger.info(f"")
        # ===== END NEW METRICS =====
        
        # 4. Gini coefficient (inequality of code distribution)
        sorted_counts_gini = torch.sort(counts)[0].float()
        n = len(sorted_counts_gini)
        index = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_counts_gini.device)
        gini = (2 * (index * sorted_counts_gini).sum()) / (n * sorted_counts_gini.sum()) - (n + 1) / n
        eval_log[f"eval/codebook_gini_{cid}"] = gini.item()
        
        if cid != 0:  # Only print non-Level-0 briefly
            logger.info(f"  Level {cid}: usage={usage:.3f}, entropy_norm={normalized_entropy:.3f}, "
                       f"max_freq={max_freq_percentage:.3f}, gini={gini:.3f}")
    
    # Overall semantic ID collision analysis
    logger.info("Analyzing semantic ID collisions...")
    max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]
    
    # Compute entropy of complete semantic IDs (excluding duplicate counter)
    unique_ids, counts = torch.unique(corpus_ids[:, :-1], dim=0, return_counts=True)
    p = counts.float() / corpus_ids.shape[0]
    rqvae_entropy = -(p * torch.log(p + 1e-10)).sum()
    
    # Collision metrics
    collision_rate = 1.0 - (len(unique_ids) / corpus_ids.shape[0])
    unique_ids_ratio = len(unique_ids) / corpus_ids.shape[0]
    
    # Average collisions per unique ID
    avg_collisions = counts.float().mean().item()
    max_collisions = counts.max().item()
    
    eval_log["eval/max_id_duplicates"] = max_duplicates.cpu().item()
    eval_log["eval/rqvae_entropy"] = rqvae_entropy.cpu().item()
    eval_log["eval/collision_rate"] = collision_rate
    eval_log["eval/unique_ids_ratio"] = unique_ids_ratio
    eval_log["eval/avg_collisions"] = avg_collisions
    eval_log["eval/max_collisions"] = max_collisions
    
    logger.info(f"  Collision rate: {collision_rate:.4f} ({(1-unique_ids_ratio)*100:.2f}% duplicates)")
    logger.info(f"  Unique IDs: {len(unique_ids)}/{corpus_ids.shape[0]} ({unique_ids_ratio*100:.2f}%)")
    logger.info(f"  Avg/Max collisions: {avg_collisions:.2f}/{max_collisions}")
    
    # Basic losses
    eval_log_final = {
        "eval/total_loss": eval_losses[0],
        "eval/reconstruction_loss": eval_losses[1],
        "eval/rqvae_loss": eval_losses[2],
        "eval/encoder_infonce_loss": eval_losses[3],
        "eval/multiscale_infonce_loss": eval_losses[4],
        "eval/cost_infonce_loss": eval_losses[5],
    }
    
    # Merge all evaluation metrics
    eval_log_final.update(eval_log)
    
    return eval_log_final
# ```

# ---

# ## **📊 New Metrics You'll See**

# ### **1. Level-0 Specific Metrics (in WandB):**

# | Metric | Description | Baseline Expected | With Encoder InfoNCE |
# |--------|-------------|-------------------|---------------------|
# | `eval/level0_max_code_count` | # items using most common code | 800-1500 | <200 |
# | `eval/level0_max_code_id` | Which code is most popular | Any | Varies |
# | `eval/level0_top1_percentage` | % items in top code | 15-30% | <5% |
# | `eval/level0_top10_coverage` | % items in top-10 codes | 60-80% | <40% |
# | `eval/level0_top20_coverage` | % items in top-20 codes | 80-95% | <60% |
# | `eval/level0_effective_num_codes` | Effective codebook diversity | 50-100 | >200 |

# ### **2. Console Output Example:**
# ```
# ======================================================================
# Level-0 Collapse Analysis:
# ======================================================================
#   Most common code: 42 used by 1234 items (24.68%)
#   Top-5 codes cover: 2891 items (57.82%)
#   Top-10 codes cover: 72.45% of all items
#   Top-20 codes cover: 89.32% of all items
#   Effective # codes: 67.3 / 256 (26.3%)
#   Entropy (normalized): 0.823 (1.0 = perfect uniform)

#   Top-5 Level-0 Code Distribution:
#     Code  42: ████████████ 1234 items (24.68%)
#     Code 105: ████████     891 items (17.82%)
#     Code  17: ██████       678 items (13.56%)
#     Code 234: ████          456 items ( 9.12%)
#     Code  89: ███           321 items ( 6.42%)
# ======================================================================


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
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
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_sim_vq=False,
    vae_n_layers=3,
    dataset_split="beauty",
    use_image_features=False,
    feature_combination_mode="sum",
    run_prefix="",
    debug=False,
    # ===== TRIPLE-STAGE CONTRASTIVE LEARNING =====
    use_encoder_infonce=False,
    use_multiscale_infonce=False,
    use_cost_infonce=False,
    infonce_temperature=0.07,
    encoder_infonce_weight=0.1,
    multiscale_infonce_weight=0.3,
    cost_infonce_weight=0.2,
    encoder_dropout_rate=0.1,
    # ===== END =====
    # ===== CROSS-ATTENTION =====
    use_cross_attn=False,
    attn_heads=8,
    # ===== END =====
):

    # create logdir if not exists
    uid = str(int(time.time()))
    logger.info(f"Session Started with UID '{uid}' | Dataset '{dataset_folder}' | Split '{dataset_split}'")
    log_dir = os.path.join(os.path.expanduser("~"), log_dir, dataset_split, uid)
    os.makedirs(log_dir, exist_ok=True)
    
    # setup accelerator and device
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device
    display_args(locals())

    # ===== TRIPLE-STAGE CONTRASTIVE: Update run name =====
    if wandb_logging and accelerator.is_main_process:
        params = locals()
        
        run_name = f"rq-vae-{dataset.name.lower()}-{dataset_split}" + "/" + uid
        
        # Add prefix to indicate which features are enabled
        features = []
        if use_encoder_infonce:
            features.append("enc-infonce")
        if use_multiscale_infonce:
            features.append("ms-infonce")
        if use_cost_infonce:
            features.append("cost-infonce")
        
        if features:
            run_name = f"triple-stage-{'-'.join(features)}-{run_name}"
        else:
            run_name = f"baseline-{run_name}"
        
        if run_prefix:
            run_name = f"{run_prefix}-{run_name}"
            
        run = wandb.init(entity="RecSys-UvA",
                         name=run_name,
                         project="Loss-RQ-VAE", 
                         config=params)
    # ===== END =====
        
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
        
    # ===== TRIPLE-STAGE CONTRASTIVE: Pass new parameters =====
    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
        use_encoder_infonce=use_encoder_infonce,
        use_multiscale_infonce=use_multiscale_infonce,
        use_cost_infonce=use_cost_infonce,
        infonce_temperature=infonce_temperature,
        encoder_infonce_weight=encoder_infonce_weight,
        multiscale_infonce_weight=multiscale_infonce_weight,
        cost_infonce_weight=cost_infonce_weight,
        encoder_dropout_rate=encoder_dropout_rate,
        use_cross_attn=use_cross_attn,
        attn_heads=attn_heads,
        mixed_precision_type=mixed_precision_type,
    )
    # ===== END =====
    display_model_summary(model, device)

    # ===== TRIPLE-STAGE CONTRASTIVE: Batch size check for InfoNCE =====
    if use_encoder_infonce or use_multiscale_infonce or use_cost_infonce:
        if batch_size < 64:
            logger.warning(
                f"⚠️  Batch size ({batch_size}) is small for InfoNCE contrastive learning. "
                f"Consider using batch_size >= 64 for better negative sampling. "
                f"Small batch size means fewer negative samples which can degrade contrastive loss effectiveness."
            )
        elif batch_size >= 256:
            logger.info(f"✅ Good batch size ({batch_size}) for InfoNCE contrastive learning (many negative samples)")
        else:
            logger.info(f"ℹ️  Acceptable batch size ({batch_size}) for InfoNCE. Larger (≥128) may improve performance.")
    # ===== END =====

    # setup optimizer
    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    start_iter = 0
    if pretrained_rqvae_path is not None:
        logger.info(f"[Resume Training] Loading pretrained RQ-VAE from {pretrained_rqvae_path}.")
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(
            pretrained_rqvae_path, map_location=device, weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iteration"] + 1

    model, optimizer = accelerator.prepare(model, optimizer)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer.rq_vae = model
    
    # ===== TRIPLE-STAGE CONTRASTIVE: Log configuration =====
    logger.info(f"Training Started! - Debugging: {debug}")
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"Triple-Stage Contrastive Learning Configuration:")
    logger.info(f"{'='*70}")
    logger.info(f"  Encoder InfoNCE:     {use_encoder_infonce}")
    logger.info(f"  Multi-Scale InfoNCE: {use_multiscale_infonce}")
    logger.info(f"  CoST InfoNCE:        {use_cost_infonce}")
    if use_encoder_infonce:
        logger.info(f"    - Weight: {encoder_infonce_weight}")
    if use_multiscale_infonce:
        logger.info(f"    - Weight: {multiscale_infonce_weight}")
    if use_cost_infonce:
        logger.info(f"    - Weight: {cost_infonce_weight}")
    if use_encoder_infonce or use_multiscale_infonce or use_cost_infonce:
        logger.info(f"  Temperature: {infonce_temperature}")
    logger.info(f"{'='*70}")
    logger.info(f"")
    
    # Log expected outcomes
    if not (use_encoder_infonce or use_multiscale_infonce or use_cost_infonce):
        logger.info("📊 Running BASELINE (TIGER) - Expect:")
        logger.info("   • Collision rate: ~2.95%")
        logger.info("   • Level 0 entropy: 0.82-0.85")
        logger.info("   • Level 2 codebook usage: ~70%")
    elif use_multiscale_infonce and not use_encoder_infonce and not use_cost_infonce:
        logger.info("🎯 Running MULTI-SCALE InfoNCE ONLY - Expect:")
        logger.info("   • Collision rate: <1%")
        logger.info("   • Level 2 codebook usage: >95%")
        logger.info("   • Gini coefficient: <0.20")
    elif use_encoder_infonce and use_multiscale_infonce and not use_cost_infonce:
        logger.info("🚀 Running ENCODER + MULTI-SCALE InfoNCE - Expect:")
        logger.info("   • Collision rate: <0.5%")
        logger.info("   • Level 0 entropy: >0.92")
        logger.info("   • Level 0 codebook usage: >95%")
        logger.info("   • All levels well-balanced")
    elif use_encoder_infonce and use_multiscale_infonce and use_cost_infonce:
        logger.info("🌟 Running FULL TRIPLE-STAGE - Expect:")
        logger.info("   • Collision rate: <0.5%")
        logger.info("   • Level 0 entropy: >0.92")
        logger.info("   • Semantic correlation: >0.80")
        logger.info("   • Neighbor preservation: >0.65")
        logger.info("   • NDCG improvement: +30-50%")
    logger.info("")
    # ===== END =====

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        # ===== TRIPLE-STAGE CONTRASTIVE: Track 6 losses =====
        losses = [[], [], [], [], [], []]  # total, reconstruction, rqvae, encoder_infonce, multiscale_infonce, cost_infonce
        # ===== END =====
        
        for iter_ in range(start_iter, start_iter + 1 + iterations):
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
                vae_n_layers=vae_n_layers,
                wandb_logging=wandb_logging,
                log_every=log_every,
                eval_every=eval_every,
                save_model_every=save_model_every,
                pbar=pbar,
                log_dir=log_dir,
            )
            pbar.update(1)

    if wandb_logging:
        wandb.finish()

    # ===== Final summary =====
    logger.info("")
    logger.info("="*70)
    logger.info("Training Complete!")
    logger.info("="*70)
    logger.info(f"Final checkpoint saved to: {log_dir}/checkpoint_{start_iter + iterations}.pt")
    logger.info("="*70)
    # ===== END =====


if __name__ == "__main__":
    parse_config()
    train()