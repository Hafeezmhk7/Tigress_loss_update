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
        use_patch_encoder,
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
            # Pass patches from the batch
            model(kmeans_init_data, t, kmeans_init_data.text_patches, kmeans_init_data.text_masks)
        else:
            model(kmeans_init_data, t)

    optimizer.zero_grad()
    for _ in range(gradient_accumulate_every):
        data = next_batch(train_dataloader, device)
        
        with accelerator.autocast():
            if use_patch_encoder:
                # Use patches from the batch (already loaded by ItemData)
                model_output = model(data, gumbel_t=t, text_patches=data.text_patches, text_mask=data.text_masks)
            else:
                model_output = model(data, gumbel_t=t)
            
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

    # # compute logs
    # if accelerator.is_main_process:
    #     if ((iteration + 1) % log_every == 0 or iteration + 1 == iterations):
    #         emb_norms_avg = model_output.embs_norm.mean(axis=0)
    #         emb_norms_avg_log = {
    #             f"train/emb_avg_norm_{i}": emb_norms_avg[i].cpu().item()
    #             for i in range(num_codebooks)
    #         }
    #         train_log = {
    #             "train/learning_rate": optimizer.param_groups[0]["lr"],
    #             "train/total_loss": total_loss.cpu().item(),
    #             "train/reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
    #             "train/pqvae_loss": model_output.pqvae_loss.cpu().item(),
    #             "train/temperature": t,
    #             "train/p_unique_ids": model_output.p_unique_ids.cpu().item(),
    #             **emb_norms_avg_log,
    #         }
            
            # if use_patch_encoder:
            #     train_log["train/diversity_loss"] = model_output.diversity_loss.cpu().item()
            
            # # print train metrics
            # display_metrics(metrics=train_log, title="Training Metrics")
            
            # # log metrics
            # if wandb_logging:
            #     wandb.log(train_log, step=iteration+1)
    
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


def evaluate(model, eval_dataloader, device, use_patch_encoder=False, t=0.2):
    model.eval()
    eval_losses = [[], [], [], []] if use_patch_encoder else [[], [], []]
    pbar = tqdm(eval_dataloader, desc=f"Eval")
    with torch.no_grad():
        for batch in pbar:
            data = batch_to(batch, device)
            
            if use_patch_encoder:
                # Use patches from the batch (already loaded by ItemData)
                model_output = model(data, gumbel_t=t, text_patches=data.text_patches, text_mask=data.text_masks)
            else:
                model_output = model(data, gumbel_t=t)
            
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
    dataset_folder="dataset/amazon/2014",
    dataset=RecDataset.AMAZON,
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
    eval_every=5000,
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
    num_codebooks=4,
    dataset_split="beauty",
    use_image_features=False,
    feature_combination_mode="sum",
    run_prefix="",
    debug=False,
    # Patch-based parameters (hierarchical)
    use_patch_encoder=False,
    patch_token_dim=1024,                # NEW: Input patch dimension
    patch_token_embed_dim=192,           # NEW: Dimension per semantic token
    patch_hidden_dim=512,                # NEW: Processing dimension
    patch_num_heads=8,                   # NEW: Attention heads
    patch_num_layers=2,                  # NEW: Self-attention layers
    patch_dropout=0.1,
    patch_diversity_weight=0.01,         # NEW: Diversity loss weight
    patch_hybrid_mode=False,
    patch_num_text_codebooks=3,          # NEW: For hybrid mode
    patch_num_image_codebooks=1,         # NEW: For hybrid mode
    # Decoder parameters (hierarchical)
    decoder_hidden_dim=512,              # NEW: Decoder hidden dimension
    decoder_num_layers=2,                # NEW: Decoder layers
    decoder_num_heads=8,                 # NEW: Decoder attention heads
    decoder_dropout=0.1,                 # NEW: Decoder dropout
    decoder_output_dim=768,              # NEW: Decoder output dimension
    # Patch embeddings parameters (for ItemData)
    use_patch_embeddings=False,
    patch_model_name="sentence-transformers/sentence-t5-xl",
    patch_max_seq_length=77,
):

    # create logdir if not exists
    uid = str(int(time.time()))
    logger.info(f"Session Started with UID '{uid}' | Dataset '{dataset_folder}' | Split '{dataset_split}'")
    if use_patch_encoder:
        logger.info(f"[PATCH MODE] Using Product Quantization with {num_codebooks} independent codebooks")
        logger.info(f"[HIERARCHICAL] Automatic patch processing: {use_patch_embeddings}")
    log_dir = os.path.join(os.path.expanduser("~"), log_dir, dataset_split, uid)
    os.makedirs(log_dir, exist_ok=True)
    
    # setup accelerator and device
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device
    display_args(locals())

    # logging
    if wandb_logging and accelerator.is_main_process:
        params = locals()
        mode_suffix = "-pq-hierarchical" if use_patch_encoder else "-pq"
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
        # Enable automatic patch processing
        use_patch_embeddings=use_patch_embeddings,
        patch_model_name=patch_model_name,
        patch_max_seq_length=patch_max_seq_length,
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
        use_patch_embeddings=use_patch_embeddings,
        patch_model_name=patch_model_name,
        patch_max_seq_length=patch_max_seq_length,
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
            use_patch_embeddings=use_patch_embeddings,
            patch_model_name=patch_model_name,
            patch_max_seq_length=patch_max_seq_length,
        )
    )

    # load model
    if use_image_features and feature_combination_mode == "concat":
        vae_input_dim = vae_input_dim * 2
    
    model = PqVae(
        input_dim=vae_input_dim,
        output_dim=decoder_output_dim,
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
        # Hierarchical patch parameters
        use_patch_encoder=use_patch_encoder,
        patch_token_dim=patch_token_dim,
        patch_token_embed_dim=patch_token_embed_dim,
        patch_hidden_dim=patch_hidden_dim,
        patch_num_heads=patch_num_heads,
        patch_num_layers=patch_num_layers,
        patch_dropout=patch_dropout,
        use_diversity_loss=True,
        diversity_weight=patch_diversity_weight,
        patch_hybrid_mode=patch_hybrid_mode,
        patch_num_text_codebooks=patch_num_text_codebooks,
        patch_num_image_codebooks=patch_num_image_codebooks,
        # Decoder parameters
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=decoder_num_layers,
        decoder_num_heads=decoder_num_heads,
        decoder_dropout=decoder_dropout,
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
                use_patch_encoder=use_patch_encoder,
                t=t,
            )
            pbar.update(1)

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()