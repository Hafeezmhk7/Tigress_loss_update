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
from modules.constrastive_rqvae import ContrastiveMultimodalRqVae
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
    model, optimizer, tokenizer, train_dataloader, eval_dataloader, 
    index_dataset, vae_codebook_size, accelerator,
    gradient_accumulate_every, device, iteration, iterations,
    losses, use_kmeans_init, train_dataset, 
    vae_n_layers, wandb_logging, log_every,
    eval_every, save_model_every,
    log_dir, pbar,
    t=0.2,
):
    model.train()
    # set variables
    total_loss = 0
    eval_log = {}

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

    # autograd
    accelerator.backward(total_loss)
    optimizer.step()
    accelerator.wait_for_everyone()

    # dynamically handle losses/metrics
    if not losses:  # initialize if empty
        metric_names = [k for k in model_output._fields]  # assumes namedtuple
        for _ in metric_names:
            losses.append([])

    for i, name in enumerate(model_output._fields):
        val = getattr(model_output, name)
        if torch.is_tensor(val):
            val = val.cpu().item()
        losses[i].append(val)
        losses[i] = losses[i][-1000:]  # keep last 1000

    # update pbar dynamically
    if iteration % 100 == 0:
        desc = ", ".join(f"{name}: {np.mean(losses[i]):.4f}" for i, name in enumerate(model_output._fields))
        pbar.set_description(desc)

    # compute logs depending on training model_output
    if accelerator.is_main_process:
        if ((iteration + 1) % log_every == 0 or iteration + 1 == iterations):
            emb_norms_avg = model_output.embs_norm.mean(axis=0)
            emb_norms_avg_log = {
                f"train/emb_avg_norm_{i}": emb_norms_avg[i].cpu().item()
                for i in range(vae_n_layers)
            }
            train_log = {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/temperature": t,
                "train/p_unique_ids": model_output.p_unique_ids.cpu().item(),
                **emb_norms_avg_log,
            }
            # add all dynamic metrics
            for i, name in enumerate(model_output._fields):
                val = getattr(model_output, name)
                if torch.is_tensor(val):
                    val = val.cpu().item()
                train_log[f"train/{name}"] = val

            display_metrics(metrics=train_log, title="Training Metrics")
            if wandb_logging:
                wandb.log(train_log, step=iteration + 1)

    # evaluation and checkpoint
    if accelerator.is_main_process:
        # save model checkpoint
        if ((iteration + 1) % save_model_every == 0 or (iteration + 1) == iterations):
            state = {
                "iteration": iteration + 1,
                "model": model.state_dict(),
                "model_config": model.config,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, f"{log_dir}/checkpoint_{iteration + 1}.pt")
            logger.info(f'Iteration {iteration}: Model saved.')

        if ((iteration + 1) % eval_every == 0 or (iteration + 1) == iterations):
            # start evaluation 
            logger.info('Evaluation Started!')
            eval_log = evaluate(model, eval_dataloader, device, t=t)

            # log entropy and duplicates
            tokenizer.reset()
            corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
            max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]
            _, counts = torch.unique(corpus_ids[:, :-1], dim=0, return_counts=True)
            p = counts / corpus_ids.shape[0]
            rqvae_entropy = -(p * torch.log(p)).sum()

            for cid in range(vae_n_layers):
                _, counts = torch.unique(corpus_ids[:, cid], return_counts=True)
                eval_log[f"eval/codebook_usage_{cid}"] = len(counts) / vae_codebook_size
            eval_log["eval/rqvae_entropy"] = rqvae_entropy.cpu().item()
            eval_log["eval/max_id_duplicates"] = max_duplicates.cpu().item()

            # print eval metrics
            display_metrics(metrics=eval_log, title="Evaluation Metrics")
            
            # log metrics
            if wandb_logging:
                wandb.log(eval_log, step=iteration + 1)
                
            model.train()  # switch back to training mode after validation


def evaluate(model, eval_dataloader, device, t=0.2):
    model.eval()
    metrics_dict = {name: [] for name in model._fields}

    pbar = tqdm(eval_dataloader, desc="Eval")
    with torch.no_grad():
        for batch in pbar:
            data = batch_to(batch, device)
            out = model(data, gumbel_t=t)

            for name in out._fields:
                val = getattr(out, name)
                if torch.is_tensor(val):
                    val = val.cpu().item()
                metrics_dict[name].append(val)

    # compute mean for each metric
    eval_metrics = {f"eval/{k}": np.mean(v) for k, v in metrics_dict.items()}
    return eval_metrics


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
    use_cross_attn = False,
    attn_heads = 8,
    debug=False,
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

    # logging
    if wandb_logging and accelerator.is_main_process:
        params = locals()
        # wandb.login()
        run_name = f"rq-vae-{dataset.name.lower()}-{dataset_split}" + "/" + uid
        if run_prefix:
            run_name = f"{run_prefix}-{run_name}"
        run = wandb.init(entity="RecSys-UvA",
                         name=run_name,
                         project="rq-vae-training", 
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
    # TODO: Investigate bug with prepare eval_dataloader

    # load model
    if use_image_features and feature_combination_mode == "concat":
        vae_input_dim = vae_input_dim * 2
        
    model = ContrastiveMultimodalRqVae(
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
        use_cross_attn=use_cross_attn,
        attn_heads=attn_heads,
        mixed_precision_type=mixed_precision_type,
        use_contrastive=True,                       
        contrastive_weight=0.5,                     
        recon_contrastive_weight=0.5,               
    )

    # model = RqVae(
    #     input_dim=vae_input_dim,
    #     embed_dim=vae_embed_dim,
    #     hidden_dims=vae_hidden_dims,
    #     codebook_size=vae_codebook_size,
    #     codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
    #     codebook_normalize=vae_codebook_normalize,
    #     codebook_sim_vq=vae_sim_vq,
    #     codebook_mode=vae_codebook_mode,
    #     n_layers=vae_n_layers,
    #     n_cat_features=vae_n_cat_feats,
    #     commitment_weight=commitment_weight,
    #     use_cross_attn=use_cross_attn,
    #     attn_heads=attn_heads,
    #     mixed_precision_type=mixed_precision_type,
    # )
    display_model_summary(model, device)

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
    
    # starting the training
    logger.info(f"Training Started! - Debugging: {debug}")

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        # list of lists, one per metric dynamically
        losses = []

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
                vae_n_layers=vae_n_layers,
                wandb_logging=wandb_logging,
                log_every=log_every,
                eval_every=eval_every,
                save_model_every=save_model_every,
                pbar=pbar,
                log_dir=log_dir,
                t=t,
            )
            pbar.update(1)

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
