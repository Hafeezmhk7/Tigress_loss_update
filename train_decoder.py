import argparse
import os
import gin
import torch
import wandb
import time
from accelerate import Accelerator
from data.processed import ItemData, RecDataset, SeqData
from data.utils import batch_to, cycle, next_batch, describe_dataloader
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import (compute_debug_metrics, parse_config, 
                           display_args, display_metrics, 
                           display_model_summary, set_seed, 
                           collate_seqbatch, clamp_ids)
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from tqdm import tqdm
from rich.logging import RichHandler
import logging
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# logging
os.environ["WANDB__SERVICE_WAIT"] = "300"

# create logger
logger = logging.getLogger("recsys_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = RichHandler(show_path=False)
    logger.addHandler(handler)
    logger.propagate = False


def train_iteration(model, optimizer, tokenizer,
                    accelerator, lr_scheduler, metrics_accumulator,
                    train_dataloader, eval_dataloader,
                    gradient_accumulate_every, device,
                    pbar, log_dir, iteration, iterations,
                    save_model_every, log_every, eval_every, 
                    use_image_features,
                    wandb_logging):
    # set model to training mode
    model.train()
    # set variables
    total_loss = 0
    debug_metrics = []
    num_batches = 0

    optimizer.zero_grad()
    for _ in range(gradient_accumulate_every):
        data = next_batch(train_dataloader, device)
        tokenized_data = tokenizer(data)
        # clamp sem ids
        valid_max = model.num_embeddings - 1
        tokenized_data = clamp_ids(tokenized_data, valid_max)

        with accelerator.autocast():
            model_output = model(tokenized_data)
            loss = model_output.loss / gradient_accumulate_every
            total_loss += loss
            num_batches += 1

        if accelerator.is_main_process:
            train_debug_metrics = compute_debug_metrics(
                tokenized_data, model_output, "train"
            )
            debug_metrics.append(train_debug_metrics)

    accelerator.backward(total_loss)
    assert model.sem_id_embedder.emb.weight.grad is not None

    if iteration % 100 == 0:
        pbar.set_description(f"loss: {total_loss.item():.4f}")
    # autograd stuff
    accelerator.wait_for_everyone()
    optimizer.step()
    lr_scheduler.step()
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        if ((iteration + 1) % log_every == 0 or iteration + 1 == iterations):
            train_log = {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/total_loss": total_loss.cpu().item(),
            }
            # average debug metrics
            averaged_debug_metrics = {}
            for key in debug_metrics[0].keys():
                averaged_debug_metrics[key] = sum(d[key] for d in debug_metrics) / num_batches
            train_log.update(averaged_debug_metrics)

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
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
            }
            torch.save(state, f"{log_dir}/checkpoint_{iteration+1}.pt")
            logger.info(f'Iteration {iteration}: Model saved.')
            
        if ((iteration + 1) % eval_every == 0 or (iteration + 1) == iterations):
            # start evaluation    
            logger.info('Evaluation Started!')
            eval_log = evaluate(model, eval_dataloader, device, tokenizer, 
                                metrics_accumulator, use_image_features)
            
            # print eval metrics
            display_metrics(eval_log, title="Evaluation Metrics")
            
            # log metrics
            if wandb_logging:
                wandb.log(eval_log, step=iteration+1)
                
            model.train()  # switch back to training mode after validation

    return


def evaluate(model, eval_dataloader, device, tokenizer, metrics_accumulator, use_image_features):
    # set model to evaluation mode
    model.eval()
    total_loss = 0
    debug_metrics = []
    num_batches = 0
    pbar = tqdm(eval_dataloader, desc=f"Eval")
    for batch in pbar:
        data = batch_to(batch, device)
        tokenized_data = tokenizer(data)
        # clamp sem ids
        valid_max = model.num_embeddings - 1
        tokenized_data = clamp_ids(tokenized_data, valid_max)
        model.enable_generation = False
        # debug metrics
        with torch.no_grad():
            model_output_eval = model(tokenized_data)
            loss = model_output_eval.loss.detach().cpu().item()
            total_loss += loss
            num_batches += 1
            eval_debug_metrics = compute_debug_metrics(
                tokenized_data, model_output_eval, "eval"
            )
            debug_metrics.append(eval_debug_metrics)
        # eval metrics
        model.enable_generation = True
        generated = model.generate_next_sem_id(
            tokenized_data, top_k=True, temperature=1
        )
        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
        # calculate IR measures
        metrics_accumulator.accumulate(
            actual=actual, top_k=top_k, tokenizer=tokenizer
        )
        
    eval_metrics = metrics_accumulator.reduce()
    eval_metrics = {f"metrics/{k}": v for k, v in eval_metrics.items()}
    # reset the metrics accumulator
    metrics_accumulator.reset()
    
    # average debug metrics
    averaged_debug_metrics = {}
    for key in debug_metrics[0].keys():
        averaged_debug_metrics[key] = sum(d[key] for d in debug_metrics) / num_batches

    averaged_debug_metrics["eval/loss"] = total_loss / num_batches
    eval_metrics.update(averaged_debug_metrics)
            
    return eval_metrics


@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    log_dir="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    log_every=5000,
    eval_every=5000,
    save_model_every=1000000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    rope=False,
    prefix_matching=False,
    dataset_split="beauty",
    train_data_subsample=True,
    model_jagged_mode=True,
    category=None,
    use_image_features=False,
    feature_combination_mode="sum",
    run_prefix="",
    debug=False,
    enable_image_cross_attn=False,
    use_rqvae_cross_attn=False,
):

    # create logdir if not exists
    uid = str(int(time.time()))
    logger.info(
        f"Session Started with UID '{uid}' | Dataset '{dataset_folder}' | Split '{dataset_split}'")
    log_dir = os.path.join(os.path.expanduser("~"), log_dir, dataset_split, uid)
    os.makedirs(log_dir, exist_ok=True)

    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    # setup accelerator and device
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device
    display_args(locals())
    
    if pretrained_rqvae_path is None:
        logger.error("No pretrained rqvae path provided. Please provide a valid path to continue training.")
        return
    
    # extract rq-vae uid
    rqvae_uid = pretrained_rqvae_path.split("/")[-2]

    # logging
    if wandb_logging and accelerator.is_main_process:
        # get local scope parameters for logging
        params = locals()
        # wandb.login()
        run_name = f"decoder-{dataset.name.lower()}-{dataset_split}" + \
            "/" + rqvae_uid + "/" + uid
        if run_prefix:
            run_name = f"{run_prefix}-{run_name}"
        run = wandb.init(entity="RecSys-UvA",
                         name=run_name,
                         project="Loss-Decoder",
                         config=params)

    # load items dataset
    item_dataset = (
        ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=force_dataset_process,
            split=dataset_split,
            use_image_features=use_image_features,
            feature_combination_mode=feature_combination_mode,
            device=device,
        )
        if category is None
        else ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=force_dataset_process,
            split=dataset_split,
            category=category,
            use_image_features=use_image_features,
            feature_combination_mode=feature_combination_mode,
            device=device,
        )
    )
    # load train dataset
    train_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        data_split="train",
        subsample=train_data_subsample,
        split=dataset_split,
        use_image_features=use_image_features,
        feature_combination_mode=feature_combination_mode,
        device=device,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    describe_dataloader(train_dataloader, title="Train DataLoader Summary")
    train_dataloader = cycle(train_dataloader)

    # load eval dataset
    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        data_split="test",
        subsample=False,
        split=dataset_split,
        use_image_features=use_image_features,
        feature_combination_mode=feature_combination_mode,
        device=device,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size*2, shuffle=True)
    describe_dataloader(eval_dataloader, title="Eval DataLoader Summary")
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # load rq-vae tokenizer
    if use_image_features and feature_combination_mode == "concat":
        vae_input_dim = vae_input_dim * 2
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
        use_cross_attn=use_rqvae_cross_attn,
        attn_heads=attn_heads,
        # use_projection_head=False, 
        mixed_precision_type=mixed_precision_type,
    )
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    # load model
    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=train_dataset.max_seq_len * tokenizer.sem_ids_dim,
        jagged_mode=model_jagged_mode,
        rope=rope,
        prefix_matching=prefix_matching,
        enable_image_cross_attn=enable_image_cross_attn,
    )
    display_model_summary(model, device)

    # load optimizer and scheduler
    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer, warmup_steps=10000)

    start_iter = 0
    if pretrained_decoder_path is not None:
        logger.info(
            f"[Resume Training] Loading pretrained Decoder from {pretrained_decoder_path}.")
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iteration"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler)

    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])

    # starting the training
    logger.info(f"Training Started! - Debugging: {debug}")

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        for iter_ in range(start_iter, start_iter + 1 + iterations):
            train_iteration(model=model,
                            optimizer=optimizer,
                            tokenizer=tokenizer,
                            accelerator=accelerator,
                            lr_scheduler=lr_scheduler,
                            metrics_accumulator=metrics_accumulator,
                            train_dataloader=train_dataloader,
                            eval_dataloader=eval_dataloader,
                            gradient_accumulate_every=gradient_accumulate_every,
                            device=device,
                            pbar=pbar,
                            log_dir=log_dir,
                            iteration=iter_,
                            iterations=iterations,
                            save_model_every=save_model_every,
                            log_every=log_every,
                            eval_every=eval_every,
                            use_image_features=use_image_features,
                            wandb_logging=wandb_logging)
            pbar.update(1)

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    parse_config()
    train()
