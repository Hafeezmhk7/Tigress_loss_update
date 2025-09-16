"""
Encoder-Decoder model for retrieval tasks with Transformer architecture.
This model supports both jagged and padded input formats, allowing for flexible
handling of variable-length sequences. It includes mechanisms for embedding
semantic IDs, user IDs, and positional encodings, and is designed to be used
for both training and inference tasks, including generation of next semantic IDs.
"""

# imports
import gin
import torch

from einops import rearrange
from enum import Enum
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder
from modules.transformer.model import TransformerEncoderDecoder
from modules.utils import eval_mode
from modules.utils import maybe_repeat_interleave
from modules.utils import reset_encoder_cache
from modules.utils import reset_kv_cache
from modules.utils import select_columns_per_row
from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from typing import NamedTuple, Optional
from torch import nn
from torch import Tensor
from torch.nn import functional as F

# Needed to make torch.compile succeed
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class EncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        max_pos=2048,
        jagged_mode: bool = True,
        rope=False,
        prefix_matching=False,
        enable_image_cross_attn=False,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False

        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.norm_cxt = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim,
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        self.rope = rope
        self.prefix_matching = prefix_matching
        if not self.rope:
            self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
            self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)

        self.enable_image_cross_attn = enable_image_cross_attn
        
        self.transformer = (
            TransformerEncoderDecoder(
                d_in=attn_dim,
                d_out=attn_dim,
                dropout=dropout,
                num_heads=num_heads,
                encoder_layers=n_layers // 2,
                decoder_layers=n_layers // 2,
                rope=self.rope,
                enable_image_cross_attn=self.enable_image_cross_attn
            )
            if self.jagged_mode
            else nn.Transformer(
                d_model=attn_dim,
                nhead=num_heads,
                num_encoder_layers=n_layers // 2,
                num_decoder_layers=n_layers // 2,
                dim_feedforward=1024,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)

    def _predict(self, batch: TokenizedSeqBatch, image_emb: Optional[Tensor] = None) -> AttentionInput:
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        seq_lengths = batch.seq_mask.sum(axis=1)

        B, N, D = sem_ids_emb.shape

        # pos_max = N // self.sem_id_dim
        # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)

        # positional embedding
        if not self.rope:
            pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
            wpe = self.wpe(pos)
            input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        else:
            # using RoPE internally
            input_embedding = torch.cat([user_emb, sem_ids_emb], axis=1)
        input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
        if sem_ids_emb_fut is not None:
            if not self.rope:
                tte_fut = self.tte(batch.token_type_ids_fut)
                input_embedding_fut = torch.cat(
                    [input_embedding_fut, sem_ids_emb_fut + tte_fut], axis=1
                )
            else:
                input_embedding_fut = torch.cat([input_embedding_fut, sem_ids_emb_fut], axis=1)

        if self.jagged_mode:
            input_embedding = padded_to_jagged_tensor(
                input_embedding,
                lengths=seq_lengths + 1,
                max_len=input_embedding.shape[1],
            )

            seq_lengths_fut = torch.tensor(
                input_embedding_fut.shape[1],
                device=input_embedding_fut.device,
                dtype=torch.int64,
            ).repeat(B)
            input_embedding_fut = padded_to_jagged_tensor(
                input_embedding_fut,
                lengths=seq_lengths_fut,
                max_len=input_embedding_fut.shape[1],
            )
        else:
            mem_mask = torch.cat(
                [
                    torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
                    batch.seq_mask,
                ],
                axis=1,
            )
            f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
            f_mask[~mem_mask] = float("-inf")

        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        transformer_context_image = self.in_proj_context(self.do(self.norm(image_emb))) if self.enable_image_cross_attn and image_emb is not None else None
        transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))

        if self.jagged_mode:
            transformer_output = self.transformer(
                x=transformer_input,
                context=transformer_context,
                context_image=transformer_context_image,
                padding_mask=batch.seq_mask,
                jagged=self.jagged_mode,
            )
        else:
            transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                transformer_input.shape[1]
            )
            transformer_output = self.transformer(
                src=transformer_context,
                tgt=transformer_input,
                tgt_is_causal=True,
                tgt_mask=causal_mask,
                src_key_padding_mask=f_mask,
                memory_key_padding_mask=f_mask,
            )

        return transformer_output

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id(
        self, batch: TokenizedSeqBatch, temperature: int = 1, top_k: bool = True
    ) -> GenerationOutput:

        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 32 if top_k else 1
        n_top_k_candidates = 200 if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None,
            x_image=batch.x_image,
        )

        for i in range(self.sem_id_dim):
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            samples_batched = torch.multinomial(
                probas_batched, num_samples=n_top_k_candidates
            )

            if generated is None:
                # for the first token, check validity of each candidate token
                is_valid_prefix = self.inference_verifier_fn(
                    samples_batched.unsqueeze(-1)
                )
            else:
                # for subsequent tokens, check validity of each prefix formed by generated tokens + new candidate
                prefix = torch.cat(
                    [
                        generated.flatten(0, 1)
                        .unsqueeze(1)
                        .repeat_interleave(n_top_k_candidates, axis=1),
                        samples_batched.unsqueeze(-1),
                    ],
                    axis=-1,
                )
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)

            sampled_log_probas = torch.log(
                torch.gather(probas_batched, 1, samples_batched)
            ).reshape(B, -1)
            samples = samples_batched.reshape(B, -1)

            if not self.prefix_matching:
                # Get top-K:
                sorted_log_probas, sorted_indices = (
                    -10000 * (~is_valid_prefix)
                    + sampled_log_probas
                    + maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
                ).sort(-1, descending=True)

                top_k_log_probas, top_k_indices = (
                    sorted_log_probas[:, :k],
                    sorted_indices[:, :k],
                )
                top_k_samples = torch.gather(samples, 1, top_k_indices)
            else:
                # prefix matching & filtering for top-K valid candidates
                valid_mask = is_valid_prefix
                # apply mask: retain log probs for valid candidates only
                valid_log_probas = torch.where(
                    valid_mask,
                    sampled_log_probas + maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1),
                    sampled_log_probas.new_full(sampled_log_probas.shape, float('-inf'))
                )

                # sort and get top-K valid candidates
                sorted_log_probas, sorted_indices = valid_log_probas.sort(-1, descending=True)

                # ensure we have enough valid candidates, if not expand beam
                valid_counts = valid_mask.sum(dim=1)
                min_valid = valid_counts.min()

                if min_valid < k:
                    # expand beam size dynamically
                    k_actual = min(n_top_k_candidates, max(k, min_valid.item()))
                else:
                    k_actual = k

                top_k_log_probas = sorted_log_probas[:, :k_actual]
                top_k_indices = sorted_indices[:, :k_actual]
                top_k_samples = torch.gather(samples, 1, top_k_indices)

                # filter out any remaining invalid IDs (shouldn't happen but safety check)
                final_valid_mask = torch.gather(valid_mask, 1, top_k_indices)
                if not final_valid_mask.all():
                    # Fallback: keep only valid ones, pad if necessary
                    for b in range(B):
                        valid_idx = final_valid_mask[b].nonzero(as_tuple=True)[0]
                        if len(valid_idx) < k:
                            if len(valid_idx) == 0:
                                # fallback to first `k` candidates regardless of validity
                                valid_idx = torch.arange(k, device=top_k_samples.device)
                            else:
                                # pad with best valid candidate
                                pad_size = k - len(valid_idx)
                                valid_idx = torch.cat([valid_idx, valid_idx[:1].repeat(pad_size)])
                        
                        top_k_samples[b] = top_k_samples[b][valid_idx[:k]]
                        top_k_log_probas[b] = top_k_log_probas[b][valid_idx[:k]]

            if generated is not None:
                parent_id = torch.gather(
                    generated,
                    1,
                    (top_k_indices // n_top_k_candidates)
                    .unsqueeze(2)
                    .expand(-1, -1, i),
                )
                top_k_samples = torch.cat(
                    [parent_id, top_k_samples.unsqueeze(-1)], axis=-1
                )

                next_sem_ids = top_k_samples.flatten(end_dim=1)
                # determine how many children per original batch row we have (beam width used)
                children_per_example = top_k_samples.shape[1]  # k or k_actual

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.arange(
                        next_sem_ids.shape[1], device=next_sem_ids.device
                    ).repeat(next_sem_ids.shape[0], 1),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids,
                    x_image=(input_batch.x_image.repeat_interleave(children_per_example, dim=0)
                    if (input_batch.x_image is not None) else None),
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)

                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions
                # (E.g. Implement repeat_interleave jagged kernel)
                if self.jagged_mode:
                    cache = torch.zeros(
                        input_batch.sem_ids.shape[0],
                        input_batch.sem_ids.shape[1] + 1,
                        self.attn_dim,
                        device=input_batch.sem_ids.device,
                    )
                    cache_mask = torch.cat(
                        [
                            torch.ones(
                                input_batch.sem_ids.shape[0],
                                1,
                                dtype=bool,
                                device=input_batch.seq_mask.device,
                            ),
                            input_batch.seq_mask,
                        ],
                        axis=1,
                    )
                    cache[cache_mask] = self.transformer.cached_enc_output.values()
                    lengths = (
                        self.transformer.cached_enc_output.offsets()
                        .diff()
                        .repeat_interleave(k)
                    )
                    cache = cache.repeat_interleave(k, dim=0)
                    self.transformer.cached_enc_output = padded_to_jagged_tensor(
                        cache, lengths, max_len=cache.shape[1]
                    )

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.zeros_like(next_sem_ids),
                    seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=input_batch.token_type_ids.repeat_interleave(
                        k, dim=0
                    ),
                    x_image=(input_batch.x_image.repeat_interleave(k, dim=0)
                    if (input_batch.x_image is not None) else None),
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())

        return GenerationOutput(
            sem_ids=generated.squeeze(), log_probas=log_probas.squeeze()
        )

    @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        # pass image emb through if present
        trnsf_out = self._predict(batch, image_emb=batch.x_image)

        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits = rearrange(
                    jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B
                )[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(
                    F.cross_entropy(logits, target, reduction="none", ignore_index=-1),
                    "(b n) -> b n",
                    b=B,
                )
                loss = unred_loss.sum(axis=1).mean()
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                loss = (
                    rearrange(
                        F.cross_entropy(out, target, reduction="none", ignore_index=-1),
                        "(b n) -> b n",
                        b=B,
                    )
                    .sum(axis=1)
                    .mean()
                )
            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None
            loss_d = unred_loss.mean(axis=0)
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(
                jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B
            )[:, -1, :]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None
        else:
            trnsf_out_flattened = trnsf_out[:, -1, :]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None

        return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)
