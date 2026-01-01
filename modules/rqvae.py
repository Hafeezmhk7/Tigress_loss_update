import torch
import logging
from data.schemas import SeqBatch
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.loss import QuantizeLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from typing import NamedTuple, Optional
from torch import nn
from torch import Tensor
from modules.transformer.attention import MultiHeadAttention

# ===== TRIPLE-STAGE CONTRASTIVE LEARNING =====
from modules.loss import EncoderInfoNCELoss, MultiScaleInfoNCELoss, CoSTInfoNCELoss
# ===== END =====

# fetch logger
logger = logging.getLogger("recsys_logger")
torch.set_float32_matmul_precision("high")


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor
    # ===== TRIPLE-STAGE CONTRASTIVE LEARNING =====
    encoder_output: Optional[Tensor] = None
    quantized_per_level: Optional[List[Tensor]] = None
    residuals_per_level: Optional[List[Tensor]] = None
    reconstructed: Optional[Tensor] = None
    # ===== END =====


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor
    # ===== TRIPLE-STAGE CONTRASTIVE LEARNING =====
    encoder_infonce_loss: Tensor = torch.tensor(0.0)
    multiscale_infonce_loss: Tensor = torch.tensor(0.0)
    cost_infonce_loss: Tensor = torch.tensor(0.0)
    # ===== END =====


class RqVae(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 18,
        # ===== TRIPLE-STAGE CONTRASTIVE LEARNING =====
        use_encoder_infonce: bool = False,
        use_multiscale_infonce: bool = False,
        use_cost_infonce: bool = False,
        infonce_temperature: float = 0.07,
        encoder_infonce_weight: float = 0.1,
        multiscale_infonce_weight: float = 0.3,
        multiscale_level_weights: List[float] = None,
        cost_infonce_weight: float = 0.2,
        encoder_dropout_rate: float = 0.1,
        # ===== END =====
        # ===== RECONSTRUCTION LOSS =====
        use_reconstruction_loss: bool = True,
        reconstruction_weight: float = 1.0,
        # ===== END =====
        # ===== CROSS-ATTENTION =====
        use_cross_attn: bool = False,
        attn_heads: int = 8,
        mixed_precision_type: str = "fp16",
        # ===== END =====
    ) -> None:
        self._config = locals()

        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features

        # ===== TRIPLE-STAGE CONTRASTIVE LEARNING =====
        self.use_encoder_infonce = use_encoder_infonce
        self.use_multiscale_infonce = use_multiscale_infonce
        self.use_cost_infonce = use_cost_infonce
        self.encoder_infonce_weight = encoder_infonce_weight
        self.multiscale_infonce_weight = multiscale_infonce_weight
        self.cost_infonce_weight = cost_infonce_weight
        # ===== END =====
        
        # ===== RECONSTRUCTION LOSS =====
        self.use_reconstruction_loss = use_reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        # ===== END =====

        self.layers = nn.ModuleList(
            modules=[
                Quantize(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    forward_mode=codebook_mode,
                    do_kmeans_init=codebook_kmeans_init,
                    codebook_normalize=i == 0 and codebook_normalize,
                    sim_vq=codebook_sim_vq,
                    commitment_weight=commitment_weight,
                )
                for i in range(n_layers)
            ]
        )

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=True,
        )

        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features)
            if n_cat_features != 0
            else ReconstructionLoss()
        )

        # ===== CROSS-ATTENTION =====
        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            self.img_proj = nn.Linear(input_dim, embed_dim)
            self.cross_attn = MultiHeadAttention(
                d_in=embed_dim,
                d_out=embed_dim,
                num_heads=attn_heads,
                cross_attn=True,
                dropout=0.1,
            )
        # ===== END =====
        
        # ===== TRIPLE-STAGE CONTRASTIVE LEARNING: Initialize losses =====
        if self.use_encoder_infonce:
            self.encoder_infonce = EncoderInfoNCELoss(
                temperature=infonce_temperature,
                dropout_rate=encoder_dropout_rate
            )
            logger.info(f"✓ Initialized Encoder InfoNCE Loss")
        
        if self.use_multiscale_infonce:
            self.multiscale_infonce = MultiScaleInfoNCELoss(
                n_levels=n_layers,
                temperature=infonce_temperature,
                level_weights=multiscale_level_weights,
            )
            logger.info(f"✓ Initialized Multi-Scale InfoNCE Loss")
        
        if self.use_cost_infonce:
            self.cost_infonce = CoSTInfoNCELoss(
                temperature=infonce_temperature
            )
            logger.info(f"✓ Initialized CoST InfoNCE Loss")
        # ===== END =====

    @cached_property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        logger.info(f"Loaded RQ-VAE Model | Iteration {state['iteration']}")

    def encode(self, x: Tensor, x_cross: Optional[Tensor] = None) -> Tensor:
        h = self.encoder(x)
        if self.use_cross_attn and x_cross is not None:
            x_cross = x_cross.to(h.device, dtype=h.dtype)
            x_cross_proj = self.img_proj(x_cross)
            h = h + self.cross_attn(x=h, x_kv=x_cross_proj, jagged=True).squeeze(1)
        return h

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_semantic_ids(
        self, 
        x: Tensor, 
        x_image: Optional[Tensor] = None,
        gumbel_t: float = 0.001
    ) -> RqVaeOutput:
        # Encode input
        res = self.encode(x, x_cross=x_image)
        
        # ===== TRIPLE-STAGE CONTRASTIVE: Store encoder output for losses =====
        encoder_output = res.clone() if (self.use_encoder_infonce or self.use_cost_infonce) else None
        # ===== END =====

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []
        
        # ===== TRIPLE-STAGE CONTRASTIVE: Track per-level for multi-scale InfoNCE =====
        quantized_per_level = [] if self.use_multiscale_infonce else None
        residuals_per_level = [] if self.use_multiscale_infonce else None
        # ===== END =====

        for level_idx, layer in enumerate(self.layers):
            # Store residual
            residuals.append(res)
            if self.use_multiscale_infonce:
                residuals_per_level.append(res.clone())
            
            # Quantize
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss += quantized.loss
            emb_quantized, id = quantized.embeddings, quantized.ids
            
            # ===== TRIPLE-STAGE CONTRASTIVE: Track quantized for multi-scale InfoNCE =====
            if self.use_multiscale_infonce:
                quantized_per_level.append(emb_quantized)
            # ===== END =====
            
            # Update residual
            res = res - emb_quantized.detach()
            
            sem_ids.append(id)
            embs.append(emb_quantized)

        # ===== TRIPLE-STAGE CONTRASTIVE: Compute reconstruction for CoST =====
        reconstructed = None
        if self.use_cost_infonce:
            reconstructed = torch.stack(embs, dim=0).sum(axis=0)
        # ===== END =====

        return RqVaeOutput(
            embeddings=torch.stack(embs, dim=0),
            residuals=torch.stack(residuals, dim=0),
            sem_ids=torch.stack(sem_ids, dim=1),
            quantize_loss=quantize_loss,
            encoder_output=encoder_output,
            quantized_per_level=quantized_per_level,
            residuals_per_level=residuals_per_level,
            reconstructed=reconstructed,
        )

    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: SeqBatch, gumbel_t: float) -> RqVaeComputedLosses:
        x, x_image = batch.x, batch.x_image
        quantized = self.get_semantic_ids(x, x_image=x_image, gumbel_t=gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        
        # Decode using embeddings
        x_hat = self.decode(embs.sum(axis=0))
        
        x_hat = torch.cat(
            [l2norm(x_hat[..., : -self.n_cat_feats]), x_hat[..., -self.n_cat_feats :]],
            axis=-1,
        )

        # Compute reconstruction loss
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        
        # Quantization loss
        rqvae_loss = quantized.quantize_loss
        
        # ===== TRIPLE-STAGE CONTRASTIVE: Compute all three InfoNCE losses =====
        encoder_infonce_loss = torch.tensor(0.0, device=x.device)
        multiscale_infonce_loss = torch.tensor(0.0, device=x.device)
        cost_infonce_loss = torch.tensor(0.0, device=x.device)
        
        if self.use_encoder_infonce and quantized.encoder_output is not None:
            encoder_infonce_loss = self.encoder_infonce(quantized.encoder_output)
        
        if self.use_multiscale_infonce and quantized.quantized_per_level is not None:
            multiscale_infonce_loss = self.multiscale_infonce(
                quantized_list=quantized.quantized_per_level,
                residual_list=quantized.residuals_per_level
            )
        
        if self.use_cost_infonce and quantized.reconstructed is not None:
            cost_infonce_loss = self.cost_infonce(
                reconstructed=quantized.reconstructed,
                original=quantized.encoder_output
            )
        # ===== END =====
        
        # Total loss with configurable reconstruction weight
        loss = (
            (self.reconstruction_weight * reconstruction_loss if self.use_reconstruction_loss else 0) +
            rqvae_loss + 
            self.encoder_infonce_weight * encoder_infonce_loss +
            self.multiscale_infonce_weight * multiscale_infonce_loss +
            self.cost_infonce_weight * cost_infonce_loss
        ).mean()

        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(quantized.sem_ids, "b d -> b 1 d")
                        == rearrange(quantized.sem_ids, "b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            # ===== TRIPLE-STAGE CONTRASTIVE: Return all InfoNCE losses =====
            encoder_infonce_loss=encoder_infonce_loss,
            multiscale_infonce_loss=multiscale_infonce_loss,
            cost_infonce_loss=cost_infonce_loss,
            # ===== END =====
        )