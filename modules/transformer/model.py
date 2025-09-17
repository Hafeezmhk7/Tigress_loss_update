from modules.encoder import MLP
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.attention import MultiHeadAttention
from typing import List
from typing import Optional
from torch import nn
from torch import Tensor


class KVCacheOpsMixin:
    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            layer.reset_kv_cache()

    def apply_to_kv_cache(self, fn) -> None:
        for layer in self.layers:
            layer.apply_to_kv_cache(fn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        mlp_hidden_dims: List[int] = [1024],
        do_cross_attn: bool = False,
        enable_kv_cache: bool = True,
        rope: bool = False,
        enable_image_cross_attn: bool = False,
    ) -> None:
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.do_cross_attn = do_cross_attn
        self.enable_kv_cache = enable_kv_cache
        self.enable_image_cross_attn = enable_image_cross_attn

        self.attention = MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            num_heads=num_heads,
            cross_attn=False,
            dropout=dropout,
            qkv_bias=qkv_bias,
            enable_kv_cache=enable_kv_cache,
            rope=rope,
        )

        self.ff = nn.Sequential(
            RMSNorm(d_out),
            MLP(
                input_dim=d_out,
                hidden_dims=mlp_hidden_dims,
                out_dim=d_out,
                dropout=dropout,
                normalize=False,
            ),
            nn.Dropout(dropout),
        )

        self.attn_norm = RMSNorm(d_out)
        self.ffn_norm = RMSNorm(d_out)
        self.do = nn.Dropout(dropout)

        if self.do_cross_attn:
            self.cross_attention_text = MultiHeadAttention(
                d_in=d_out,
                d_out=d_out,
                num_heads=num_heads,
                cross_attn=True,
                dropout=dropout,
                qkv_bias=qkv_bias,
                rope=rope,
            )
            if self.enable_image_cross_attn:
                self.cross_attention_image = MultiHeadAttention(
                    d_in=d_out,
                    d_out=d_out,
                    num_heads=num_heads,
                    cross_attn=True,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                    rope=rope,
                )
            self.cross_attn_norm = RMSNorm(d_out)

    def forward(
        self,
        x: AttentionInput,
        x_kv: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        jagged: Optional[bool] = False,
        image_context: Optional[Tensor] = None,
    ) -> AttentionInput:
        attn_out = x + self.attention(
            self.do(self.attn_norm(x)),
            padding_mask=padding_mask,
            is_causal=is_causal,
            jagged=jagged,
            use_cache=not self.training and self.enable_kv_cache,
        )
        if self.do_cross_attn:
            attn_out_text = self.cross_attention_text(
                x=self.do(self.cross_attn_norm(x)),
                x_kv=x_kv,
                padding_mask=padding_mask,
                is_causal=False,
                jagged=jagged,
                use_cache=not self.training and self.enable_kv_cache,
            )

            attn_out_image = 0
            if self.enable_image_cross_attn and image_context is not None:
                attn_out_image = self.cross_attention_image(
                    x=self.do(self.cross_attn_norm(x)),
                    x_kv=image_context,
                    padding_mask=padding_mask,
                    is_causal=False,
                    jagged=jagged,
                    use_cache=not self.training and self.enable_kv_cache,
                )

            attn_out = attn_out + attn_out_text + attn_out_image

        proj_out = attn_out + self.ff(attn_out)
        return proj_out

    def reset_kv_cache(self):
        self.attention.kv_cache.reset()
        if self.do_cross_attn:
            self.cross_attention.kv_cache.reset()

    def apply_to_kv_cache(self, fn):
        self.attention.kv_cache.apply(fn)
        if self.do_cross_attn:
            self.cross_attention.kv_cache.apply(fn)


class TransformerDecoder(nn.Module, KVCacheOpsMixin):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        n_layers: int,
        do_cross_attn: bool = False,
        enable_kv_cache: bool = True,
        rope: bool = False,
        enable_image_cross_attn: bool = False,
    ) -> None:
        super().__init__()

        self.do_cross_attn = do_cross_attn

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_in=d_in,
                    d_out=d_out,
                    dropout=dropout,
                    num_heads=num_heads,
                    qkv_bias=False,
                    do_cross_attn=self.do_cross_attn,
                    enable_kv_cache=enable_kv_cache,
                    rope=rope,
                    enable_image_cross_attn=enable_image_cross_attn,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: AttentionInput,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        context: Optional[Tensor] = None,
        image_context: Optional[Tensor] = None,
        jagged: Optional[bool] = None,
    ) -> AttentionInput:
        for layer in self.layers:
            x = layer(
                x=x,
                x_kv=context,
                padding_mask=padding_mask,
                is_causal=is_causal,
                jagged=jagged,
                image_context=image_context,
            )
        return x

    @property
    def seq_lengths(self) -> Tensor:
        return self.layers[0].attention.kv_cache.seq_lengths


class TransformerEncoderDecoder(nn.Module, KVCacheOpsMixin):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        rope: bool = False,
        enable_image_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        
        self.encoder = TransformerDecoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=encoder_layers,
            do_cross_attn=False,
            enable_kv_cache=False,
            rope=rope,
        )

        self.decoder = TransformerDecoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            n_layers=decoder_layers,
            do_cross_attn=True,
            enable_kv_cache=False,
            rope=rope,
            enable_image_cross_attn=enable_image_cross_attn,
        )

        self.layers = [self.encoder, self.decoder]
        self.cached_enc_output = None

    def forward(
        self,
        x: AttentionInput,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        image_context: Optional[Tensor] = None,
        jagged: Optional[bool] = None,
    ) -> AttentionInput:
        if self.cached_enc_output is None:
            context_memory = self.encoder(
                context,
                padding_mask=padding_mask,
                is_causal=False,
                context=None,
                jagged=jagged,
            )
            # just pass image embeddings directly
            image_memory = image_context
            if not self.training:
                self.cached_enc_output = context_memory
        else:
            context_memory = self.cached_enc_output
            image_memory = image_context

        out = self.decoder(
            x,
            padding_mask=None,
            is_causal=True,
            context=context_memory,
            image_context=image_memory,
            jagged=jagged,
        )
        return out
