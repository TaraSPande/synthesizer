from typing import Optional, List

import torch
import torch.nn as nn

from transformer.attention import PluggableMHA
from transformer.config import TransformerConfig

# -----------------------------------------------------------------------------
# Helper layers
# -----------------------------------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.weight = nn.Embedding(max_len, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, T) with position indices
        return self.weight(positions)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------------------------------------------------------
# Transformer blocks
# -----------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.self_attn = PluggableMHA(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            mode=cfg.attn_mode_self_enc,
            max_q_len=cfg.max_src_len,
            max_k_len=cfg.max_src_len,
            synth_hidden=cfg.synth_hidden,
            synth_fixed_random=cfg.synth_fixed_random,
            gate_init=cfg.gate_init,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Pre-LN
        y = self.self_attn(self.ln1(x), x, key_padding_mask=key_padding_mask, causal=False)
        x = x + self.drop(y)
        y = self.ff(self.ln2(x))
        x = x + self.drop(y)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.self_attn = PluggableMHA(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            mode=cfg.attn_mode_self_dec,
            max_q_len=cfg.max_tgt_len,
            max_k_len=cfg.max_tgt_len,
            synth_hidden=cfg.synth_hidden,
            synth_fixed_random=cfg.synth_fixed_random,
            gate_init=cfg.gate_init,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        cross_mode = cfg.attn_mode_cross
        self.cross_attn = PluggableMHA(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            mode=cross_mode,
            max_q_len=cfg.max_tgt_len,
            max_k_len=cfg.max_src_len,
            synth_hidden=cfg.synth_hidden,
            synth_fixed_random=cfg.synth_fixed_random,
            gate_init=cfg.gate_init,
        )
        self.ln3 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        self_kpm: Optional[torch.Tensor],
        enc_kpm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        y = self.self_attn(self.ln1(x), x, key_padding_mask=self_kpm, causal=True)
        x = x + self.drop(y)
        # Cross-attention: allow decoder-only mode (skip when no encoder memory)
        if enc is None or enc_kpm is None:
            y = 0.0 * x
        else:
            assert enc.dim() == 3, "encoder hidden states must be (B, Tk, d)"
            assert enc_kpm.dim() == 2, "encoder_attention_mask must be (B, Tk)"
            assert enc.size(0) == x.size(0), "batch size mismatch between decoder and encoder"
            assert enc.size(1) == enc_kpm.size(1), "enc length must match enc_kpm length"
            y = self.cross_attn(self.ln2(x), enc, key_padding_mask=enc_kpm, causal=False)
        x = x + self.drop(y)
        y = self.ff(self.ln3(x))
        x = x + self.drop(y)
        return x
    

# -----------------------------------------------------------------------------
# Stacks
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers_enc)])

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers_dec)])

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        self_kpm: Optional[torch.Tensor],
        enc_kpm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc, self_kpm, enc_kpm)
        return x
