from typing import Optional, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.config import AttentionMode

# -----------------------------------------------------------------------------
# Pluggable Multi-Head Attention
# -----------------------------------------------------------------------------

class PluggableMHA(nn.Module):
    """
    Multi-Head module with multiple attention mechanisms.

    Modes:
      - 'vanilla'       : logits = QK^T / sqrt(d)
      - 'synth_dense'   : logits = MLP(Q)  (shape B,H,Tq,Lk)
      - 'synth_random'  : logits = Param[H,Lq,Lk] (learnable or fixed)
      - 'hybrid_dense'  : logits = (1-g)*vanilla + g*synth_dense
      - 'hybrid_random' : logits = (1-g)*vanilla + g*synth_random
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        mode: AttentionMode,
        max_q_len: int,
        max_k_len: int,
        synth_hidden: int = 0,
        synth_fixed_random: bool = False,
        gate_init: float = 0.5,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.mode: AttentionMode = mode
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
        self.max_q_len = max_q_len
        self.max_k_len = max_k_len

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Synthesizer dense head(s): produce logits over keys from Q
        if mode in ('synth_dense', 'hybrid_dense'):
            if synth_hidden and synth_hidden > 0:
                self.synth = nn.Sequential(
                    nn.Linear(d_model, synth_hidden),
                    nn.GELU(),
                    nn.Linear(synth_hidden, n_heads * self.max_k_len),
                )
            else:
                self.synth = nn.Linear(d_model, n_heads * self.max_k_len)

        # Synthesizer random: per-head [Lq, Lk]
        if mode in ('synth_random', 'hybrid_random'):
            rand = torch.randn(n_heads, self.max_q_len, self.max_k_len) / math.sqrt(self.max_k_len)
            self.rand_logits = nn.Parameter(rand, requires_grad=not synth_fixed_random)
            if synth_fixed_random:
                self.rand_logits.requires_grad_(False)

        # Hybrid gates (per head), gate is weight on synthesizer branch
        if mode in ('hybrid_dense', 'hybrid_random'):
            g = math.log(gate_init / (1 - gate_init))  # inverse sigmoid init
            self.gate = nn.Parameter(torch.full((n_heads,), g))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) → (B, H, T, d_head)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _apply_masks(self, logits: torch.Tensor, key_padding_mask: Optional[torch.Tensor], causal: bool) -> torch.Tensor:
        # logits: (B, H, Tq, Tk)
        B, H, Tq, Tk = logits.shape
        if key_padding_mask is not None:
            # key_padding_mask: (B, Tk) with 1 for keep, 0 for pad
            if key_padding_mask.dim() != 2 or key_padding_mask.size(0) != B or key_padding_mask.size(1) != Tk:
                raise ValueError(f"key_padding_mask shape {tuple(key_padding_mask.shape)} does not match (B={B}, Tk={Tk})")
            mask = (1.0 - key_padding_mask.float()).view(B, 1, 1, Tk)  # 1 for pad
            logits = logits.masked_fill(mask.bool(), float('-inf'))
        if causal:
            causal_mask = torch.triu(torch.ones(Tq, Tk, device=logits.device, dtype=torch.bool), diagonal=1)
            logits = logits.masked_fill(causal_mask, float('-inf'))
        return logits

    def forward(
        self,
        x_q: torch.Tensor,                 # (B, Tq, d_model)
        x_kv: torch.Tensor,                # (B, Tk, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Tk) 1=keep,0=pad
        causal: bool = False,
    ) -> torch.Tensor:
        B, Tq, _ = x_q.shape
        Tk = x_kv.shape[1]

        # Project Q,K,V
        Q = self._shape(self.W_q(x_q))   # (B,H,Tq,Dh)
        K = self._shape(self.W_k(x_kv))  # (B,H,Tk,Dh)
        V = self._shape(self.W_v(x_kv))  # (B,H,Tk,Dh)

        # Vanilla logits
        vanilla_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,H,Tq,Tk)

        # Synthesizer logits
        synth_logits = None
        if self.mode in ('synth_dense', 'hybrid_dense'):
            # MLP on Q to produce per-head logits over keys (length = max_k_len)
            S = self.synth(x_q)  # (B,Tq,H*max_k_len)
            S = S.view(B, Tq, self.n_heads, self.max_k_len).permute(0, 2, 1, 3)  # (B,H,Tq,max_k_len)
            synth_logits = S[:, :, :, :Tk]
        elif self.mode in ('synth_random', 'hybrid_random'):
            synth_logits = self.rand_logits[:, :Tq, :Tk]  # (H,Tq,Tk)
            synth_logits = synth_logits.unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,Tq,Tk)

        # Combine per mode
        if self.mode == 'vanilla':
            logits = vanilla_logits
        elif self.mode == 'synth_dense' or self.mode == 'synth_random':
            logits = synth_logits  # type: ignore[assignment]
        else:  # hybrids
            gate = torch.sigmoid(self.gate).view(1, self.n_heads, 1, 1)
            logits = (1.0 - gate) * vanilla_logits + gate * synth_logits  # type: ignore[operator]

        logits = self._apply_masks(logits, key_padding_mask, causal)
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, V)  # (B,H,Tq,Dh)
        out = ctx.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.W_o(self.dropout(out))
