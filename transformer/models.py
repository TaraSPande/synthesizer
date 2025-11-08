from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.layers import LearnedPositionalEmbedding, Encoder, Decoder
from transformer.config import TransformerConfig

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class EncoderDecoderTransformer(nn.Module):
    """Full seq2seq Transformer compatible with Trainer seq2seq signature."""
    def __init__(self, cfg: TransformerConfig, pad_token_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = pad_token_id

        self.src_tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_token_id)
        self.tgt_tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_token_id)
        self.src_pos = LearnedPositionalEmbedding(cfg.max_src_len, cfg.d_model)
        self.tgt_pos = LearnedPositionalEmbedding(cfg.max_tgt_len, cfg.d_model)
        # Store max positions for clamping
        self._max_src_pos = cfg.max_src_len
        self._max_tgt_pos = cfg.max_tgt_len

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_tgt_embeddings:
            self.lm_head.weight = self.tgt_tok.weight

        self.drop = nn.Dropout(cfg.dropout)
        self.ln_f_enc = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ln_f_dec = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def _positions(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # Default encoder positions
        return self._enc_positions(x, offset)
        
    def _enc_positions(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(offset, offset + T, device=x.device).unsqueeze(0).expand(B, T)
        # Clamp to encoder max positions
        pos = torch.clamp(pos, max=self._max_src_pos - 1)
        return pos
        
    def _dec_positions(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(offset, offset + T, device=x.device).unsqueeze(0).expand(B, T)
        # Clamp to decoder max positions
        pos = torch.clamp(pos, max=self._max_tgt_pos - 1)
        return pos

    def _kpm(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # prefer provided attention_mask (1=keep,0=pad), else infer from pad id; normalize to {0,1} long
        if attn_mask is not None:
            return (attn_mask > 0).long()
        return (input_ids != self.pad_token_id).long()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        # Encoder
        enc_pos = self._enc_positions(input_ids)
        enc_emb = self.src_tok(input_ids) + self.src_pos(enc_pos)
        enc_emb = self.drop(enc_emb)
        enc_kpm = self._kpm(input_ids, attention_mask)
        enc_out = self.encoder(enc_emb, enc_kpm)
        enc_out = self.ln_f_enc(enc_out)
        # Runtime sanity checks to ensure encoder–decoder connection remains valid
        with torch.no_grad():
            assert enc_out.dim() == 3, "enc_out must be (B, Tk, d)"
            assert enc_kpm is not None, "encoder_attention_mask must be provided"
            assert enc_kpm.shape[0] == enc_out.shape[0] and enc_kpm.shape[1] == enc_out.shape[1], "enc_kpm shape must match enc_out (B, Tk)"
            m = enc_out.abs().mean()
            if not torch.isfinite(m):
                raise ValueError("Non-finite encoder output")
            if m <= 0:
                raise ValueError("Encoder output near-zero; possible dead encoder")

        # Decoder
        assert decoder_input_ids is not None, "decoder_input_ids required for seq2seq"
        dec_pos = self._dec_positions(decoder_input_ids)
        dec_emb = self.tgt_tok(decoder_input_ids) + self.tgt_pos(dec_pos)
        dec_emb = self.drop(dec_emb)
        dec_kpm = self._kpm(decoder_input_ids, decoder_attention_mask)
        dec_out = self.decoder(dec_emb, enc_out, dec_kpm, enc_kpm)
        dec_out = self.ln_f_dec(dec_out)

        logits = self.lm_head(dec_out)
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
            )
            out["loss"] = loss
        return out


class DecoderOnlyTransformer(nn.Module):
    """Causal LM using only decoder stack with causal self-attn."""
    def __init__(self, cfg: TransformerConfig, pad_token_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = pad_token_id

        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_token_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_tgt_len, cfg.d_model)
        # Store max position for clamping
        self._max_pos = cfg.max_tgt_len
        # reuse Decoder with self-attn config from cfg.attn_mode_self_dec
        self.decoder = Decoder(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_tgt_embeddings:
            self.lm_head.weight = self.tok.weight
        self.drop = nn.Dropout(cfg.dropout)
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def _positions(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        # Clamp positions to prevent out-of-bounds access to positional embeddings
        max_pos = getattr(self, '_max_pos', 256)  # Default fallback  
        pos = torch.clamp(pos, max=max_pos - 1)
        return pos

    def _kpm(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attn_mask is not None:
            return (attn_mask > 0).long()
        return (input_ids != self.pad_token_id).long()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        pos = self._positions(input_ids)
        x = self.tok(input_ids) + self.pos(pos)
        x = self.drop(x)
        kpm = self._kpm(input_ids, attention_mask)
        # Use decoder with empty encoder memory: implement via self-attn only path
        # We can pass enc=0 and enc_kpm=None; DecoderLayer uses enc in cross-attn, so provide zeros
        enc_dummy = torch.zeros_like(x)
        x = self.decoder(x, enc_dummy, kpm, None)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            out["loss"] = loss
        return out


class EncoderClassifier(nn.Module):
    """Encoder-only classifier (GLUE/SuperGLUE) using mean pooling."""
    def __init__(self, cfg: TransformerConfig, num_labels: int, pad_token_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = pad_token_id

        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_token_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_src_len, cfg.d_model)
        # Store max position for clamping
        self._max_pos = cfg.max_src_len
        # Build encoder-only stack: reuse Encoder
        self.encoder = Encoder(cfg)
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, num_labels)

    def _positions(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        # Clamp positions to prevent out-of-bounds access to positional embeddings
        max_pos = getattr(self, '_max_pos', 256)  # Default fallback
        pos = torch.clamp(pos, max=max_pos - 1)
        return pos

    def _kpm(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attn_mask is not None:
            return attn_mask
        return (input_ids != self.pad_token_id).long()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        pos = self._positions(input_ids)
        x = self.tok(input_ids) + self.pos(pos)
        x = self.drop(x)
        kpm = self._kpm(input_ids, attention_mask)
        x = self.encoder(x, kpm)
        x = self.ln_f(x)
        # mean-pool over non-pad tokens
        mask = kpm.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(pooled)
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out["loss"] = loss
        return out
