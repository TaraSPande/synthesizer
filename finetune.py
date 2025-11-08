import argparse
import dataclasses
import json
import os
import re
from datetime import datetime
from typing import Optional, Tuple

import torch
from datasets import load_dataset

from preprocess import (
    TaskType,
    DataConfig,
    DATA_REGISTRY,
    TokenizerConfig,
    build_tokenizer,
    build_seq2seq_spm_tokenizer,
)
from trainer import Trainer, TrainConfig
from transformer.models import (
    build_seq2seq_model,
    build_decoder_lm,
    build_classifier,
)
from utils import ensure_special_tokens


def parse_arch_from_run_name(run_dir_name: str) -> Tuple[int, int, int, int, str, str, str]:
    """
    Attempt to parse architecture hyperparameters and attention modes from a run directory name.

    Expected pattern (examples):
      - cnn_dailymail-enc6dec6-d512h8-vanilla.vanilla.vanilla_20251104_083232
      - lm1b-enc6dec6-d512h8-jto.jto.jto_20251024_163411
      - agnews-enc6dec6-d512h8-vanilla.vanilla.vanilla_20251107_213008

    Returns: (n_layers_enc, n_layers_dec, d_model, n_heads, self_enc_mode, self_dec_mode, cross_mode)
    Falls back to reasonable defaults if parsing fails.
    """
    base = os.path.basename(run_dir_name.rstrip("/"))
    # Defaults if parse fails
    n_layers_enc, n_layers_dec, d_model, n_heads = 6, 6, 512, 8
    self_enc_mode, self_dec_mode, cross_mode = "vanilla", "vanilla", "vanilla"

    m = re.search(r"enc(\d+)dec(\d+)", base)
    if m:
        n_layers_enc = int(m.group(1))
        n_layers_dec = int(m.group(2))
    m = re.search(r"d(\d+)h(\d+)", base)
    if m:
        d_model = int(m.group(1))
        n_heads = int(m.group(2))
    # modes appear as ...-<a>.<b>.<c>[_-]timestamp
    m = re.search(r"-([A-Za-z_]+)\.([A-Za-z_]+)\.([A-Za-z_]+)", base)
    if m:
        self_enc_mode, self_dec_mode, cross_mode = m.group(1), m.group(2), m.group(3)

    return n_layers_enc, n_layers_dec, d_model, n_heads, self_enc_mode, self_dec_mode, cross_mode


def load_ckpt_tokenizer_path(ckpt_dir: str) -> Optional[str]:
    """
    Prefer to load tokenizer from the checkpoint directory (most robust).
    If not available, attempt to read run_config.json's 'tokenizer' field.
    """
    # Check for common tokenizer artifacts in the checkpoint dir
    tok_candidates = ["tokenizer.json", "vocab.json", "merges.txt", "spiece.model", "special_tokens_map.json"]
    if any(os.path.isfile(os.path.join(ckpt_dir, f)) for f in tok_candidates):
        return ckpt_dir

    cfg_path = os.path.join(ckpt_dir, "run_config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            tok_path = cfg.get("tokenizer")
            # Normalize possible backslashes stored on Windows
            if isinstance(tok_path, str):
                tok_path = tok_path.replace("\\", os.sep)
                if os.path.isdir(tok_path):
                    return tok_path
        except Exception:
            pass
    return None


def maybe_freeze(model: torch.nn.Module, freeze_encoder: bool, freeze_decoder: bool, freeze_embeddings: bool) -> None:
    # Freeze encoder block + related embeddings/ln for seq2seq/classifier
    if freeze_encoder:
        for name, p in model.named_parameters():
            if any(k in name for k in ["encoder", "src_tok", "src_pos", "ln_f_enc"]):
                p.requires_grad = False
    # Freeze decoder block + related embeddings/ln/head for seq2seq/decoder-only
    if freeze_decoder:
        for name, p in model.named_parameters():
            if any(k in name for k in ["decoder", "tgt_tok", "pos", "tgt_pos", "ln_f_dec", "ln_f", "lm_head"]):
                p.requires_grad = False
    # Freeze all token embeddings
    if freeze_embeddings:
        for name, p in model.named_parameters():
            if name.endswith(".weight") and any(k in name for k in ["tok.weight", "tok.weight", "src_tok.weight", "tgt_tok.weight"]):
                p.requires_grad = False


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an existing checkpoint on a new task/dataset.")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint directory (contains pytorch_model.bin)")
    parser.add_argument("--data_key", required=True, type=str, help=f"Target dataset key. One of: {list(DATA_REGISTRY.keys())}")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save the fine-tuned run. Default auto-generated.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer name/path. Defaults to loading tokenizer from checkpoint.")
    parser.add_argument("--spm32k", action="store_true", help="For seq2seq: train/use a shared SentencePiece tokenizer on the new dataset (32000).")

    # Training overrides
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every_epoch", action="store_true", default=True)

    # Model length overrides (affect tokenizer max lengths too)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=128)

    # Optional freezing
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--freeze_embeddings", action="store_true")

    # Optional overrides for attention modes and layers if parsing fails or you want to change them
    parser.add_argument("--n_layers_enc", type=int, default=None)
    parser.add_argument("--n_layers_dec", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--attn_self_enc", type=str, default=None, choices=["vanilla", "synth_dense", "synth_random", "hybrid_dense", "hybrid_random", "jto"])
    parser.add_argument("--attn_self_dec", type=str, default=None, choices=["vanilla", "synth_dense", "synth_random", "hybrid_dense", "hybrid_random", "jto"])
    parser.add_argument("--attn_cross", type=str, default=None, choices=["vanilla", "synth_dense", "synth_random", "hybrid_dense", "hybrid_random", "jto"])
    parser.add_argument("--synth_hidden", type=int, default=0)
    parser.add_argument("--synth_fixed_random", action="store_true")
    parser.add_argument("--gate_init", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    ckpt_dir = args.ckpt
    if not os.path.isdir(ckpt_dir) or not os.path.isfile(os.path.join(ckpt_dir, "pytorch_model.bin")):
        raise FileNotFoundError(f"Checkpoint not found or invalid: {ckpt_dir}")

    # New dataset config
    if args.data_key not in DATA_REGISTRY:
        raise KeyError(f"Unknown data_key: {args.data_key}. Available: {list(DATA_REGISTRY.keys())}")
    data_cfg: DataConfig = DATA_REGISTRY[args.data_key]
    # Override lengths from CLI
    data_cfg = dataclasses.replace(data_cfg, max_length=args.max_len, max_target_length=args.max_tgt_len)

    # Build tokenizer: default to tokenizer saved in checkpoint; allow override
    tok_path = args.tokenizer or load_ckpt_tokenizer_path(ckpt_dir)
    if data_cfg.task_type == TaskType.SEQ2SEQ and args.spm32k:
        # Note: using a new SPM tokenizer will change vocab; embeddings from ckpt won't match and will be re-initialized.
        tok = build_seq2seq_spm_tokenizer(data_cfg, vocab_size=32000)
    else:
        if tok_path is None:
            # Fallback to default HF tokenizer if checkpoint has none
            tok_path = "gpt2"
        tok = build_tokenizer(TokenizerConfig(name_or_path=tok_path, max_length=args.max_len, max_target_length=args.max_tgt_len))
    # Ensure special tokens present (idempotent)
    tok = ensure_special_tokens(tok)
    vocab_size = len(tok)

    # Try to read run_config.json for reference (optional)
    run_cfg_path = os.path.join(ckpt_dir, "run_config.json")
    prev_model_class = None
    if os.path.isfile(run_cfg_path):
        try:
            with open(run_cfg_path, "r", encoding="utf-8") as f:
                rc = json.load(f)
            prev_model_class = rc.get("model_class")
        except Exception:
            pass

    # Parse architecture & attention modes from run directory name, allow CLI overrides
    n_layers_enc, n_layers_dec, d_model, n_heads, attn_self_enc, attn_self_dec, attn_cross = parse_arch_from_run_name(ckpt_dir)
    if args.n_layers_enc is not None:
        n_layers_enc = args.n_layers_enc
    if args.n_layers_dec is not None:
        n_layers_dec = args.n_layers_dec
    if args.d_model is not None:
        d_model = args.d_model
    if args.n_heads is not None:
        n_heads = args.n_heads
    if args.attn_self_enc is not None:
        attn_self_enc = args.attn_self_enc
    if args.attn_self_dec is not None:
        attn_self_dec = args.attn_self_dec
    if args.attn_cross is not None:
        attn_cross = args.attn_cross

    common_kwargs = dict(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=4 * d_model,
        dropout=args.dropout,
        synth_hidden=args.synth_hidden,
        synth_fixed_random=args.synth_fixed_random,
        gate_init=args.gate_init,
    )

    # Build model for TARGET TASK
    if data_cfg.task_type == TaskType.SEQ2SEQ:
        model = build_seq2seq_model(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            n_layers_enc=n_layers_enc,
            n_layers_dec=n_layers_dec,
            max_src_len=data_cfg.max_length,
            max_tgt_len=data_cfg.max_target_length,
            attn_mode_self_enc=attn_self_enc,
            attn_mode_self_dec=attn_self_dec,
            attn_mode_cross=attn_cross,
            **common_kwargs,
        )
    elif data_cfg.task_type == TaskType.CAUSAL_LM:
        model = build_decoder_lm(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            n_layers_dec=n_layers_dec,
            max_tgt_len=data_cfg.max_length,
            attn_mode_self_dec=attn_self_dec,
            **common_kwargs,
        )
    else:  # CLASSIFICATION
        raw = load_dataset(data_cfg.dataset_id, data_cfg.dataset_config)
        num_labels = raw[data_cfg.split_train].features[data_cfg.label_field].num_classes  # type: ignore[index]
        model = build_classifier(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            num_labels=num_labels,
            n_layers_enc=n_layers_enc,
            max_src_len=data_cfg.max_length,
            attn_mode_self_enc=attn_self_enc,
            **common_kwargs,
        )

    # Load checkpoint weights (strict=False inside helper)
    model = Trainer.load_model_from_checkpoint(model, ckpt_dir, map_location="cpu")

    # Optional freezing
    maybe_freeze(model, freeze_encoder=args.freeze_encoder, freeze_decoder=args.freeze_decoder, freeze_embeddings=args.freeze_embeddings)

    # Output directory
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use target dataset name in run, preserve original arch and modes
        base = f"{data_cfg.name}-enc{n_layers_enc}dec{n_layers_dec}-d{d_model}h{n_heads}-{attn_self_enc}.{attn_self_dec}.{attn_cross}_finetune_{ts}"
        out_dir = os.path.join("runs", base)
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Train
    train_cfg = TrainConfig(
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        log_every=args.log_every,
        fp16=args.fp16,
        num_workers=args.num_workers,
        save_every_epoch=args.save_every_epoch,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tok,
        data_config=data_cfg,
        train_config=train_cfg,
    )

    trainer.train()


if __name__ == "__main__":
    main()
