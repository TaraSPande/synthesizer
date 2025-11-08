"""
Supports multiple attention mechanisms selectable per layer:
- 'vanilla' : standard scaled dot-product attention
- 'synth_dense' : Synthesizer (Dense): attn from MLP(Q) → logits over keys
- 'synth_random' : Synthesizer (Random): learned/fixed random logits
- 'hybrid_dense' : gated mixture of vanilla + synth_dense (learnable gate per head)
- 'hybrid_random' : gated mixture of vanilla + synth_random (learnable gate per head)


This module provides three model classes, all sharing the same attention code path:
- EncoderDecoderTransformer (seq2seq; for WMT14, CNN/DM)
- DecoderOnlyTransformer (causal LM; for LM1B, C4, PersonaChat-as-LM)
- EncoderClassifier (classification; for GLUE/SuperGLUE)
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import dataclasses
from datetime import datetime
from typing import List, Optional

import torch

# Data registry
from preprocess import (
    DATA_REGISTRY,
    TaskType,
    TokenizerConfig,
    build_tokenizer,
    build_seq2seq_spm_tokenizer,
)

# Trainer
from trainer import (
    Trainer,
    TrainConfig
)

# Pluggable Transformer factories
from transformer.models import (
    build_seq2seq_model,
    build_decoder_lm,
    build_classifier,
)

from experiments import (
    Experiment,
    EXPERIMENTS
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextmanager
def tee_to_file(log_path: str):
    """Mirror stdout/stderr to a logfile while also keeping console output."""
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", buffering=1) as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def unique_run_dir(root: str, slug: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(root, f"{slug}_{ts}")
    path = base
    n = 1
    while os.path.exists(path):
        n += 1
        path = f"{base}_{n}"
    os.makedirs(path, exist_ok=False)
    return path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(exp: Experiment):
    data_cfg = DATA_REGISTRY[exp.data_key]
    # Override lengths from experiment settings
    data_cfg = dataclasses.replace(data_cfg, max_length=exp.max_len, max_target_length=exp.max_tgt_len)

    # Resolve layers per side
    n_enc = exp.layers_enc or exp.layers
    n_dec = exp.layers_dec or exp.layers

    # Build tokenizer first (use shared SentencePiece for seq2seq if requested)
    if data_cfg.task_type == TaskType.SEQ2SEQ and getattr(exp, "spm32k", False):
        tok = build_seq2seq_spm_tokenizer(data_cfg, vocab_size=32000)
    else:
        tok = build_tokenizer(TokenizerConfig(name_or_path=exp.tokenizer, max_length=exp.max_len, max_target_length=exp.max_tgt_len))
    vocab_size = len(tok)

    # Build model based on task type
    common_kwargs = dict(
        d_model=exp.model_dim,
        n_heads=exp.heads,
        d_ff=exp.d_ff_scale * exp.model_dim,
        dropout=exp.dropout,
        synth_hidden=exp.synth_hidden,
        synth_fixed_random=exp.synth_fixed_random,
        gate_init=exp.gate_init,
    )

    if data_cfg.task_type == TaskType.SEQ2SEQ:
        model = build_seq2seq_model(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            n_layers_enc=n_enc,
            n_layers_dec=n_dec,
            max_src_len=exp.max_len,
            max_tgt_len=exp.max_tgt_len,
            attn_mode_self_enc=exp.attn_self_enc,
            attn_mode_self_dec=exp.attn_self_dec,
            attn_mode_cross=exp.attn_cross,
            **common_kwargs,
        )
    elif data_cfg.task_type == TaskType.CAUSAL_LM:
        model = build_decoder_lm(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            n_layers_dec=n_dec,
            max_tgt_len=exp.max_len,
            attn_mode_self_dec=exp.attn_self_dec,
            **common_kwargs,
        )
    else:  # CLASSIFICATION
        # determine num_labels
        from datasets import load_dataset
        raw = load_dataset(data_cfg.dataset_id, data_cfg.dataset_config)
        num_labels = raw["train"].features[data_cfg.label_field].num_classes  # type: ignore[index]
        model = build_classifier(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            num_labels=num_labels,
            n_layers_enc=n_enc,
            max_src_len=exp.max_len,
            attn_mode_self_enc=exp.attn_self_enc,
            **common_kwargs,
        )

    # Make a unique run directory + log file
    slug = exp.slug()
    run_dir = unique_run_dir(exp.out_root, slug)

    # Persist experiment meta
    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        json.dump(asdict(exp), f, indent=2)

    log_path = os.path.join(run_dir, "train.log")
    with tee_to_file(log_path):
        print("=" * 80)
        print("Starting experiment:", exp.name)
        print("Run dir:", run_dir)
        print("Slug   :", slug)
        print("Config :", json.dumps(asdict(exp), indent=2))
        print("=" * 80)

        trainer = Trainer(
            model=model,
            tokenizer=tok,
            data_config=data_cfg,
            train_config=TrainConfig(
                output_dir=run_dir,
                epochs=exp.epochs,
                batch_size=exp.batch_size,
                lr=exp.lr,
                fp16=exp.fp16,
                grad_accum_steps=exp.grad_accum_steps,
                warmup_steps=exp.warmup_steps,
            ),
        )
        trainer.train()

    print(f"Finished: {exp.name} → {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():

    if torch.cuda.is_available():
        # Enable algorithm benchmarking and TF32 on Ampere+ for faster matmuls
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        # try:
        #     # PyTorch 2.x: improves matmul perf; safe no-op on older versions
        #     torch.set_float32_matmul_precision("high")
        # except Exception:
        #     pass

    parser = argparse.ArgumentParser(description="Run multiple Transformer training jobs sequentially.")
    parser.add_argument("--only", type=str, default="", help="Comma-separated experiment names to run (from EXPERIMENTS)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    # Optional overrides for tokenizer/schedule across selected experiments
    parser.add_argument("--spm32k", action="store_true", help="Enable shared SentencePiece (32k) for seq2seq experiments")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override warmup steps for all selected experiments")
    args = parser.parse_args()

    selected = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    exps = [e for e in EXPERIMENTS if (selected is None or e.name in selected)]

    if not exps:
        print("No experiments selected. Check --only names.")
        return

    # Apply optional CLI overrides
    # if getattr(args, "spm32k", False):
    #     for e in exps:
    #         e.spm32k = True
    # if getattr(args, "warmup_steps", None) is not None:
    #     for e in exps:
    #         e.warmup_steps = args.warmup_steps

    print("Planned experiments:")
    for e in exps:
        print(f" - {e.name}: {e.slug()} (dataset={e.data_key})")

    if args.dry_run:
        return

    for e in exps:
        try:
            run_experiment(e)
        except Exception as err:
            print("[ERROR] Experiment failed:", e.name, "→", err)
            # Keep going to the next one
            continue


if __name__ == "__main__":
    main()
