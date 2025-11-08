import json
import math
import os
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

from transformer.models import (
    build_seq2seq_model,
    build_decoder_lm,
    build_classifier,
    TransformerConfig,
)

from utils import set_seed
from preprocess import (
    TaskType, 
    DataConfig, 
    build_dataset, 
    DATA_REGISTRY,
    TokenizerConfig,
    build_tokenizer,
    build_seq2seq_spm_tokenizer,
    tokenize_dataset,
)
from collator import CausalLMCollator, Seq2SeqCollator, ClassificationCollator

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

@dataclass
class TrainConfig:
    output_dir: str
    epochs: int = 1
    batch_size: int = 8
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    seed: int = 42
    log_every: int = 50
    fp16: bool = False
    grad_accum_steps: int = 1
    num_workers: int = 2
    save_every_epoch: bool = True


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """LR schedule: linear warmup then cosine decay over the total training steps computed from epochs and loader length."""
    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """Noam schedule from 'Attention Is All You Need': d_model^-0.5 * min(step^-0.5, step * warmup^-1.5) with linear warmup."""
    def __init__(self, optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 data_config: DataConfig,
                 train_config: TrainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.cfg = train_config

        set_seed(self.cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Build dataset, pre-tokenize where applicable, and dataloaders
        self.dataset = build_dataset(self.data_config)
        self.dataset = tokenize_dataset(self.dataset, self.tokenizer, self.data_config)
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Optimizer & scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        decay_params = []
        nodecay_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in no_decay):
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.cfg.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        total_steps = (len(self.train_loader) * self.cfg.epochs) // self.cfg.grad_accum_steps
        if self.data_config.task_type == TaskType.SEQ2SEQ:
            d_model = getattr(getattr(self.model, "cfg", None), "d_model", 512)
            self.scheduler = NoamLR(self.optimizer, d_model=d_model, warmup_steps=self.cfg.warmup_steps)
        else:
            self.scheduler = CosineWithWarmup(self.optimizer, warmup_steps=self.cfg.warmup_steps, max_steps=total_steps)

        self.scaler = GradScaler("cuda", enabled=self.cfg.fp16)

        # Create output dir
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        cfg = self.data_config
        tok = self.tokenizer

        if cfg.task_type == TaskType.CAUSAL_LM:
            collate = CausalLMCollator(tok, max_length=cfg.max_length)
        elif cfg.task_type == TaskType.SEQ2SEQ:
            collate = Seq2SeqCollator(tok, max_source_len=cfg.max_length, max_target_len=cfg.max_target_length)
        elif cfg.task_type == TaskType.CLASSIFICATION:
            collate = ClassificationCollator(tok, max_length=cfg.max_length, fields=cfg.text_fields, label_field=cfg.label_field)  # type: ignore[arg-type]
        else:
            raise ValueError("Unknown task type")

        train_set = self.dataset[self.data_config.split_train]
        val_set = self.dataset[self.data_config.split_val]

        pin = self.device.type == "cuda"
        persistent = self.cfg.num_workers > 0
        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate,
            pin_memory=pin,
            persistent_workers=persistent
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate,
            pin_memory=pin,
            persistent_workers=persistent
        )
        return train_loader, val_loader

    def _step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        for k, v in batch.items():
            batch[k] = v.to(self.device, non_blocking=True)

        if self.data_config.task_type == TaskType.CAUSAL_LM:
            with autocast("cuda", enabled=self.cfg.fp16):
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch.get("labels"))
                loss = out["loss"]
        elif self.data_config.task_type == TaskType.SEQ2SEQ:
            with autocast("cuda", enabled=self.cfg.fp16):
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch.get("decoder_attention_mask"),
                    labels=batch.get("labels"),
                )
                loss = out["loss"]
        elif self.data_config.task_type == TaskType.CLASSIFICATION:
            with autocast("cuda", enabled=self.cfg.fp16):
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch.get("labels"))
                loss = out["loss"]
        else:
            raise ValueError("Unknown task type")
        return loss

    def train(self) -> None:
        global_step = 0
        best_val = float("inf")
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            pbar = tqdm(enumerate(self.train_loader, start=1), total=len(self.train_loader), desc=f"Epoch {epoch}")
            epoch_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in pbar:
                loss = self._step(batch) / self.cfg.grad_accum_steps
                self.scaler.scale(loss).backward()

                if step % self.cfg.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    global_step += 1

                epoch_loss += loss.item() * self.cfg.grad_accum_steps
                if global_step % self.cfg.log_every == 0:
                    pbar.set_postfix({"loss": f"{loss.item()*self.cfg.grad_accum_steps:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})

            val_loss = self.evaluate()
            print(f"Epoch {epoch} finished. Train loss={epoch_loss/len(self.train_loader):.4f} | Val loss={val_loss:.4f}")

            if self.cfg.save_every_epoch:
                self.save_checkpoint(os.path.join(self.cfg.output_dir, f"epoch{epoch}"))

            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(os.path.join(self.cfg.output_dir, "best"))

        # Always save final
        self.save_checkpoint(os.path.join(self.cfg.output_dir, "final"))

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses: List[float] = []
        for batch in self.val_loader:
            for k, v in batch.items():
                batch[k] = v.to(self.device, non_blocking=True)
            if self.data_config.task_type == TaskType.CAUSAL_LM:
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch.get("labels")).get("loss")
            elif self.data_config.task_type == TaskType.SEQ2SEQ:
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch.get("decoder_attention_mask"),
                    labels=batch.get("labels"),
                ).get("loss")
            elif self.data_config.task_type == TaskType.CLASSIFICATION:
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch.get("labels")).get("loss")
            else:
                raise ValueError
            losses.append(float(out))
        self.model.train()
        return sum(losses) / max(1, len(losses))

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        # Save optimizer & scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.cfg.fp16 else None,
        }, os.path.join(path, "trainer_states.pt"))
        # Save tokenizer
        try:
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            print(f"[warn] failed to save tokenizer: {e}")
        # Save run config
        with open(os.path.join(path, "run_config.json"), "w") as f:
            json.dump({
                "data_config": dataclasses.asdict(self.data_config),
                "train_config": dataclasses.asdict(self.cfg),
                "tokenizer": self.tokenizer.name_or_path if hasattr(self.tokenizer, "name_or_path") else "",
                "model_class": self.model.__class__.__name__,
            }, f, indent=2)
        print(f"Saved checkpoint -> {path}")

    @staticmethod
    def load_model_from_checkpoint(model: nn.Module, path: str, map_location: Optional[str] = None) -> nn.Module:
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=map_location)
        model.load_state_dict(state, strict=False)
        return model


# ------------------------------------------------------------
# Command Line Demo (CLI)
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-task Transformer Trainer")
    parser.add_argument("data_key", type=str, help=f"One of: {list(DATA_REGISTRY.keys())}")
    parser.add_argument("output_dir", type=str, help="Where to save checkpoints")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer name or path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)

    choices = ["vanilla", "synth_dense", "synth_random", "hybrid_dense", "hybrid_random"]
    parser.add_argument("--attn-self-enc", default="vanilla", choices=choices,
                        help="Encoder self-attention mode")
    parser.add_argument("--attn-self-dec", default="vanilla", choices=choices,
                        help="Decoder self-attention mode")
    parser.add_argument("--attn-cross", default="vanilla", choices=choices,
                        help="Cross attention mode (decoder ➜ encoder)")
    parser.add_argument("--synth-hidden", type=int, default=0,
                        help="Hidden size for dense Synthesizer MLP (0 = single linear)")
    parser.add_argument("--synth-fixed-random", action="store_true",
                        help="If set with synth_random/hybrid_random, freezes the random logits")
    parser.add_argument("--gate-init", type=float, default=0.5,
                        help="Initial hybrid gate (sigmoid) weight on synthesizer branch")
    parser.add_argument("--warmup_steps", type=int, default=4000,
                        help="Linear warmup steps before cosine decay.")
    parser.add_argument("--spm32k", action="store_true",
                        help="Use shared SentencePiece (32k unigram) tokenizer for seq2seq tasks.")

    args = parser.parse_args()

    data_cfg = DATA_REGISTRY[args.data_key]
    # override lengths from CLI
    data_cfg = dataclasses.replace(data_cfg, max_length=args.max_len, max_target_length=args.max_tgt_len)

    tok_cfg = TokenizerConfig(name_or_path=args.tokenizer, max_length=args.max_len, max_target_length=args.max_tgt_len)
    if data_cfg.task_type == TaskType.SEQ2SEQ and args.spm32k:
        tok = build_seq2seq_spm_tokenizer(data_cfg, vocab_size=32000)
    else:
        tok = build_tokenizer(tok_cfg)

# ------------------------------------------------------------

    vocab_size = len(tok)

    common_kwargs = dict(
        d_model=args.model_dim,
        n_heads=args.heads,
        d_ff=4 * args.model_dim,
        dropout=0.1,
        synth_hidden=args.synth_hidden,
        synth_fixed_random=args.synth_fixed_random,
        gate_init=args.gate_init,
    )

    if data_cfg.task_type == TaskType.SEQ2SEQ:
        # Full encoder–decoder
        model = build_seq2seq_model(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            n_layers_enc=args.layers,
            n_layers_dec=args.layers,
            max_src_len=args.max_len,
            max_tgt_len=args.max_tgt_len,
            attn_mode_self_enc=args.attn_self_enc,
            attn_mode_self_dec=args.attn_self_dec,
            attn_mode_cross=args.attn_cross,
            **common_kwargs,
        )

    elif data_cfg.task_type == TaskType.CAUSAL_LM:
        # Decoder-only (causal LM). Cross-attn mode is ignored here.
        model = build_decoder_lm(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            n_layers_dec=args.layers,
            max_tgt_len=args.max_len,
            attn_mode_self_dec=args.attn_self_dec,
            **common_kwargs,
        )

    else:  # CLASSIFICATION
        raw = load_dataset(data_cfg.dataset_id, data_cfg.dataset_config)
        num_labels = raw["train"].features[data_cfg.label_field].num_classes  # type: ignore[index]
        model = build_classifier(
            vocab_size=vocab_size,
            pad_token_id=tok.pad_token_id,
            num_labels=num_labels,
            n_layers_enc=args.layers,
            max_src_len=args.max_len,
            attn_mode_self_enc=args.attn_self_enc,
            **common_kwargs,
        )

# ------------------------------------------------------------

    trainer = Trainer(
        model=model,
        tokenizer=tok,
        data_config=data_cfg,
        train_config=TrainConfig(output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, fp16=args.fp16, warmup_steps=args.warmup_steps),
    )

    trainer.train()
