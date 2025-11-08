import json
import math
import re
from pathlib import Path

import torch

# Ensure project root on sys.path for 'transformer' imports when running from evaluation/ dir
import sys, os
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformer.config import TransformerConfig
from transformer.models import DecoderOnlyTransformer
from transformers import AutoTokenizer
from datasets import load_dataset

# --------------------------
# Load model path
# --------------------------
# Set full path to checkpoint directory (in run folder); CLM (decoder-only) models ONLY!
model_path = Path("<path>/synthesizer/runs/<model>/<epoch>")

# Load run config (to pick lengths, etc.)
with open(model_path / "run_config.json") as f:
    cfg_dict = json.load(f)

# --------------------------
# Load tokenizer and state dict
# --------------------------
# Load tokenizer saved with the run (ensures special tokens align with model)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load state dict first to infer shapes
state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")

# --------------------------
# Infer model dims and config from weights
# --------------------------
# Infer hyperparameters from checkpoint and run_config to avoid size mismatches
train_cfg = cfg_dict.get("train_config", {})
data_cfg0 = cfg_dict.get("data_config", {})

# d_model and max_tgt_len from positional embedding if present; fallback to tok/lm_head
d_model = None
max_tgt_len = None
for k, v in state_dict.items():
    if k.endswith("pos.weight.weight"):
        max_tgt_len, d_model = v.shape[0], v.shape[1]
        break
if d_model is None and "tok.weight" in state_dict:
    d_model = state_dict["tok.weight"].shape[1]
if max_tgt_len is None:
    max_tgt_len = int(data_cfg0.get("max_length", 512))

# vocab size from embedding if present; else tokenizer length
vocab_size = state_dict["tok.weight"].shape[0] if "tok.weight" in state_dict else len(tokenizer)

# feed-forward hidden size from first FF weight if available
d_ff = 4 * int(d_model)
for k, v in state_dict.items():
    if k.endswith("ff.net.0.weight"):
        d_ff = v.shape[0]
        break

# number of decoder layers from state_dict keys
max_layer = -1
for k in state_dict.keys():
    m = re.match(r"decoder\.layers\.(\d+)\.", k)
    if m:
        max_layer = max(max_layer, int(m.group(1)))
n_layers_dec = (max_layer + 1) if max_layer >= 0 else int(train_cfg.get("layers", 6))

# heads and attention modes from saved train_config when available
# Infer n_heads from checkpoint when possible (gate or rand_logits), else parse from path, else fallback
n_heads_det = None
for key in (
    "decoder.layers.0.self_attn.gate",
    "encoder.layers.0.self_attn.gate",
    "decoder.layers.0.cross_attn.gate",
    "encoder.layers.0.cross_attn.gate",
):
    if key in state_dict:
        try:
            n_heads_det = int(state_dict[key].numel())
            break
        except Exception:
            pass
if n_heads_det is None:
    for key in (
        "decoder.layers.0.self_attn.rand_logits",
        "encoder.layers.0.self_attn.rand_logits",
        "decoder.layers.0.cross_attn.rand_logits",
        "encoder.layers.0.cross_attn.rand_logits",
    ):
        if key in state_dict:
            try:
                n_heads_det = int(state_dict[key].shape[0])
                break
            except Exception:
                pass
if n_heads_det is None:
    m = re.search(r"h(\d+)", str(model_path))
    if m:
        try:
            n_heads_det = int(m.group(1))
        except Exception:
            n_heads_det = None
n_heads = int(train_cfg.get("heads", n_heads_det if n_heads_det is not None else 8))

def _detect_mode(sd, layer, kind):
    """
    layer: 'encoder' or 'decoder'
    kind : 'self_attn' or 'cross_attn'
    """
    base = f"{layer}.layers.0.{kind}"
    def has(prefix: str) -> bool:
        return any(k.startswith(f"{base}.{prefix}") for k in sd.keys())
    # Random synthesizer branch
    if has("rand_logits"):
        return "hybrid_random" if has("gate") else "synth_random"
    # Dense synthesizer branch
    if has("synth"):
        return "hybrid_dense" if has("gate") else "synth_dense"
    # Default
    return "vanilla"

# Infer synth_hidden from weights to avoid mismatched MLP vs Linear
def _infer_synth_hidden(sd):
    # Prefer decoder modules; check both self_attn and cross_attn
    # If 2-layer MLP exists, 'synth.0.weight' is present and its out_features is the hidden size
    for prefix in ("decoder.layers.0.self_attn", "decoder.layers.0.cross_attn"):
        key = f"{prefix}.synth.0.weight"
        if key in sd:
            return int(sd[key].shape[0])
    # Single Linear head case
    for prefix in ("decoder.layers.0.self_attn", "decoder.layers.0.cross_attn"):
        key = f"{prefix}.synth.weight"
        if key in sd:
            return 0
    # Fallback to train_config if not present in weights
    return int(train_cfg.get("synth_hidden", 0))

# Prefer values in run_config if present; otherwise detect from weights
# Detect from weights to avoid stale run_config
attn_self_dec = _detect_mode(state_dict, "decoder", "self_attn")
#attn_self_enc = _detect_mode(state_dict, "encoder", "self_attn")
attn_cross = _detect_mode(state_dict, "decoder", "cross_attn")

synth_hidden = _infer_synth_hidden(state_dict)
synth_fixed_random = bool(train_cfg.get("synth_fixed_random", False))
gate_init = float(train_cfg.get("gate_init", 0.5))

print(f"[info] Detected config -> n_heads={n_heads}, self_dec={attn_self_dec}")#, self_enc={attn_self_enc}, cross={attn_cross}")

# --------------------------
# Build model config and load
# --------------------------
model_cfg = TransformerConfig(
    vocab_size=vocab_size,
    d_model=int(d_model),
    n_heads=n_heads,
    d_ff=int(d_ff),
    n_layers_dec=n_layers_dec,
    max_tgt_len=int(max_tgt_len),
    max_src_len=int(max_tgt_len),
    attn_mode_self_dec=attn_self_dec,
    attn_mode_cross=attn_cross,
    dropout=0.1,
    tie_tgt_embeddings=True,
    layer_norm_eps=1e-5,
    synth_hidden=synth_hidden,
    synth_fixed_random=synth_fixed_random,
    gate_init=gate_init,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate and load weights
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0)
if not (0 <= pad_id < len(tokenizer)):
    pad_id = 0
model = DecoderOnlyTransformer(model_cfg, pad_token_id=pad_id)
# Filter to decoder-only keys to avoid loading encoder params into a decoder-only model
allowed_prefixes = ("tok.", "pos.", "decoder.", "lm_head.", "ln_f.")
state_dict = {k: v for k, v in state_dict.items() if k.startswith(allowed_prefixes)}
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"[warn] Missing keys: {len(missing)} -> {missing}")
if unexpected:
    print(f"[warn] Unexpected keys: {len(unexpected)} -> {unexpected}")
model = model.to(device)
model.eval()
print("Model Loaded!")

# --------------------------
# Dataset
# --------------------------
sample_num = 500
data_cfg = cfg_dict.get("data_config", {})
dataset_id = data_cfg.get("dataset_id", "dvruette/lm1b")
dataset_config = data_cfg.get("dataset_config", None)
split_val = data_cfg.get("split_val", "validation") or "validation"
# Take a small slice for a quick demo; increase for a full evaluation
split_spec = f"{split_val}[:sample_num]"

if dataset_config is not None:
    ds = load_dataset(dataset_id, dataset_config, split=split_spec)
else:
    ds = load_dataset(dataset_id, split=split_spec)

# Build list of raw texts
cols = ds.column_names
text_fields = data_cfg.get("text_fields", []) or []
texts = None
if "text" in cols:
    texts = ds["text"]
elif text_fields and text_fields[0] in cols:
    texts = ds[text_fields[0]]
elif "dialog" in cols:
    # Personachat-like: stitch dialog utterances into a single conversation string
    def _join_dialog(d):
        utterances = []
        for u in d:
            if isinstance(u, dict) and "text" in u:
                utterances.append(u["text"])
            elif isinstance(u, str):
                utterances.append(u)
        return "\n".join(f"<u{i}>: {utt}" for i, utt in enumerate(utterances))
    texts = [ _join_dialog(d) for d in ds["dialog"] ]
else:
    raise ValueError(f"Could not infer text field from dataset columns: {cols}")

print("Dataset Loaded!")

# --------------------------
# Compute PPL
# --------------------------
@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, max_length=512, batch_size=8, device=device):
    """
    Computes perplexity = exp(average negative log-likelihood) over all tokens.
    Labels are built to predict the next token (shifted right); PAD and the last
    position in each sequence are masked with -100 and excluded from loss.
    """
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Build next-token prediction labels (ignore last position)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # ignore final position; no next token
        # Also ignore PAD positions (where attention_mask == 0)
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Model returns mean CE over valid (labels != -100) positions
        batch_loss = float(out["loss"])
        num_tokens = int((labels != -100).sum().item())

        total_nll += batch_loss * num_tokens
        total_tokens += num_tokens

    avg_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, total_tokens

max_len = cfg_dict["data_config"].get("max_length", 512)
ppl, avg_nll, ntoks = compute_perplexity(model, tokenizer, texts, max_length=max_len, batch_size=8, device=device)

# --------------------------
# Evaluate PPL
# --------------------------
print(f"Evaluated tokens: {ntoks}")
print(f"Average NLL: {avg_nll:.4f}")
print(f"Perplexity: {ppl:.2f}")
print(f"Log Perplexity (nats): {math.log(ppl):.4f}")
