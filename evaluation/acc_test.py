import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

# Ensure project root on sys.path for 'transformer' imports when running from evaluation/ dir
import sys, os
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformer.config import TransformerConfig
from transformer.models import EncoderClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load model path
# --------------------------
# Set full path to checkpoint directory (in run folder); MLM (encoder-only) models ONLY!
model_path = Path("<path>/synthesizer/runs/<model>/<epoch>")

# --------------------------
# Load tokenizer and state dict
# --------------------------
# Load tokenizer saved with the run (ensures special tokens align with model)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# Load run config (for dataset lengths etc.; used as fallback if not inferable from weights)
with open(model_path / "run_config.json") as f:
    cfg_dict = json.load(f)

train_cfg = cfg_dict.get("train_config", {})
data_cfg0 = cfg_dict.get("data_config", {})

# Load state dict first to infer shapes and attention modes
state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")

# --------------------------
# Infer model dims and config from weights (mirrors other eval scripts)
# --------------------------
d_model = None
max_src_len = None

# Prefer positional embeddings to determine lengths and d_model
# Try classifier pos first, then seq2seq src/tgt as fallback
for k, v in state_dict.items():
    if k.endswith("pos.weight.weight"):  # matches 'pos.weight.weight', 'src_pos.weight.weight', 'tgt_pos.weight.weight'
        max_src_len, d_model = int(v.shape[0]), int(v.shape[1])
        break

# Fallbacks
if d_model is None:
    # Try token embeddings
    if "tok.weight" in state_dict:
        d_model = int(state_dict["tok.weight"].shape[1])
    elif "src_tok.weight" in state_dict:
        d_model = int(state_dict["src_tok.weight"].shape[1])
    elif "tgt_tok.weight" in state_dict:
        d_model = int(state_dict["tgt_tok.weight"].shape[1])
    else:
        d_model = int(train_cfg.get("d_model", 512))

if max_src_len is None:
    max_src_len = int(data_cfg0.get("max_length", 512))

# Vocab size from embeddings if present; else tokenizer length
if "tok.weight" in state_dict:
    vocab_size = int(state_dict["tok.weight"].shape[0])
elif "src_tok.weight" in state_dict:
    vocab_size = int(state_dict["src_tok.weight"].shape[0])
elif "tgt_tok.weight" in state_dict:
    vocab_size = int(state_dict["tgt_tok.weight"].shape[0])
else:
    vocab_size = len(tokenizer)

# Feed-forward hidden size from first FF weight if available
d_ff = 4 * int(d_model)
for k, v in state_dict.items():
    if k.endswith("encoder.layers.0.ff.net.0.weight"):
        d_ff = int(v.shape[0])
        break

# Number of encoder layers from state_dict keys
max_layer_enc = -1
for k in state_dict.keys():
    m = re.match(r"encoder\.layers\.(\d+)\.", k)
    if m:
        max_layer_enc = max(max_layer_enc, int(m.group(1)))
n_layers_enc = (max_layer_enc + 1) if max_layer_enc >= 0 else int(train_cfg.get("layers_enc", train_cfg.get("layers", 6)))

# Infer n_heads from checkpoint when possible (via gate or rand_logits on encoder self-attn), else parse from path, else fallback
n_heads_det = None
for key in (
    "encoder.layers.0.self_attn.gate",
):
    if key in state_dict:
        try:
            n_heads_det = int(state_dict[key].numel())
            break
        except Exception:
            pass
if n_heads_det is None:
    for key in ("encoder.layers.0.self_attn.rand_logits",):
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
    Detect attention mode from weights.
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

def _infer_synth_hidden(sd):
    # Prefer encoder self_attn for classifier; also check decoder keys for completeness
    for prefix in (
        "encoder.layers.0.self_attn",
        "decoder.layers.0.self_attn",
        "decoder.layers.0.cross_attn",
    ):
        key = f"{prefix}.synth.0.weight"
        if key in sd:
            return int(sd[key].shape[0])
    for prefix in (
        "encoder.layers.0.self_attn",
        "decoder.layers.0.self_attn",
        "decoder.layers.0.cross_attn",
    ):
        key = f"{prefix}.synth.weight"
        if key in sd:
            return 0
    # Fallback to train_config if not present in weights
    return int(train_cfg.get("synth_hidden", 0))

# Detect attention modes from weights to avoid mismatches with stale run_config
attn_self_enc = _detect_mode(state_dict, "encoder", "self_attn")
# For completeness (not used by EncoderClassifier), default decoder-side to vanilla
attn_self_dec = "vanilla"
attn_cross    = "vanilla"

synth_hidden = _infer_synth_hidden(state_dict)
synth_fixed_random = bool(train_cfg.get("synth_fixed_random", False))
gate_init = float(train_cfg.get("gate_init", 0.5))

print(f"[info] Detected config -> d_model={d_model}, n_heads={n_heads}, enc_layers={n_layers_enc}, self_enc={attn_self_enc}")

# --------------------------
# Build model config and load
# --------------------------
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0)
if not (0 <= int(pad_id) < len(tokenizer)):
    pad_id = 0

# Infer num_labels from classifier head weight if available; fallback to run config or AG News default (4)
num_labels = None
if "classifier.weight" in state_dict:
    num_labels = int(state_dict["classifier.weight"].shape[0])
elif "num_labels" in train_cfg:
    num_labels = int(train_cfg["num_labels"])
else:
    num_labels = 4  # AG News has 4 labels

model_cfg = TransformerConfig(
    vocab_size=int(vocab_size),
    d_model=int(d_model),
    n_heads=int(n_heads),
    d_ff=int(d_ff),
    n_layers_enc=int(n_layers_enc),
    n_layers_dec=0,  # not used by the classifier
    dropout=0.1,
    max_src_len=int(max_src_len),
    max_tgt_len=1,   # not used by the classifier
    tie_tgt_embeddings=True,
    layer_norm_eps=1e-5,
    attn_mode_self_enc=attn_self_enc,
    attn_mode_self_dec=attn_self_dec,
    attn_mode_cross=attn_cross,
    synth_hidden=int(synth_hidden),
    synth_fixed_random=bool(synth_fixed_random),
    gate_init=float(gate_init),
)

model = EncoderClassifier(model_cfg, num_labels=num_labels, pad_token_id=int(pad_id))
# Load with strict=False to tolerate attention-mode-specific keys or naming differences
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"[warn] Missing keys: {len(missing)} -> {missing}")
if unexpected:
    print(f"[warn] Unexpected keys: {len(unexpected)} -> {unexpected}")
model = model.to(device)
model.eval()
print("Model Loaded!")

# --------------------------
# Dataset: AG News (text, label)
# --------------------------
# Use a modest subset for a quick accuracy estimate; increase for stability.
# Valid splits: 'train', 'test'
sample_num = 2000
subset = "test[:sample_num]"  # adjust as needed
ds = load_dataset("ag_news", split=subset)

texts = [ex["text"] for ex in ds]
labels = torch.tensor([int(ex["label"]) for ex in ds], dtype=torch.long)

# --------------------------
# Evaluation loop (batched)
# --------------------------
max_len_eval = min(512, int(max_src_len))
batch_size = 64

correct = 0
total = 0

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size].to(device)

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len_eval,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        preds = torch.argmax(logits, dim=-1)

        correct += (preds == batch_labels).sum().item()
        total += batch_labels.size(0)

acc = correct / max(1, total)
print(f"Accuracy: {acc * 100:.2f}%")
