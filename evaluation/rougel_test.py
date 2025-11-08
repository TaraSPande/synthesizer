import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer

# Ensure project root on sys.path for 'transformer' imports when running from evaluation/ dir
import sys, os
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformer.config import TransformerConfig
from transformer.models import EncoderDecoderTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load model path
# --------------------------
# Set full path to checkpoint directory (in run folder); Seq2Seq models ONLY!
model_path = Path("<path>/synthesizer/runs/<model>/<epoch>")

# --------------------------
# Load tokenizer and state dict
# --------------------------
# Load tokenizer saved with the run (ensures special tokens align with model)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# Load run config (for dataset lengths etc.)
with open(model_path / "run_config.json") as f:
    cfg_dict = json.load(f)

train_cfg = cfg_dict.get("train_config", {})
data_cfg0 = cfg_dict.get("data_config", {})

# Load state dict first to infer shapes and attention modes
state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")

# --------------------------
# Infer model dims and config from weights
# --------------------------
d_model = None
max_src_len = None
max_tgt_len = None

# Prefer positional embeddings to determine lengths and d_model
for k, v in state_dict.items():
    if k.endswith("src_pos.weight.weight"):
        max_src_len, d_model = int(v.shape[0]), int(v.shape[1])
        break
if d_model is None:
    for k, v in state_dict.items():
        if k.endswith("tgt_pos.weight.weight"):
            max_tgt_len, d_model = int(v.shape[0]), int(v.shape[1])
            break

# Fallbacks
if d_model is None:
    if "src_tok.weight" in state_dict:
        d_model = int(state_dict["src_tok.weight"].shape[1])
    elif "tgt_tok.weight" in state_dict:
        d_model = int(state_dict["tgt_tok.weight"].shape[1])
    else:
        d_model = int(train_cfg.get("d_model", 512))

if max_src_len is None:
    max_src_len = int(data_cfg0.get("max_length", 512))
if max_tgt_len is None:
    max_tgt_len = int(data_cfg0.get("max_target_length", data_cfg0.get("max_length", 128)))

# Vocab size from embeddings if present; else tokenizer length
if "src_tok.weight" in state_dict:
    vocab_size = int(state_dict["src_tok.weight"].shape[0])
elif "tgt_tok.weight" in state_dict:
    vocab_size = int(state_dict["tgt_tok.weight"].shape[0])
else:
    vocab_size = len(tokenizer)

# Feed-forward hidden size from first FF weight if available
d_ff = 4 * int(d_model)
for k, v in state_dict.items():
    if k.endswith("encoder.layers.0.ff.net.0.weight") or k.endswith("decoder.layers.0.ff.net.0.weight"):
        d_ff = int(v.shape[0])
        break

# Number of encoder/decoder layers from state_dict keys
max_layer_enc = -1
max_layer_dec = -1
for k in state_dict.keys():
    m = re.match(r"encoder\.layers\.(\d+)\.", k)
    if m:
        max_layer_enc = max(max_layer_enc, int(m.group(1)))
    m = re.match(r"decoder\.layers\.(\d+)\.", k)
    if m:
        max_layer_dec = max(max_layer_dec, int(m.group(1)))
n_layers_enc = (max_layer_enc + 1) if max_layer_enc >= 0 else int(train_cfg.get("layers_enc", train_cfg.get("layers", 6)))
n_layers_dec = (max_layer_dec + 1) if max_layer_dec >= 0 else int(train_cfg.get("layers_dec", train_cfg.get("layers", 6)))

# Infer n_heads from checkpoint when possible (via gate or rand_logits), else parse from path, else fallback
n_heads_det = None
for key in (
    "decoder.layers.0.self_attn.gate",
    "encoder.layers.0.self_attn.gate",
    "decoder.layers.0.cross_attn.gate",
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
    # Prefer decoder modules; check both self_attn and cross_attn, then encoder self
    for prefix in (
        "decoder.layers.0.self_attn",
        "decoder.layers.0.cross_attn",
        "encoder.layers.0.self_attn",
    ):
        key = f"{prefix}.synth.0.weight"
        if key in sd:
            return int(sd[key].shape[0])
    for prefix in (
        "decoder.layers.0.self_attn",
        "decoder.layers.0.cross_attn",
        "encoder.layers.0.self_attn",
    ):
        key = f"{prefix}.synth.weight"
        if key in sd:
            return 0
    # Fallback to train_config if not present in weights
    return int(train_cfg.get("synth_hidden", 0))

# Detect attention modes from weights to avoid mismatches with stale run_config
attn_self_enc = _detect_mode(state_dict, "encoder", "self_attn")
attn_self_dec = _detect_mode(state_dict, "decoder", "self_attn")
attn_cross    = _detect_mode(state_dict, "decoder", "cross_attn")

synth_hidden = _infer_synth_hidden(state_dict)
synth_fixed_random = bool(train_cfg.get("synth_fixed_random", False))
gate_init = float(train_cfg.get("gate_init", 0.5))

print(f"[info] Detected config -> d_model={d_model}, n_heads={n_heads}, enc_layers={n_layers_enc}, dec_layers={n_layers_dec}, self_enc={attn_self_enc}, self_dec={attn_self_dec}, cross={attn_cross}")

# --------------------------
# Build model config and load
# --------------------------
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0)
if not (0 <= int(pad_id) < len(tokenizer)):
    pad_id = 0

model_cfg = TransformerConfig(
    vocab_size=int(vocab_size),
    d_model=int(d_model),
    n_heads=int(n_heads),
    d_ff=int(d_ff),
    n_layers_enc=int(n_layers_enc),
    n_layers_dec=int(n_layers_dec),
    dropout=0.1,
    max_src_len=int(max_src_len),
    max_tgt_len=int(max_tgt_len),
    tie_tgt_embeddings=True,
    layer_norm_eps=1e-5,
    attn_mode_self_enc=attn_self_enc,
    attn_mode_self_dec=attn_self_dec,
    attn_mode_cross=attn_cross,
    synth_hidden=int(synth_hidden),
    synth_fixed_random=bool(synth_fixed_random),
    gate_init=float(gate_init),
)

model = EncoderDecoderTransformer(model_cfg, pad_token_id=int(pad_id))
# Load with strict=False to tolerate attention-mode-specific keys
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
# Use a modest subset for a quick ROUGE estimate; increase for stability.
sample_num = 50
ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation[:sample_num]")
src_texts = [ex["article"] for ex in ds]
tgt_texts = [ex["highlights"] for ex in ds]

# --------------------------
# Summarization (greedy or beam)
# --------------------------
def summarize_batch(model, tokenizer, texts, max_len=128, device=device, beam_size=4):
    """
    Deterministic decoding for metric evaluation.
    - If beam_size <= 1: greedy decoding (batched)
    - Else: simple per-sample beam search (no length normalization)
    """
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    name = getattr(tokenizer, "name_or_path", "") or tokenizer.__class__.__name__
    # T5-style uses pad as decoder start
    if "t5" in name.lower():
        start_id = tokenizer.pad_token_id
    else:
        start_id = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id
    if start_id is None:
        start_id = tokenizer.eos_token_id or 0
    eos_id = tokenizer.eos_token_id

    # Greedy path (batched)
    if beam_size <= 1:
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.full((batch_size, 1), start_id, device=device)
        decoder_attention_mask = torch.ones_like(decoder_input_ids, device=device)
        with torch.no_grad():
            for _ in range(max_len):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
                logits = out["logits"][:, -1, :]  # (B, V)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
                decoder_attention_mask = torch.cat([decoder_attention_mask, torch.ones_like(next_token, device=device)], dim=-1)
                if eos_id is not None and torch.all(next_token.squeeze(-1) == eos_id):
                    break
        return tokenizer.batch_decode(decoder_input_ids[:, 1:], skip_special_tokens=True)

    # Beam search path (per-sample)
    preds = []
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            print("Sample:", i)
            src_ids = input_ids[i:i+1]
            src_mask = attention_mask[i:i+1] if attention_mask is not None else None
            beams = [(torch.tensor([start_id], device=device, dtype=src_ids.dtype), 0.0, False)]
            for _ in range(max_len):
                new_beams = []
                all_finished = True
                for seq, logp, finished in beams:
                    if finished:
                        new_beams.append((seq, logp, True))
                        continue
                    all_finished = False
                    dec_in = seq.unsqueeze(0)
                    dec_mask = torch.ones_like(dec_in, device=device)
                    out = model(
                        input_ids=src_ids,
                        attention_mask=src_mask,
                        decoder_input_ids=dec_in,
                        decoder_attention_mask=dec_mask,
                    )
                    logits = out["logits"][:, -1, :].squeeze(0)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_logp, topk_idx = torch.topk(log_probs, beam_size)
                    for k in range(beam_size):
                        nid = topk_idx[k].unsqueeze(0)
                        nseq = torch.cat([seq, nid], dim=0)
                        nlogp = logp + float(topk_logp[k].item())
                        nfin = eos_id is not None and int(nid.item()) == int(eos_id)
                        new_beams.append((nseq, nlogp, nfin))
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_size]
                if all_finished:
                    break
            best_seq = max(beams, key=lambda x: x[1])[0]
            preds.append(tokenizer.decode(best_seq.tolist()[1:], skip_special_tokens=True))
    return preds

# --------------------------
# Generate summaries
# --------------------------
preds = summarize_batch(model, tokenizer, src_texts)
print("Example Output:", preds[:1])

# --------------------------
# Evaluate ROUGE-L
# --------------------------
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = [scorer.score(ref, pred) for ref, pred in zip(tgt_texts, preds)]

# Compute average ROUGE-L F1
avg_rougeL_f1 = sum([s['rougeL'].fmeasure for s in scores]) / len(scores)
print(f"ROUGE-L F1: {avg_rougeL_f1*100:.2f}")
