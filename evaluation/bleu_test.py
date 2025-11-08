import json
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
from transformer.models import EncoderDecoderTransformer
from transformers import AutoTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu

# --------------------------
# Load model path
# --------------------------
# Set full path to checkpoint directory (in run folder); Seq2Seq models ONLY!
model_path = Path("<path>/synthesizer/runs/<model>/<epoch>")

# Load run config (for dataset lengths etc.)
with open(model_path / "run_config.json") as f:
    cfg_dict = json.load(f)

train_cfg = cfg_dict.get("train_config", {})
data_cfg0 = cfg_dict.get("data_config", {})

# --------------------------
# Load tokenizer and state dict
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

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

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate and load weights
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0)
if not (0 <= pad_id < len(tokenizer)):
    pad_id = 0

model = EncoderDecoderTransformer(model_cfg, pad_token_id=int(pad_id))

# Load state dict with strict=False to tolerate attention-mode-specific keys
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
sample_num = 50
ds = load_dataset("wmt/wmt14", "de-en", split="validation[:sample_num]")

src_texts = [ex["translation"]["en"] for ex in ds]
tgt_texts = [ex["translation"]["de"] for ex in ds]

print("Dataset Loaded!")

# --------------------------
# Translation (greedy or beam)
# --------------------------
def translate_batch_fast(model, tokenizer, sentences, max_len=256, device=device, beam_size=4, length_penalty_alpha=0.6):
    """
    Fast decoding that mirrors translate_batch semantics but avoids recomputing the encoder
    and vectorizes beams per sample.

    Behavior parity:
    - If beam_size <= 1: greedy decoding (cached encoder, batched).
    - Else: per-sample beam search; we sort by cumulative log-prob ONLY during expansion,
            and apply length penalty ONLY for final selection.
    """
    model.eval()
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)  # (B, S)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    start_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    def length_penalty(L: int, alpha: float = length_penalty_alpha) -> float:
        # L = number of generated tokens (excluding the initial start token)
        return ((5.0 + float(L)) ** alpha) / ((5.0 + 1.0) ** alpha)

    with torch.no_grad():
        # Greedy (batched) with cached encoder
        if beam_size <= 1:
            # Encode once for the whole batch
            enc_pos = model._enc_positions(input_ids)
            enc_emb = model.src_tok(input_ids) + model.src_pos(enc_pos)
            enc_emb = model.drop(enc_emb)
            enc_kpm = model._kpm(input_ids, attention_mask)
            enc_out = model.encoder(enc_emb, enc_kpm)
            enc_out = model.ln_f_enc(enc_out)

            B = input_ids.size(0)
            dec_ids = torch.full((B, 1), start_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_len):
                dec_pos = model._dec_positions(dec_ids)
                dec_emb = model.tgt_tok(dec_ids) + model.tgt_pos(dec_pos)
                dec_emb = model.drop(dec_emb)
                dec_kpm = torch.ones_like(dec_ids, device=device)

                dec_out = model.decoder(dec_emb, enc_out, dec_kpm, enc_kpm)
                dec_out = model.ln_f_dec(dec_out)
                logits = model.lm_head(dec_out)[:, -1, :]  # (B, V)

                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
                dec_ids = torch.cat([dec_ids, next_token], dim=-1)

                if eos_id is not None:
                    finished |= next_token.squeeze(-1) == eos_id
                    if torch.all(finished):
                        break

            return tokenizer.batch_decode(dec_ids[:, 1:], skip_special_tokens=True)

        # Beam search: per-sample with cached encoder and vectorized beams
        preds = []
        B = input_ids.size(0)
        for i in range(B):
            print("Sample:", i)
            src_ids = input_ids[i : i + 1]                    # (1, S)
            src_kpm = model._kpm(src_ids, attention_mask[i : i + 1] if attention_mask is not None else None)

            # Cache encoder for this sample
            enc_pos = model._enc_positions(src_ids)
            enc_emb = model.src_tok(src_ids) + model.src_pos(enc_pos)
            enc_emb = model.drop(enc_emb)
            enc_out_1 = model.encoder(enc_emb, src_kpm)       # (1, S, d)
            enc_out_1 = model.ln_f_enc(enc_out_1)

            # Initialize beams (vectorized)
            K = beam_size
            beams_seq = torch.full((K, 1), start_id, dtype=torch.long, device=device)  # include BOS
            beams_logp = torch.zeros(K, dtype=torch.float32, device=device)
            beams_finished = torch.zeros(K, dtype=torch.bool, device=device)
            # Track effective generated length per beam (excluding BOS); finished beams do not increase length
            beams_len = torch.zeros(K, dtype=torch.long, device=device)

            for _ in range(max_len):
                # Prepare decoder inputs for all beams (same length by construction)
                dec_pos = model._dec_positions(beams_seq)
                dec_emb = model.tgt_tok(beams_seq) + model.tgt_pos(dec_pos)
                dec_emb = model.drop(dec_emb)
                dec_kpm = torch.ones_like(beams_seq, device=device)

                # Expand encoder cache across beams
                enc_out = enc_out_1.expand(K, -1, -1)        # (K, S, d)
                enc_kpm = src_kpm.expand(K, -1)              # (K, S)

                # Decoder forward for all beams
                dec_out = model.decoder(dec_emb, enc_out, dec_kpm, enc_kpm)
                dec_out = model.ln_f_dec(dec_out)
                logits = model.lm_head(dec_out)[:, -1, :]     # (K, V)
                log_probs = torch.log_softmax(logits, dim=-1) # (K, V)

                all_candidates = []
                for b in range(K):
                    if beams_finished[b]:
                        # Propagate finished beam unchanged: append EOS token for shape, no logp change, length unchanged
                        next_id = torch.tensor([eos_id if eos_id is not None else 0], device=device, dtype=torch.long)
                        seq_candidate = torch.cat([beams_seq[b], next_id], dim=0)
                        logp_candidate = beams_logp[b]
                        fin_candidate = True
                        len_candidate = beams_len[b]  # do not increase effective length
                        all_candidates.append((seq_candidate, logp_candidate, fin_candidate, len_candidate))
                    else:
                        topk_logp, topk_idx = torch.topk(log_probs[b], K, dim=-1)
                        for k in range(K):
                            nid = topk_idx[k].unsqueeze(0)                  # (1,)
                            nseq = torch.cat([beams_seq[b], nid], dim=0)    # (+1 length)
                            nlogp = beams_logp[b] + topk_logp[k]
                            nfin = (eos_id is not None) and (int(nid.item()) == int(eos_id))
                            nlen = beams_len[b] + 1                          # effective generated length increments by 1
                            all_candidates.append((nseq, nlogp, nfin, nlen))

                # Select top K by raw cumulative log-prob (NO length penalty here)
                all_candidates.sort(key=lambda x: float(x[1]), reverse=True)
                kept = all_candidates[:K]

                beams_seq = torch.stack([c[0] for c in kept], dim=0)
                beams_logp = torch.stack([c[1] for c in kept]).to(device)
                beams_finished = torch.tensor([bool(c[2]) for c in kept], dtype=torch.bool, device=device)
                beams_len = torch.stack([c[3] for c in kept]).to(device)

                if eos_id is not None and torch.all(beams_finished):
                    break

            # Final selection with length penalty
            L = torch.clamp(beams_len.to(torch.float32), min=1.0)  # ensure at least 1
            lp = torch.tensor([length_penalty(float(L[j].item())) for j in range(K)], device=device)
            scores = beams_logp / lp
            best_idx = torch.argmax(scores)
            best_seq = beams_seq[best_idx]  # includes BOS and extra EOS paddings

            preds.append(tokenizer.decode(best_seq.tolist()[1:], skip_special_tokens=True))

        return preds

# --------------------------
# Generate translations
# --------------------------
preds = translate_batch_fast(model, tokenizer, src_texts)
print("Example Output:", preds[:1])

# --------------------------
# Evaluate BLEU
# --------------------------
bleu = corpus_bleu(preds, [tgt_texts])
print(f"BLEU: {bleu.score:.2f}")
