from dataclasses import dataclass
from typing import Literal

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

AttentionMode = Literal[
    'vanilla', 'synth_dense', 'synth_random', 'hybrid_dense', 'hybrid_random'
]


@dataclass
class TransformerConfig:
    # Model dims
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers_enc: int = 6
    n_layers_dec: int = 6
    dropout: float = 0.1

    # Positional embeddings
    max_src_len: int = 2048
    max_tgt_len: int = 2048

    # Attention flavors
    attn_mode_self_enc: AttentionMode = 'vanilla'
    attn_mode_self_dec: AttentionMode = 'vanilla'
    attn_mode_cross: AttentionMode = 'vanilla'

    # Synthesizer options
    synth_hidden: int = 0  # 0 = single linear; >0 = 2-layer MLP hidden size
    synth_fixed_random: bool = False  # for synth_random only (False = learnable)

    # Hybrid gates init (sigmoid(gate)) ≈ weight on synthesizer branch
    gate_init: float = 0.5

    # Tie embeddings and LM head for decoder side
    tie_tgt_embeddings: bool = True

    # Norm eps
    layer_norm_eps: float = 1e-5