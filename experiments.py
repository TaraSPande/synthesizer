from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    data_key: str
    out_root: str = "./runs"
    tokenizer: str = "gpt2"

    # training
    epochs: int = 10
    batch_size: int = 128
    lr: float = 5e-4
    fp16: bool = False #true when using new enough GPU + supporting dataset
    grad_accum_steps: int = 1
    warmup_steps: int = 4000
    spm32k: bool = False #sentencepiece 32k overrides tokenizer (use for seq2seq)

    # model
    model_dim: int = 512
    heads: int = 8
    layers: int = 6  # used for encoder and/or decoder unless overridden
    layers_enc: Optional[int] = None
    layers_dec: Optional[int] = None
    d_ff_scale: int = 4  # FF size = d_ff_scale * model_dim -> filter size
    dropout: float = 0.1

    # lengths
    max_len: int = 256         # encoder or decoder (LM)
    max_tgt_len: int = 128     # decoder (seq2seq only)

    # attention choices
    attn_self_enc: str = "vanilla"
    attn_self_dec: str = "vanilla"
    attn_cross: str = "vanilla"
    synth_hidden: int = 512
    synth_fixed_random: bool = False
    gate_init: float = 0.5

    def slug(self) -> str:
        enc_layers = self.layers_enc or self.layers
        dec_layers = self.layers_dec or self.layers
        return (
            f"{self.data_key}-enc{enc_layers}dec{dec_layers}-d{self.model_dim}h{self.heads}-"
            f"{self.attn_self_enc}.{self.attn_self_dec}.{self.attn_cross}"
        )


# Experiment Suite
EXPERIMENTS: List[Experiment] = [
    # Table 2 - NMT Machine Translation (WMT14) (English to German)
    Experiment(
        name="wmt14_vanilla",
        data_key="wmt14_en_de",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        spm32k=True, #tokenizer
        epochs=7,
        lr=0.5,
        warmup_steps=35000,
    ),
    Experiment(
        name="wmt14_dense",
        data_key="wmt14_en_de",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
        spm32k=True, #tokenizer
        epochs=7,
        lr=0.5,
        warmup_steps=35000,
    ),
    Experiment(
        name="wmt14_random",
        data_key="wmt14_en_de",
        attn_self_enc="synth_random",
        attn_self_dec="synth_random",
        attn_cross="synth_random",
        spm32k=True, #tokenizer
        epochs=7,
        lr=0.5,
        warmup_steps=35000,
    ),
    # Table 2 - NMT Machine Translation (WMT14) (English to French)
    Experiment(
        name="wmt14_vanilla",
        data_key="wmt14_en_fr",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        spm32k=True, #tokenizer
        lr=0.5,
        warmup_steps=35000,
    ),
    Experiment(
        name="wmt14_dense",
        data_key="wmt14_en_fr",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
        spm32k=True, #tokenizer
        lr=0.5,
        warmup_steps=35000,
    ),
    Experiment(
        name="wmt14_random",
        data_key="wmt14_en_fr",
        attn_self_enc="synth_random",
        attn_self_dec="synth_random",
        attn_cross="synth_random",
        spm32k=True, #tokenizer
        lr=0.5,
        warmup_steps=35000,
    ),
    # Table 2 - LM (LM1B)
    Experiment(
        name="lm1b_vanilla",
        data_key="lm1b",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        lr=5e-4,
        epochs=5,
        warmup_steps=4000,
    ),
    Experiment(
        name="lm1b_dense",
        data_key="lm1b",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
        lr=5e-4,
        epochs=5,
        warmup_steps=4000,
    ),
    Experiment(
        name="lm1b_random",
        data_key="lm1b",
        attn_self_enc="synth_random",
        attn_self_dec="synth_random",
        attn_cross="synth_random",
        lr=5e-4,
        epochs=5,
        warmup_steps=4000,
    ),
    # Table 3 - Summarization (CNN/DM)
    Experiment(
        name="cnn_dm_vanilla",
        data_key="cnn_dailymail",
        max_len=512,
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        spm32k=True,
        lr=0.5,
        epochs=50,
        warmup_steps=30000
    ),
    Experiment(
        name="cnn_dm_dense",
        data_key="cnn_dailymail",
        max_len=512,
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
        spm32k=True,
        lr=0.5,
        epochs=50,
        warmup_steps=30000
    ),
    Experiment(
        name="cnn_dm_random",
        data_key="cnn_dailymail",
        max_len=512,
        attn_self_enc="synth_random",
        attn_self_dec="synth_random",
        attn_cross="synth_random",
        spm32k=True,
        lr=0.5,
        epochs=50,
        warmup_steps=30000
    ),
    # Table 3 - Dialogue Generation (PersonaChat)
    Experiment(
        name="pc_vanilla",
        data_key="personachat",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        lr=4e-5,
        epochs=100,
        warmup_steps=2500
    ),
    Experiment(
        name="pc_dense",
        data_key="personachat",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
    ),
    Experiment(
        name="pc_random",
        data_key="personachat",
        attn_self_enc="synth_random",
        attn_self_dec="synth_random",
        attn_cross="synth_random",
    ),
    # Table 4 - LM (C4) 
    Experiment(
        name="c4_vanilla",
        data_key="c4_en",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
    ),
    Experiment(
        name="c4_dense",
        data_key="c4_en",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
    ),
    Experiment(
        name="c4_random",
        data_key="c4_en",
        attn_self_enc="synth_random",
        attn_self_dec="synth_random",
        attn_cross="synth_random",
    ),
    # Table 5 - GLUE
    # Multi-task finetune C4 model with all GLUE + SuperGLUE datasets
    # Table 6 - SuperGLUE
    # Multi-task finetune C4 model with all GLUE + SuperGLUE datasets

    # Table 7 - Classification
    Experiment(
        name="agnews_vanilla",
        data_key="agnews",
        attn_self_enc="vanilla",
        attn_self_dec="vanilla",
        attn_cross="vanilla",
        spm32k=True,
        lr=5e-4,
        epochs=2,
        warmup_steps=1000
    ),
    Experiment(
        name="agnews_dense",
        data_key="agnews",
        attn_self_enc="synth_dense",
        attn_self_dec="synth_dense",
        attn_cross="synth_dense",
        spm32k=True,
        lr=5e-4,
        epochs=2,
        warmup_steps=1000
    ),
    Experiment(
        name="agnews_random",
        data_key="agnews",
        attn_self_enc="synth_random",
        attn_self_dec="synth_random",
        attn_cross="synth_random",
        spm32k=True,
        lr=5e-4,
        epochs=2,
        warmup_steps=1000
    ),
]
