from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5Tokenizer

from utils import ensure_special_tokens
import os
try:
    import sentencepiece as spm
except Exception:
    spm = None

# ------------------------------------------------------------
# Task types & config
# ------------------------------------------------------------

class TaskType(str, Enum):
    """
    Three modes the trainer understands:
        - CAUSUAL_LM (LM1B, C4, PersonaChat-as-LM)
        - SEQ2SEQ (WMT14, CNN/DailyMail)
        - CLASSIFICATION (GLUE, SuperGLUE)
    """
    CAUSAL_LM = "causal_lm"
    SEQ2SEQ = "seq2seq"
    CLASSIFICATION = "classification"


@dataclass
class DataConfig:
    """Declarative “recipe” for the dataset."""
    name: str                      # human name (e.g., "wmt14_en_de")
    task_type: TaskType
    dataset_id: str                # datasets.load_dataset name
    dataset_config: Optional[str]  # e.g., "de-en", "en", "3.0.0"
    text_fields: Tuple[str, ...]   # which columns from HF dataset to use
    label_field: Optional[str] = None   # classification label column
    source_lang: Optional[str] = None   # for translation
    target_lang: Optional[str] = None   # for translation
    split_train: str = "train"
    split_val: str = "validation"
    split_test: Optional[str] = None
    max_length: int = 512
    max_target_length: int = 128       # for seq2seq targets (e.g., summaries)

# ------------------------------------------------------------
# Dataset registry & preprocessors
# ------------------------------------------------------------

# Helper: trim very long texts for faster demo runs
def _truncate_text(text: str, max_chars: int = 4000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def build_dataset(config: DataConfig) -> DatasetDict:
    ds = load_dataset(config.dataset_id, config.dataset_config)

    if config.task_type == TaskType.SEQ2SEQ:
        # translation or summarization
        train_cols = ds[config.split_train].column_names
        if "translation" in train_cols:
            # Translation-style: {"translation": {src_lang: ..., tgt_lang: ...}}
            src_lang = config.source_lang or "en"
            tgt_lang = config.target_lang or "de"

            def map_fn(batch):
                src = [ex[src_lang] for ex in batch["translation"]]
                tgt = [ex[tgt_lang] for ex in batch["translation"]]
                return {"src": src, "tgt": tgt}

            ds = ds.map(map_fn, batched=True, remove_columns=train_cols)

        elif ("article" in train_cols and "highlights" in train_cols) or (
            isinstance(config.text_fields, tuple) and set(config.text_fields) == {"article", "highlights"}
        ):
            # Summarization-style: article/highlights
            src_field, tgt_field = config.text_fields

            def map_fn(batch):
                src = [_truncate_text(a) for a in batch[src_field]]
                tgt = batch[tgt_field]
                return {"src": src, "tgt": tgt}

            ds = ds.map(map_fn, batched=True)

        else:
            raise ValueError(f"Unsupported seq2seq dataset: {config.dataset_id}")

    elif config.task_type == TaskType.CAUSAL_LM:
        field = config.text_fields[0]

        if config.dataset_id == "personachat":
            def map_fn(batch):
                dialogs = batch["dialog"]
                texts = []
                for d in dialogs:
                    utterances = [u["text"] if isinstance(u, dict) and "text" in u else (u if isinstance(u, str) else "") for u in d]
                    convo = "\n".join([f"<u{i}>: {utt}" for i, utt in enumerate(utterances)])
                    texts.append(convo)
                return {"text": texts}
            ds = ds.map(map_fn, batched=True, remove_columns=ds[config.split_train].column_names)
        else:
            def map_fn(batch):
                return {"text": [_truncate_text(t) for t in batch[field]]}
            ds = ds.map(map_fn, batched=True)

    elif config.task_type == TaskType.CLASSIFICATION:
        if config.label_field is None:
            raise ValueError("label_field required for classification tasks")
    else:
        raise ValueError(f"Unknown task type: {config.task_type}")

    return ds


# Predefined DataConfigs for requested datasets
DATA_REGISTRY: Dict[str, DataConfig] = {
    # Machine Translation (WMT14 parquet-backed)
    "wmt14_en_de": DataConfig(
        name="wmt14_en_de", task_type=TaskType.SEQ2SEQ, dataset_id="wmt/wmt14",
        dataset_config="de-en", text_fields=("translation",), source_lang="en", target_lang="de",
        split_train="train", split_val="validation"
    ),
    "wmt14_en_fr": DataConfig(
        name="wmt14_en_fr", task_type=TaskType.SEQ2SEQ, dataset_id="wmt/wmt14",
        dataset_config="fr-en", text_fields=("translation",), source_lang="en", target_lang="fr",
        split_train="train", split_val="validation"
    ),

    # Language Modeling
    "lm1b": DataConfig(
        name="lm1b", task_type=TaskType.CAUSAL_LM, dataset_id="dvruette/lm1b",
        dataset_config=None, text_fields=("text",), split_train="train", split_val="test"
    ),
    "c4_en": DataConfig(
        name="c4_en", task_type=TaskType.CAUSAL_LM, dataset_id="allenai/c4",
        dataset_config="en", text_fields=("text",), split_train="train", split_val="validation"
    ),

    # Summarization
    "cnn_dailymail": DataConfig(
        name="cnn_dailymail", task_type=TaskType.SEQ2SEQ, dataset_id="abisee/cnn_dailymail",
        dataset_config="3.0.0", text_fields=("article", "highlights"),
        split_train="train", split_val="validation"
    ),

    # Dialogue Generation (as LM) — left as-is; adjust if you use a parquet mirror
    "personachat": DataConfig(
        name="personachat", task_type=TaskType.CAUSAL_LM, dataset_id="google/Synthetic-Persona-Chat",
        dataset_config=None, text_fields=("user 1 personas", "user 2 personas", "Best Generated Conversation"),
        split_train="train", split_val="validation"
    ),

    # GLUE (parquet mirror)
    "glue_cola": DataConfig(
        name="glue_cola", task_type=TaskType.CLASSIFICATION, dataset_id="nyu-mll/glue",
        dataset_config="cola", text_fields=("sentence",), label_field="label",
        split_train="train", split_val="validation"
    ),
    "glue_sst2": DataConfig(
        name="glue_sst2", task_type=TaskType.CLASSIFICATION, dataset_id="nyu-mll/glue",
        dataset_config="sst2", text_fields=("sentence",), label_field="label",
        split_train="train", split_val="validation"
    ),
    "glue_mrpc": DataConfig(
        name="glue_mrpc", task_type=TaskType.CLASSIFICATION, dataset_id="nyu-mll/glue",
        dataset_config="mrpc", text_fields=("sentence1", "sentence2"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "glue_stsb": DataConfig(
        name="glue_stsb", task_type=TaskType.CLASSIFICATION, dataset_id="nyu-mll/glue",
        dataset_config="stsb", text_fields=("sentence1", "sentence2"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "glue_qqp": DataConfig(
        name="glue_qqp", task_type=TaskType.CLASSIFICATION, dataset_id="nyu-mll/glue",
        dataset_config="qqp", text_fields=("question1", "question2"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "glue_mnli": DataConfig(
        name="glue_mnli", task_type=TaskType.CLASSIFICATION, dataset_id="nyu-mll/glue",
        dataset_config="mnli", text_fields=("premise", "hypothesis"), label_field="label",
        split_train="train", split_val="validation_matched"
    ),
    "glue_qnli": DataConfig(
        name="glue_qnli", task_type=TaskType.CLASSIFICATION, dataset_id="nyu-mll/glue",
        dataset_config="qnli", text_fields=("question", "sentence"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "glue_rte": DataConfig(
        name="glue_rte", task_type=TaskType.CLASSIFICATION, dataset_id="nyu-mll/glue",
        dataset_config="rte", text_fields=("sentence1", "sentence2"), label_field="label",
        split_train="train", split_val="validation"
    ),

    # SuperGLUE (parquet mirror)
    "superglue_boolq": DataConfig(
        name="superglue_boolq", task_type=TaskType.CLASSIFICATION, dataset_id="aps/super_glue",
        dataset_config="boolq", text_fields=("question", "passage"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "superglue_cb": DataConfig(
        name="superglue_cb", task_type=TaskType.CLASSIFICATION, dataset_id="aps/super_glue",
        dataset_config="cb", text_fields=("premise", "hypothesis"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "superglue_copa": DataConfig(
        name="superglue_copa", task_type=TaskType.CLASSIFICATION, dataset_id="aps/super_glue",
        dataset_config="copa", text_fields=("premise", "choice1", "choice2", "question"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "superglue_multirc": DataConfig(
        name="superglue_multirc", task_type=TaskType.CLASSIFICATION, dataset_id="aps/super_glue",
        dataset_config="multirc", text_fields=("paragraph", "question", "answer"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "superglue_record": DataConfig(
        name="superglue_record", task_type=TaskType.SEQ2SEQ, dataset_id="aps/super_glue",
        dataset_config="record", text_fields=("passage", "query", "answers"),
        split_train="train", split_val="validation"
    ),
    "superglue_rte": DataConfig(
        name="superglue_rte", task_type=TaskType.CLASSIFICATION, dataset_id="aps/super_glue",
        dataset_config="rte", text_fields=("premise", "hypothesis"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "superglue_wic": DataConfig(
        name="superglue_wic", task_type=TaskType.CLASSIFICATION, dataset_id="aps/super_glue",
        dataset_config="wic", text_fields=("sentence1","sentence2", "word"), label_field="label",
        split_train="train", split_val="validation"
    ),
    "superglue_wsc": DataConfig(
        name="superglue_wsc", task_type=TaskType.CLASSIFICATION, dataset_id="aps/super_glue",
        dataset_config="wsc", text_fields=("text", "span1_text", "span2_text"), label_field="label",
        split_train="train", split_val="validation"
    ),
}

@dataclass
class TokenizerConfig:
    name_or_path: str = "gpt2"
    use_fast: bool = True
    max_length: int = 512
    max_target_length: int = 128


def build_tokenizer(cfg: TokenizerConfig) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(cfg.name_or_path, use_fast=cfg.use_fast)
    tok = ensure_special_tokens(tok)
    return tok


# ------------------------------------------------------------
# Offline tokenization (speeds up training by avoiding per-batch tokenization)
# ------------------------------------------------------------
def tokenize_dataset(ds: DatasetDict, tokenizer: PreTrainedTokenizerBase, config: DataConfig) -> DatasetDict:
    """
    Pre-tokenize dataset examples so the collator only pads/batches.

    - SEQ2SEQ (e.g., WMT14): produce "src_ids" and "tgt_ids".
    - CAUSAL_LM: produce "input_ids" from "text".
    - CLASSIFICATION: produce "input_ids" from text_fields (single or pair).

    All sequences are kept variable-length; padding happens in the collator.
    """
    if config.task_type == TaskType.SEQ2SEQ:
        def tok_fn_seq2seq(batch):
            enc_src = tokenizer(
                batch["src"],
                truncation=True,
                max_length=config.max_length,
                padding=False,
                add_special_tokens=True,
            )
            enc_tgt = tokenizer(
                batch["tgt"],
                truncation=True,
                max_length=config.max_target_length,
                padding=False,
                add_special_tokens=True,
            )
            return {
                "src_ids": enc_src["input_ids"],
                "tgt_ids": enc_tgt["input_ids"],
            }

        train_cols = ds[config.split_train].column_names
        remove_cols = [c for c in ("src", "tgt") if c in train_cols]
        ds = ds.map(tok_fn_seq2seq, batched=True, remove_columns=remove_cols)
        return ds

    elif config.task_type == TaskType.CAUSAL_LM:
        def tok_fn_lm(batch):
            enc = tokenizer(
                batch["text"],
                truncation=True,
                max_length=config.max_length,
                padding=False,
                add_special_tokens=True,
            )
            return {"input_ids": enc["input_ids"]}

        train_cols = ds[config.split_train].column_names
        remove_cols = [c for c in ("text",) if c in train_cols]
        ds = ds.map(tok_fn_lm, batched=True, remove_columns=remove_cols)
        return ds

    elif config.task_type == TaskType.CLASSIFICATION:
        fields = config.text_fields

        def tok_fn_cls(batch):
            if len(fields) == 1:
                enc = tokenizer(
                    batch[fields[0]],
                    truncation=True,
                    max_length=config.max_length,
                    padding=False,
                    add_special_tokens=True,
                )
            else:
                enc = tokenizer(
                    batch[fields[0]],
                    batch[fields[1]],
                    truncation=True,
                    max_length=config.max_length,
                    padding=False,
                    add_special_tokens=True,
                )
            return {"input_ids": enc["input_ids"]}

        train_cols = ds[config.split_train].column_names
        # remove only the text fields; keep label_field and any metadata
        remove_cols = [c for c in fields if c in train_cols]
        ds = ds.map(tok_fn_cls, batched=True, remove_columns=remove_cols)
        return ds

    # Default: no changes
    return ds


def build_seq2seq_spm_tokenizer(config: DataConfig, vocab_size: int = 32000, save_dir: Optional[str] = None) -> PreTrainedTokenizerBase:
    """
    Train or load a shared SentencePiece unigram tokenizer for seq2seq datasets.
    Produces a directory with spiece.model and minimal tokenizer configs, then
    loads it as a T5Tokenizer so special tokens and decoding conventions are sane.

    Especially useful for translation task (GPT-2 is english only).
    """
    if save_dir is None:
        save_dir = os.path.join("runs", "tokenizers", f"{config.name}-spm{vocab_size}")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "spiece.model")

    if not os.path.isfile(model_path):
        if spm is None:
            raise ImportError("sentencepiece is required for build_seq2seq_spm_tokenizer. Please `pip install sentencepiece`.")
        # Build dataset and collect src + tgt text
        ds = build_dataset(config)
        split = config.split_train
        corpus_path = os.path.join(save_dir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for ex in ds[split]:
                if "src" in ex:
                    s = ex["src"]
                    if isinstance(s, str) and s:
                        f.write(s.replace("\n", " ") + "\n")
                if "tgt" in ex:
                    t = ex["tgt"]
                    if isinstance(t, str) and t:
                        f.write(t.replace("\n", " ") + "\n")

        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=os.path.join(save_dir, "spiece"),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            input_sentence_size=2000000,
            shuffle_input_sentence=True
        )
        # Minimal configs so Transformers can load as a tokenizer
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            f.write('{"model_max_length": %d}' % int(config.max_length))
        with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            f.write('{"pad_token": "<pad>", "eos_token": "</s>", "bos_token": "<s>"}')

    tok = T5Tokenizer.from_pretrained(save_dir)
    tok.bos_token = "<s>" #add bos token
    return tok
