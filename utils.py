import random
import torch
from transformers import PreTrainedTokenizerBase, GPT2Tokenizer, GPT2TokenizerFast

def set_seed(seed: int) -> None:
    """Sets Python & PyTorch random seeds for reproducibility (CPU+CUDA)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_special_tokens(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """
    Only modify special tokens for GPT-2 tokenizers. For all other tokenizers
    (e.g., Helsinki-NLP/opus-mt-*), return unchanged to respect their built-ins.
    For GPT-2, ensure we have distinct pad/bos/unk tokens and avoid tying to eos.
    Caller must resize model embeddings if tokenizer size changes.
    """
    # Detect GPT-2 family
    is_gpt2 = isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)) or (
        hasattr(tokenizer, "name_or_path") and "gpt2" in str(tokenizer.name_or_path).lower()
    )
    if not is_gpt2:
        return tokenizer

    # Only for GPT-2: add missing tokens and set fields
    # Do not alter eos if it already exists (GPT-2 has eos_token by default)
    if tokenizer.pad_token is None or (
        tokenizer.eos_token is not None and tokenizer.pad_token_id == tokenizer.eos_token_id
    ):
        if "<|pad|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        tokenizer.pad_token = "<|pad|>"

    if tokenizer.bos_token is None or (
        tokenizer.eos_token is not None and tokenizer.bos_token_id == tokenizer.eos_token_id
    ):
        if "<|bos|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|bos|>"]})
        tokenizer.bos_token = "<|bos|>"

    if tokenizer.unk_token is None or (
        tokenizer.eos_token is not None and tokenizer.unk_token_id == tokenizer.eos_token_id
    ):
        if "<|unk|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|unk|>"]})
        tokenizer.unk_token = "<|unk|>"

    # If eos_token is completely missing (unlikely for GPT-2), set a conventional eos
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    return tokenizer
