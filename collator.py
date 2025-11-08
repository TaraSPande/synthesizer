from typing import Any, Dict, List, Tuple
import torch

from transformers import PreTrainedTokenizerBase

class CausalLMCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Supports two modes:
          1) Pre-tokenized examples: each item has "input_ids" (list[int]).
          2) Legacy string mode: each item has "text".
        """
        if "input_ids" in batch[0]:
            ids_list = [b["input_ids"] for b in batch]
            # Truncate to max length
            ids_list = [ids[: self.max_length] for ids in ids_list]
            # Pad to tensor batch and build attention_mask
            enc = self.tok.pad([{"input_ids": ids} for ids in ids_list], padding=True, return_tensors="pt")
            input_ids = enc.input_ids
        else:
            texts = [b["text"] for b in batch]
            enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_length,
                           return_tensors="pt")
            input_ids = enc.input_ids

        labels = input_ids.clone()
        # Shift labels for LM loss: predict next token; mask last token
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        # Normalize mask to {0,1} long
        if enc.attention_mask is not None:
            enc.attention_mask = (enc.attention_mask > 0).long()
        return {
            "input_ids": input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,
        }


class Seq2SeqCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_source_len: int, max_target_len: int):
        self.tok = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Supports two modes:
          1) Pre-tokenized examples (preferred): each item has "src_ids" and "tgt_ids" (lists of ints).
          2) Legacy string mode: each item has "src" and "tgt" strings (fallback).
        """
        pad_id = self.tok.pad_token_id
        eos_id = self.tok.eos_token_id

        if "src_ids" in batch[0] and "tgt_ids" in batch[0]:
            # Pre-tokenized path: just pad/pack
            src_ids_list = [b["src_ids"] for b in batch]
            tgt_ids_list = [b["tgt_ids"] for b in batch]

            # Truncate to max lengths
            src_ids_list = [ids[: self.max_source_len] for ids in src_ids_list]
            tgt_ids_list = [ids[: self.max_target_len] for ids in tgt_ids_list]

            # Ensure EOS present at end of each target sequence (if tokenizer defines one)
            if eos_id is not None:
                for i, ids in enumerate(tgt_ids_list):
                    if eos_id not in ids:
                        if len(ids) < self.max_target_len:
                            ids.append(eos_id)
                        else:
                            ids[-1] = eos_id
                    tgt_ids_list[i] = ids

            # Pad using tokenizer.pad to produce attention masks
            enc = self.tok.pad([{"input_ids": ids} for ids in src_ids_list], padding=True, return_tensors="pt")
            dec = self.tok.pad([{"input_ids": ids} for ids in tgt_ids_list], padding=True, return_tensors="pt")

        else:
            # Fallback: tokenize from raw strings (legacy behavior)
            src = [b["src"] for b in batch]
            tgt = [b["tgt"] for b in batch]
            enc = self.tok(src, truncation=True, padding=True, max_length=self.max_source_len, return_tensors="pt")
            dec = self.tok(tgt, truncation=True, padding=True, max_length=self.max_target_len, return_tensors="pt")

            # Ensure a proper EOS is present at the end of each target before padding
            if eos_id is not None:
                ids = dec.input_ids
                attn = dec.attention_mask
                B, T = ids.shape
                for i in range(B):
                    # compute number of non-pad tokens
                    if attn is not None:
                        length_i = int(attn[i].sum().item())
                    else:
                        if pad_id is not None:
                            length_i = int((ids[i] != pad_id).sum().item())
                        else:
                            length_i = T
                    nonpad_end = max(0, min(T, length_i))  # count of non-pad tokens
                    span = ids[i, :nonpad_end]
                    # if EOS already present in the non-pad span, skip
                    if (span == eos_id).any().item():
                        continue
                    if nonpad_end < T:
                        # place EOS into the first pad slot and mark it as non-pad
                        ids[i, nonpad_end] = eos_id
                        if attn is not None:
                            attn[i, nonpad_end] = 1
                    else:
                        # no room to append; replace last token with EOS
                        ids[i, nonpad_end - 1] = eos_id

        # Labels: ignore PAD for loss; prefer attention_mask if available
        labels = dec.input_ids.clone()
        if dec.attention_mask is not None:
            labels = labels.masked_fill(dec.attention_mask == 0, -100)
        elif pad_id is not None:
            labels[labels == pad_id] = -100

        # decoder_input_ids: shift-right by inserting BOS; also shift attention mask
        bos_id = self.tok.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer must have a BOS/CLS/EOS token for seq2seq.")
        decoder_input_ids = torch.cat([
            torch.full((dec.input_ids.size(0), 1), bos_id, dtype=dec.input_ids.dtype),
            dec.input_ids[:, :-1]
        ], dim=1)
        decoder_attention_mask = torch.cat([
            torch.ones((dec.attention_mask.size(0), 1), dtype=dec.attention_mask.dtype),
            dec.attention_mask[:, :-1]
        ], dim=1)
        # Normalize masks to {0,1} long
        if enc.attention_mask is not None:
            enc.attention_mask = (enc.attention_mask > 0).long()
        if dec.attention_mask is not None:
            dec.attention_mask = (dec.attention_mask > 0).long()
        decoder_attention_mask = (decoder_attention_mask > 0).long()

        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


class ClassificationCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int, fields: Tuple[str, ...], label_field: str):
        self.tok = tokenizer
        self.max_length = max_length
        self.fields = fields
        self.label_field = label_field

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Prefer pre-tokenized path when "input_ids" are present
        if "input_ids" in batch[0]:
            ids_list = [b["input_ids"] for b in batch]
            ids_list = [ids[: self.max_length] for ids in ids_list]
            enc = self.tok.pad([{"input_ids": ids} for ids in ids_list], padding=True, return_tensors="pt")
        else:
            if len(self.fields) == 1:
                texts = [b[self.fields[0]] for b in batch]
                enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
            else:
                t1 = [b[self.fields[0]] for b in batch]
                t2 = [b[self.fields[1]] for b in batch]
                enc = self.tok(t1, t2, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor([b[self.label_field] for b in batch], dtype=torch.long)
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labels}
