# -*- coding: utf-8 -*-
"""Tokenizer/lazy-model wrapper for RAG summarization examples.

Fixed version highlights
------------------------
- ``T5Summarizer()`` no longer loads the full seq2seq model by default. The
  dataset builder only needs the tokenizer, so this saves RAM/VRAM.
- The generation model is loaded lazily only when ``generate`` or ``save`` needs it.
- Contexts and targets are token-budgeted consistently.
- Label padding tokens are replaced with -100 for HuggingFace training loss.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class T5Config:
    model_name: str = "t5-base"
    device: str = "auto"

    max_input_length: int = 1024
    max_output_length: int = 256

    max_context_tokens_each: int = 220
    context_token_budget: int = 900

    load_model: bool = False
    use_safetensors: bool = True


def _choose_device() -> str:
    env_device = os.environ.get("SUMMARIZER_DEVICE", "").strip()
    if env_device:
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class T5Summarizer:
    """T5-style wrapper for building RAG summarization examples and inference."""

    def __init__(self, config: Optional[T5Config] = None, load_model: Optional[bool] = None) -> None:
        self.config = config or T5Config()
        if load_model is not None:
            self.config.load_model = bool(load_model)
        self.device = _choose_device() if self.config.device == "auto" else self.config.device

        logger.info("Loading tokenizer: %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        if self.config.load_model:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return
        logger.info("Loading seq2seq model: %s", self.config.model_name)
        kwargs: Dict[str, Any] = {}
        if self.config.use_safetensors:
            kwargs["use_safetensors"] = True
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name, **kwargs)
        except Exception:
            # Some local checkpoints do not have safetensors files.
            kwargs.pop("use_safetensors", None)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Seq2seq model loaded on: %s", self.device)

    @staticmethod
    def _safe_strip(text: Optional[str]) -> str:
        return "" if text is None else str(text).strip()

    def _encode_ids(self, text: str, max_length: Optional[int] = None) -> List[int]:
        return self.tokenizer(
            text,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

    def _truncate_one_context(self, text: str, max_tokens: int) -> str:
        clean = self._safe_strip(text)
        if not clean:
            return ""
        ids = self._encode_ids(clean, max_length=max_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def _format_contexts(
        self,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
    ) -> List[str]:
        if contexts:
            clean = [self._safe_strip(c) for c in contexts if self._safe_strip(c)]
            if max_contexts is not None:
                clean = clean[: max(int(max_contexts), 0)]
            return [
                c
                for c in (
                    self._truncate_one_context(c, self.config.max_context_tokens_each)
                    for c in clean
                )
                if c
            ]

        if retrieved_items:
            items = retrieved_items[:max_contexts] if max_contexts is not None else retrieved_items
            result: List[str] = []
            for item in items:
                doc = getattr(item, "document", None)
                if doc is None:
                    continue
                text = self._safe_strip(getattr(doc, "text", ""))
                if not text:
                    continue
                if include_section_headers:
                    meta = getattr(doc, "metadata", {}) or {}
                    section = self._safe_strip(meta.get("section", ""))
                    if section:
                        text = f"[{section}] {text}"
                text = self._truncate_one_context(text, self.config.max_context_tokens_each)
                if text:
                    result.append(text)
            return result

        return []

    def _fit_contexts_to_budget(self, contexts: List[str]) -> List[str]:
        kept: List[str] = []
        used = 0
        budget = max(int(self.config.context_token_budget), 1)
        for text in contexts:
            ids = self._encode_ids(text, max_length=self.config.max_context_tokens_each)
            length = len(ids)
            if length == 0:
                continue
            if used + length > budget:
                remaining = budget - used
                if remaining >= 32:
                    ids = ids[:remaining]
                    kept.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
                    used += len(ids)
                break
            kept.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
            used += length
        return [x for x in kept if x]

    def _warn_if_truncated(self, text: str) -> None:
        estimated_tokens = int(len(text.split()) * 1.3)
        if estimated_tokens > self.config.max_input_length:
            warnings.warn(
                f"Input has ~{estimated_tokens} estimated tokens but "
                f"max_input_length={self.config.max_input_length}. It will be truncated.",
                UserWarning,
                stacklevel=3,
            )

    def build_input(
        self,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text: Optional[str] = None,
        task_prefix: str = "summarize",
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
    ) -> str:
        if input_text is not None:
            clean = self._safe_strip(input_text)
            if not clean:
                raise ValueError("`input_text` is empty.")
            ids = self.tokenizer(
                f"{task_prefix}: {clean}",
                truncation=True,
                max_length=self.config.max_input_length,
                add_special_tokens=False,
            )["input_ids"]
            result = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            self._warn_if_truncated(result)
            return result

        parts: List[str] = []
        clean_query = self._safe_strip(query)
        if clean_query:
            parts.append(f"question: {clean_query}")

        final_contexts = self._format_contexts(
            contexts=contexts,
            retrieved_items=retrieved_items,
            include_section_headers=include_section_headers,
            max_contexts=max_contexts,
        )
        final_contexts = self._fit_contexts_to_budget(final_contexts)
        if final_contexts:
            parts.append("context: " + " ".join(final_contexts))

        if not parts:
            raise ValueError("Provide either `input_text`, or `query` + contexts/retrieved_items.")

        source = f"{task_prefix}: " + " ".join(parts)
        ids = self.tokenizer(
            source,
            truncation=True,
            max_length=self.config.max_input_length,
            add_special_tokens=False,
        )["input_ids"]
        result = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        self._warn_if_truncated(result)
        return result

    def tokenize(
        self,
        text: str,
        padding: Union[bool, str] = False,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            truncation=True,
            max_length=self.config.max_input_length,
            padding=padding,
        )

    def tokenize_pair(
        self,
        source_text: str,
        target_text: str,
        padding: Union[bool, str] = "max_length",
    ) -> Dict[str, Any]:
        model_inputs = self.tokenizer(
            source_text,
            max_length=self.config.max_input_length,
            truncation=True,
            padding=padding,
        )
        labels = self.tokenizer(
            text_target=target_text,
            max_length=self.config.max_output_length,
            truncation=True,
            padding=padding,
        )
        label_ids = labels["input_ids"]
        if padding == "max_length":
            pad_id = self.tokenizer.pad_token_id
            label_ids = [(x if x != pad_id else -100) for x in label_ids]
        model_inputs["labels"] = label_ids
        return model_inputs

    def build_training_example(
        self,
        target_text: str,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text: Optional[str] = None,
        task_prefix: str = "summarize",
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
    ) -> Dict[str, str]:
        source_text = self.build_input(
            query=query,
            contexts=contexts,
            retrieved_items=retrieved_items,
            input_text=input_text,
            task_prefix=task_prefix,
            include_section_headers=include_section_headers,
            max_contexts=max_contexts,
        )
        clean_target = self._safe_strip(target_text)
        if not clean_target:
            raise ValueError("`target_text` is empty.")
        target_ids = self.tokenizer(
            clean_target,
            truncation=True,
            max_length=self.config.max_output_length,
            add_special_tokens=False,
        )["input_ids"]
        clean_target = self.tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        return {"input_text": source_text, "target_text": clean_target}

    def generate(
        self,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text: Optional[str] = None,
        task_prefix: str = "summarize",
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
        max_output_length: Optional[int] = None,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        if not do_sample and (temperature != 1.0 or top_k != 50 or top_p != 0.95):
            warnings.warn(
                "`temperature`, `top_k`, and `top_p` have no effect when do_sample=False.",
                UserWarning,
                stacklevel=2,
            )

        self._ensure_model_loaded()
        assert self.model is not None

        source_text = self.build_input(
            query=query,
            contexts=contexts,
            retrieved_items=retrieved_items,
            input_text=input_text,
            task_prefix=task_prefix,
            include_section_headers=include_section_headers,
            max_contexts=max_contexts,
        )
        inputs = self.tokenize(source_text)
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "max_length": max_output_length or self.config.max_output_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        }
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_k": top_k, "top_p": top_p})

        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def save(self, output_dir: str) -> None:
        self._ensure_model_loaded()
        assert self.model is not None
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load_from_path(cls, model_path: str, device: Optional[str] = None) -> "T5Summarizer":
        cfg = T5Config(model_name=model_path, device=device or "auto", load_model=True)
        return cls(config=cfg, load_model=True)
