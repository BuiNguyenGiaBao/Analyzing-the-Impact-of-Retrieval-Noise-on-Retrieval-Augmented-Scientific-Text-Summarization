import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


# Config

@dataclass
class T5Config:
    """
    Configuration for the T5 summarization module.

    Notes
    -----
    Retrieval-augmented summarization can easily exceed the model
    input limit when multiple contexts are concatenated.

    Therefore two safeguards are used:

    1. max_context_tokens_each
       Each retrieved chunk is truncated individually.

    2. context_token_budget
       Total token budget allocated for all contexts combined.

    This prevents excessive truncation inside the tokenizer and
    ensures fair comparison between clean and noisy retrieval setups.
    """

    model_name: str = "t5-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # maximum length of encoder input
    max_input_length: int = 1536

    # maximum length of generated summary
    max_output_length: int = 256

    # truncate each context chunk
    max_context_tokens_each: int = 256

    # total token budget for ALL contexts combined
    context_token_budget: int = 1200
    

# Summariser

class T5Summarizer:
    """
    T5 wrapper for:
      1) Plain summarisation.
      2) Retrieval-augmented summarisation (raw contexts or SearchResult items).
      3) Building tokenised training examples.
    """

    def __init__(self, config: Optional[T5Config] = None) -> None:
        self.config = config or T5Config()
        self.device = self.config.device

        logger.info("Loading model: %s", self.config.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            use_safetensors=True
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info("Loaded on: %s", self.device)

    # Internal helpers

    @staticmethod
    def _safe_strip(text: Optional[str]) -> str:
        return "" if text is None else str(text).strip()

    def _format_contexts(
        self,
        contexts: Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        include_section_headers: bool = False,
        max_contexts: Optional[int] = None,
    ) -> List[str]:
        """
        Convert raw contexts or retrieved items into a clean
        list of context strings.

        Priority
        --------
        contexts > retrieved_items

        Notes
        -----
        Each context is truncated individually to prevent a
        single long chunk from dominating the input sequence.
        """

        if contexts:
            clean = [self._safe_strip(c) for c in contexts if self._safe_strip(c)]

            if max_contexts is not None:
                clean = clean[:max_contexts]

            clean = [
                self._truncate_one_context(
                    c,
                    self.config.max_context_tokens_each
                )
                for c in clean
            ]

            return clean

        if retrieved_items:

            items = (
                retrieved_items[:max_contexts]
                if max_contexts is not None
                else retrieved_items
            )

            result = []

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

                text = self._truncate_one_context(
                    text,
                    self.config.max_context_tokens_each
                )

                result.append(text)

            return result

        return []

    def _warn_if_truncated(self, text: str) -> None:
        """Emit a warning when *text* will likely be truncated."""
        # Fast heuristic: whitespace-split word count × 1.3 ≈ token count
        estimated_tokens = len(text.split()) * 1.3
        if estimated_tokens > self.config.max_input_length:
            warnings.warn(
                f"Input has ~{int(estimated_tokens)} estimated tokens but "
                f"max_input_length={self.config.max_input_length}. "
                "Context will be truncated — consider reducing max_contexts.",
                UserWarning,
                stacklevel=3,
            )
    def _truncate_one_context(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]
        return self.tokenizer.decode(ids, skip_special_tokens=True)        
    def _fit_contexts_to_budget(self, contexts: List[str]) -> List[str]:
        """
        Ensure that the concatenated contexts do not exceed
        the global token budget.

        Contexts are added sequentially until the token budget
        is exhausted.

        This keeps the final T5 input length stable and avoids
        uncontrolled truncation inside the tokenizer.
        """
        kept = []
        used = 0
        for text in contexts:
            ids = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_context_tokens_each,
                add_special_tokens=False
            )["input_ids"]
            length = len(ids)
            if used + length > self.config.context_token_budget:
                break
            kept.append(self.tokenizer.decode(ids, skip_special_tokens=True) )
            used += length
        return kept
    # Input construction

    def build_input(
        self,
        query:           Optional[str]       = None,
        contexts:        Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text:      Optional[str]       = None,
        task_prefix:     str                 = "summarize",
        include_section_headers: bool        = False,
        max_contexts:    Optional[int]       = None,
    ) -> str:
        """
        Build the T5 source string in one of three modes:

        1) Plain:
               ``summarize: <input_text>``
        2) RAG with raw contexts:
               ``summarize: question: … context: …``
        3) RAG with retrieved_items:
               ``summarize: question: … context: [Section] …``
        """
        if input_text is not None:
            clean = self._safe_strip(input_text)
            if not clean:
                raise ValueError("`input_text` is empty.")
            result = f"{task_prefix}: {clean}"
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

        # Ensure total context length stays within token budget
        final_contexts = self._fit_contexts_to_budget(final_contexts)

        if final_contexts:
            parts.append("context: " + " ".join(final_contexts))

        if not parts:
            raise ValueError(
                "Provide either `input_text`, or `query` + `contexts`/`retrieved_items`."
            )

        result = f"{task_prefix}: " + " ".join(parts)
        self._warn_if_truncated(result)
        return result

    # Tokenisation


    def tokenize(
        self,
        text:           str,
        padding:        bool = False,
        return_tensors: str  = "pt",
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
        padding:     Union[bool, str] = "max_length",
    ) -> Dict[str, Any]:
        """Tokenise a source/target pair for training."""
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
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

 # Training helpers

    def build_training_example(
        self,
        target_text:     str,
        query:           Optional[str]       = None,
        contexts:        Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text:      Optional[str]       = None,
        task_prefix:     str                 = "summarize",
        include_section_headers: bool        = False,
        max_contexts:    Optional[int]       = None,
    ) -> Dict[str, str]:
        """
        Return ``{"input_text": …, "target_text": …}`` ready for training.
        """
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
        return {"input_text": source_text, "target_text": clean_target}


    # Inference
    def generate(
        self,
        query:           Optional[str]       = None,
        contexts:        Optional[List[str]] = None,
        retrieved_items: Optional[List[Any]] = None,
        input_text:      Optional[str]       = None,
        task_prefix:     str                 = "summarize",
        include_section_headers: bool        = False,
        max_contexts:    Optional[int]       = None,
        max_output_length: Optional[int]     = None,
        num_beams:       int                 = 4,
        do_sample:       bool                = False,
        temperature:     float               = 1.0,
        top_k:           int                 = 50,
        top_p:           float               = 0.95,
    ) -> str:
        """
        Generate a summary from input_text, query+contexts, or query+retrieved_items.

        Note: `temperature`, `top_k`, and `top_p` only take effect when
        `do_sample=True`.  A warning is emitted if sampling params differ
        from their defaults but `do_sample=False`.
        """
        # Warn when sampling parameters are supplied but will be ignored
        if not do_sample and (temperature != 1.0 or top_k != 50 or top_p != 0.95):
            warnings.warn(
                "`temperature`, `top_k`, and `top_p` have no effect when "
                "`do_sample=False` (greedy / beam search is used). "
                "Set `do_sample=True` to enable stochastic decoding.",
                UserWarning,
                stacklevel=2,
            )

        source_text = self.build_input(
            query=query,
            contexts=contexts,
            retrieved_items=retrieved_items,
            input_text=input_text,
            task_prefix=task_prefix,
            include_section_headers=include_section_headers,
            max_contexts=max_contexts,
        )

        inputs       = self.tokenize(source_text)
        input_ids    = inputs["input_ids"].to(self.device)
        attn_mask    = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_length=max_output_length or self.config.max_output_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Persistence


    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load_from_path(
        cls,
        model_path: str,
        device: Optional[str] = None,
    ) -> "T5Summarizer":
        obj           = cls.__new__(cls)
        obj.config    = T5Config(model_name=model_path)
        obj.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.tokenizer = AutoTokenizer.from_pretrained(model_path)
        obj.model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        obj.model.to(obj.device)
        obj.model.eval()
        return obj