#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank-B Metrics Evaluator for Noise-Aware RAG Summarization
==========================================================

This script evaluates BART/T5 fine-tuned models for a noise-aware RAG
summarization paper.

It computes:
- ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum
- BERTScore P/R/F1
- Retrieval metrics: Hit@K, Precision@K, Recall@K, MRR, nDCG@K
- Robustness metrics: retention, absolute degradation, relative degradation
- Additive vs substitutive noise comparison
- Faithfulness proxy:
    entity precision / recall / F1
    source-supported entity rate
    unsupported entity rate
    number preservation
    source token support / sentence support proxy
- Generation quality:
    prediction length
    compression ratio
    repetition rate
    distinct-1 / distinct-2
    novel n-gram ratio
    source copy rate
- Truncation risk:
    source/target/prediction tokens
    source_truncated_risk
    target_truncated_risk
- Statistical tests:
    paired t-test
    Wilcoxon signed-rank
    Cohen's dz
    Cliff's delta
    rank-biserial correlation
    Shapiro/D'Agostino normality test of paired differences
    bootstrap 95% CI
    Holm-Bonferroni correction

Expected model output structure
-------------------------------
/workspace/outputs/bart_t5_auto_1epoch/
  01_bart_base_noiseaware/
  02_bart_base_clean_matched/
  03_t5_base_noiseaware/
  04_t5_base_clean_matched/

Expected data structure
-----------------------
/workspace/prepared_data_rankB_fixed_v2/
  test_clean.jsonl
  test_noisy_easy_additive.jsonl
  test_noisy_hard_additive.jsonl
  test_noisy_easy_substitutive.jsonl
  test_noisy_hard_substitutive.jsonl

Example
-------
python eval_rankB_metrics_runpod.py \
  --data_dir /workspace/prepared_data_rankB_fixed_v2 \
  --out_root /workspace/outputs/bart_t5_auto_1epoch \
  --output_dir /workspace/eval_outputs/rankB_metrics
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_TEST_FILES = [
    "test_clean.jsonl",
    "test_noisy_easy_additive.jsonl",
    "test_noisy_hard_additive.jsonl",
    "test_noisy_easy_substitutive.jsonl",
    "test_noisy_hard_substitutive.jsonl",
]

CORE_METRICS = [
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "bertscore_f1",
    "entity_f1_ref",
    "source_supported_entity_rate",
    "unsupported_entity_rate",
    "number_f1_ref",
    "unsupported_number_rate",
    "source_token_support_rate",
    "sentence_support_proxy",
    "distinct_1",
    "distinct_2",
    "repetition_2gram_rate",
    "novel_2gram_ratio",
]

LOWER_IS_BETTER_METRICS = {
    "unsupported_entity_rate",
    "unsupported_number_rate",
    "repetition_2gram_rate",
    "relative_degradation",
    "absolute_degradation",
}



@dataclass
class ModelSpec:
    key: str
    architecture: str
    train_setting: str
    path: str


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_json_loads(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def condition_from_filename(name: str) -> str:
    stem = Path(name).stem
    mapping = {
        "test_clean": "clean",
        "test_noisy_easy": "easy_additive",
        "test_noisy_hard": "hard_additive",
        "test_noisy_easy_additive": "easy_additive",
        "test_noisy_hard_additive": "hard_additive",
        "test_noisy_easy_substitutive": "easy_substitutive",
        "test_noisy_hard_substitutive": "hard_substitutive",
    }
    return mapping.get(stem, stem)


def normalize_paper_id(pid: Any) -> str:
    s = str(pid)
    s = re.sub(r"^test_(additive|substitutive)_", "test_", s)
    s = re.sub(r"^validation_clean_extra_", "validation_", s)
    s = re.sub(r"^train_clean_extra_", "train_", s)
    return s


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
ENTITY_RE = re.compile(
    r"""
    (?:[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,5})
    |
    (?:[A-Z]{2,}(?:-[A-Z0-9]+)*)
    """,
    re.VERBOSE,
)
NUMBER_RE = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?%?|\d{4})(?!\w)")


def tokens(text: Any) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text or ""))]


def word_count(text: Any) -> int:
    return len(str(text or "").replace("\n", " ").split())


def ngrams(tok: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tok) < n:
        return []
    return [tuple(tok[i:i+n]) for i in range(len(tok)-n+1)]


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def f1_from_pr(p: float, r: float) -> float:
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def extract_entities(text: Any) -> List[str]:
    raw = str(text or "")
    ents = []
    for m in ENTITY_RE.finditer(raw):
        ent = re.sub(r"\s+", " ", m.group(0).strip())
        # Remove single common sentence-initial words that are not informative.
        if len(ent) < 2:
            continue
        if ent.lower() in {"the", "this", "we", "in", "for", "a", "an"}:
            continue
        ents.append(ent.lower())
    return sorted(set(ents))


def extract_numbers(text: Any) -> List[str]:
    return sorted(set(m.group(0).lower() for m in NUMBER_RE.finditer(str(text or ""))))


def set_overlap_metrics(pred_items: Sequence[str], ref_items: Sequence[str]) -> Dict[str, float]:
    pred_set = set(pred_items)
    ref_set = set(ref_items)
    inter = pred_set & ref_set
    p = safe_div(len(inter), len(pred_set))
    r = safe_div(len(inter), len(ref_set))
    return {
        "precision": p,
        "recall": r,
        "f1": f1_from_pr(p, r),
        "pred_count": float(len(pred_set)),
        "ref_count": float(len(ref_set)),
        "overlap_count": float(len(inter)),
    }


def sentence_support_proxy(prediction: str, source: str) -> float:
    """Mean token-overlap support between each predicted sentence and source.

    This is not a full factuality metric, but it is a transparent proxy for
    whether generated statements are lexically grounded in the source context.
    """
    pred_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prediction) if s.strip()]
    source_tokens = set(tokens(source))
    if not pred_sents:
        return 0.0
    scores = []
    for sent in pred_sents:
        st = set(tokens(sent))
        content = {x for x in st if len(x) >= 4}
        if not content:
            continue
        scores.append(safe_div(len(content & source_tokens), len(content)))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------
# Model generation
# ---------------------------------------------------------------------

def detect_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_path: str):
    dtype = detect_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = dtype
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def batch_iter(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]


def generate_predictions_for_model(
    model_spec: ModelSpec,
    data_dir: Path,
    test_files: List[str],
    pred_dir: Path,
    max_source_length: int,
    generation_max_length: int,
    num_beams: int,
    batch_size: int,
    limit: Optional[int],
    overwrite: bool,
) -> List[Path]:
    model, tokenizer, device = load_model_and_tokenizer(model_spec.path)
    output_paths: List[Path] = []

    for test_file in test_files:
        test_path = data_dir / test_file
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file: {test_path}")

        condition = condition_from_filename(test_file)
        pred_path = pred_dir / f"{model_spec.key}__{Path(test_file).stem}.jsonl"
        output_paths.append(pred_path)

        if pred_path.exists() and not overwrite:
            print(f"[skip] predictions exist: {pred_path}")
            continue

        records = read_jsonl(test_path, limit=limit)
        out_rows: List[Dict[str, Any]] = []

        print(f"\n[generate] {model_spec.key} on {test_file} | n={len(records):,}")

        for batch in tqdm(list(batch_iter(records, batch_size)), desc=f"{model_spec.key}:{condition}"):
            sources = [str(r.get("input_text", "")) for r in batch]
            inputs = tokenizer(
                sources,
                max_length=max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_length=generation_max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for r, pred in zip(batch, preds):
                row = dict(r)
                row.update({
                    "model_key": model_spec.key,
                    "architecture": model_spec.architecture,
                    "train_setting": model_spec.train_setting,
                    "model_path": model_spec.path,
                    "test_file": test_file,
                    "condition": condition,
                    "paper_id": str(r.get("paper_id", "")),
                    "paper_id_norm": normalize_paper_id(r.get("paper_id", "")),
                    "prediction_text": pred,
                })
                out_rows.append(row)

        write_jsonl(pred_path, out_rows)
        print(f"[saved] {pred_path} | rows={len(out_rows):,}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_paths


# ---------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------

def compute_rouge(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
    )
    out = []
    for r in tqdm(rows, desc="ROUGE"):
        target = str(r.get("target_text", ""))
        pred = str(r.get("prediction_text", ""))
        scores = scorer.score(target, pred)
        item = dict(r)
        for k, v in scores.items():
            item[k] = float(v.fmeasure)
            item[f"{k}_precision"] = float(v.precision)
            item[f"{k}_recall"] = float(v.recall)
        out.append(item)
    return out


def _patch_bertscore_sent_encode(max_length: int = 512) -> None:
    """Patch bert-score tokenizer encoding to avoid OverflowError.

    Some Hugging Face tokenizers expose a very large model_max_length
    placeholder. bert-score may pass that value into tokenizer truncation,
    which can trigger: OverflowError: int too big to convert.

    This patch caps the BERTScore tokenizer length to a safe value.
    """
    import bert_score.utils as bsu

    def safe_sent_encode(tokenizer, sent):
        try:
            current_max = int(getattr(tokenizer, "model_max_length", max_length))
        except Exception:
            current_max = max_length

        if current_max <= 0 or current_max > max_length:
            try:
                tokenizer.model_max_length = max_length
            except Exception:
                pass
            current_max = max_length

        try:
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=min(current_max, max_length),
                truncation=True,
            )
        except OverflowError:
            try:
                tokenizer.model_max_length = max_length
            except Exception:
                pass
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
            )

    bsu.sent_encode = safe_sent_encode


def add_bertscore(
    rows: List[Dict[str, Any]],
    bertscore_model: str,
    bertscore_batch_size: int,
    lang: str,
    device: str,
    rescale_with_baseline: bool = False,
    bertscore_max_length: int = 512,
) -> List[Dict[str, Any]]:
    try:
        from bert_score import score as bert_score
    except Exception as e:
        raise RuntimeError("BERTScore is missing. Run: pip install bert-score") from e

    _patch_bertscore_sent_encode(max_length=bertscore_max_length)

    preds = [str(r.get("prediction_text", "")) for r in rows]
    refs = [str(r.get("target_text", "")) for r in rows]

    print(f"\n[BERTScore] {bertscore_model} | n={len(rows):,} | max_length={bertscore_max_length}")
    P, R, F1 = bert_score(
        preds,
        refs,
        model_type=bertscore_model,
        lang=lang,
        batch_size=bertscore_batch_size,
        device=device,
        verbose=True,
        rescale_with_baseline=rescale_with_baseline,
    )
    p = P.detach().cpu().numpy().astype(float)
    rec = R.detach().cpu().numpy().astype(float)
    f1 = F1.detach().cpu().numpy().astype(float)

    out = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["bertscore_p"] = float(p[i])
        item["bertscore_r"] = float(rec[i])
        item["bertscore_f1"] = float(f1[i])
        item["bertscore_model"] = bertscore_model
        item["bertscore_max_length"] = int(bertscore_max_length)
        out.append(item)
    return out


def token_count(tokenizer, text: str, add_special_tokens: bool = True) -> int:
    return len(tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)["input_ids"])


def compute_retrieval_from_flags(flags: Sequence[int], k_values: Sequence[int]) -> Dict[str, float]:
    f = [int(x) for x in flags]
    total_rel = max(sum(f), 1)
    out: Dict[str, float] = {}

    # MRR
    mrr = 0.0
    for i, hit in enumerate(f):
        if hit:
            mrr = 1.0 / (i + 1)
            break
    out["mrr"] = float(mrr)

    for k in k_values:
        top = f[:k]
        hits = sum(top)
        retrieved = len(top)
        out[f"hit@{k}"] = float(1.0 if hits > 0 else 0.0)
        out[f"precision@{k}"] = float(safe_div(hits, retrieved))
        out[f"recall@{k}"] = float(safe_div(hits, total_rel))

        dcg = 0.0
        for i, rel in enumerate(top):
            if rel:
                dcg += 1.0 / math.log2(i + 2)
        ideal_hits = min(total_rel, k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        out[f"ndcg@{k}"] = float(safe_div(dcg, idcg))

    return out


def add_rankB_record_metrics(
    rows: List[Dict[str, Any]],
    length_tokenizer: str,
    max_source_length: int,
    max_target_length: int,
    retrieval_k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    # Common tokenizer is used for cross-model reporting.
    # Model-specific tokenizer is used for model-specific truncation risk.
    common_tokenizer = AutoTokenizer.from_pretrained(length_tokenizer, use_fast=True)
    model_tokenizers: Dict[str, Any] = {}
    out = []

    for r in tqdm(rows, desc="rankB record metrics"):
        item = dict(r)

        source = str(item.get("input_text", ""))
        target = str(item.get("target_text", ""))
        pred = str(item.get("prediction_text", ""))

        # Length and truncation
        src_tok = token_count(common_tokenizer, source)
        tgt_tok = token_count(common_tokenizer, target)
        pred_tok = token_count(common_tokenizer, pred)

        # Model-specific tokenizer. This avoids judging T5 truncation with BART tokenizer only.
        model_tok_name = str(item.get("model_path") or item.get("model_key") or length_tokenizer)
        if model_tok_name not in model_tokenizers:
            try:
                model_tokenizers[model_tok_name] = AutoTokenizer.from_pretrained(model_tok_name, use_fast=True)
            except Exception:
                model_tokenizers[model_tok_name] = common_tokenizer
        model_tok = model_tokenizers[model_tok_name]

        src_tok_model = token_count(model_tok, source)
        tgt_tok_model = token_count(model_tok, target)
        pred_tok_model = token_count(model_tok, pred)

        item["source_words"] = word_count(source)
        item["target_words"] = word_count(target)
        item["prediction_words"] = word_count(pred)

        # Common-tokenizer lengths for cross-model descriptive statistics.
        item["source_tokens"] = int(src_tok)
        item["target_tokens"] = int(tgt_tok)
        item["prediction_tokens"] = int(pred_tok)
        item["source_truncated_risk"] = bool(src_tok > max_source_length)
        item["target_truncated_risk"] = bool(tgt_tok > max_target_length)

        # Model-specific lengths for fair truncation reporting per architecture.
        item["source_tokens_model_tokenizer"] = int(src_tok_model)
        item["target_tokens_model_tokenizer"] = int(tgt_tok_model)
        item["prediction_tokens_model_tokenizer"] = int(pred_tok_model)
        item["source_truncated_risk_model_tokenizer"] = bool(src_tok_model > max_source_length)
        item["target_truncated_risk_model_tokenizer"] = bool(tgt_tok_model > max_target_length)

        item["compression_ratio_pred_over_source_tokens"] = safe_div(pred_tok_model, src_tok_model)
        item["length_ratio_pred_over_target_tokens"] = safe_div(pred_tok_model, tgt_tok_model)

        # Generation quality
        src_toks = tokens(source)
        tgt_toks = tokens(target)
        pred_toks = tokens(pred)

        pred_unigrams = ngrams(pred_toks, 1)
        pred_bigrams = ngrams(pred_toks, 2)
        pred_trigrams = ngrams(pred_toks, 3)

        item["distinct_1"] = safe_div(len(set(pred_unigrams)), len(pred_unigrams))
        item["distinct_2"] = safe_div(len(set(pred_bigrams)), len(pred_bigrams))

        if pred_bigrams:
            counts2 = Counter(pred_bigrams)
            repeated = sum(c - 1 for c in counts2.values() if c > 1)
            item["repetition_2gram_rate"] = safe_div(repeated, len(pred_bigrams))
        else:
            item["repetition_2gram_rate"] = 0.0

        source_bigrams = set(ngrams(src_toks, 2))
        pred_bigrams_set = set(pred_bigrams)
        item["source_copy_2gram_rate"] = safe_div(len(pred_bigrams_set & source_bigrams), len(pred_bigrams_set))
        item["novel_2gram_ratio"] = 1.0 - item["source_copy_2gram_rate"]

        source_trigrams = set(ngrams(src_toks, 3))
        pred_trigrams_set = set(pred_trigrams)
        item["source_copy_3gram_rate"] = safe_div(len(pred_trigrams_set & source_trigrams), len(pred_trigrams_set))
        item["novel_3gram_ratio"] = 1.0 - item["source_copy_3gram_rate"]

        # Entity faithfulness proxy
        pred_entities = extract_entities(pred)
        ref_entities = extract_entities(target)
        source_entities = extract_entities(source)

        ent_ref = set_overlap_metrics(pred_entities, ref_entities)
        ref_src = set_overlap_metrics(ref_entities, source_entities)

        pred_ent_set = set(pred_entities)
        src_ent_set = set(source_entities)
        supported_ent = pred_ent_set & src_ent_set
        unsupported_ent = pred_ent_set - src_ent_set

        item["entity_precision_ref"] = ent_ref["precision"]
        item["entity_recall_ref"] = ent_ref["recall"]
        item["entity_f1_ref"] = ent_ref["f1"]
        item["prediction_entity_count"] = float(len(pred_ent_set))
        item["reference_entity_count"] = float(len(set(ref_entities)))
        item["entity_overlap_ref_count"] = ent_ref["overlap_count"]

        if len(pred_ent_set) > 0:
            item["source_supported_entity_rate"] = safe_div(len(supported_ent), len(pred_ent_set))
            item["unsupported_entity_rate"] = safe_div(len(unsupported_ent), len(pred_ent_set))
        else:
            # No predicted entities means no unsupported predicted entity.
            item["source_supported_entity_rate"] = np.nan
            item["unsupported_entity_rate"] = 0.0
        item["supported_entity_count"] = float(len(supported_ent))
        item["unsupported_entity_count"] = float(len(unsupported_ent))
        item["reference_entity_source_coverage"] = ref_src["precision"] if ref_entities else np.nan

        # Number preservation
        pred_numbers = extract_numbers(pred)
        ref_numbers = extract_numbers(target)
        src_numbers = extract_numbers(source)

        num_ref = set_overlap_metrics(pred_numbers, ref_numbers)
        pred_num_set = set(pred_numbers)
        src_num_set = set(src_numbers)
        supported_num = pred_num_set & src_num_set
        unsupported_num = pred_num_set - src_num_set

        item["number_precision_ref"] = num_ref["precision"]
        item["number_recall_ref"] = num_ref["recall"]
        item["number_f1_ref"] = num_ref["f1"]
        if len(pred_num_set) > 0:
            item["number_source_supported_rate"] = safe_div(len(supported_num), len(pred_num_set))
            item["unsupported_number_rate"] = safe_div(len(unsupported_num), len(pred_num_set))
        else:
            item["number_source_supported_rate"] = np.nan
            item["unsupported_number_rate"] = 0.0
        item["supported_number_count"] = float(len(supported_num))
        item["unsupported_number_count"] = float(len(unsupported_num))

        # Lexical source support proxy
        source_token_set = set(src_toks)
        pred_content = {x for x in pred_toks if len(x) >= 4}
        item["source_token_support_rate"] = safe_div(len(pred_content & source_token_set), len(pred_content))
        item["sentence_support_proxy"] = sentence_support_proxy(pred, source)

        # Retrieval metrics
        # Prefer context_hit_flags because they represent the final context fed into model.
        context_flags = safe_json_loads(item.get("context_hit_flags"))
        retrieved_flags = safe_json_loads(item.get("retrieved_hit_flags"))

        if isinstance(context_flags, list) and context_flags:
            ret = compute_retrieval_from_flags(context_flags, retrieval_k_values)
            for k, v in ret.items():
                item[f"context_{k}"] = v

        if isinstance(retrieved_flags, list) and retrieved_flags:
            ret = compute_retrieval_from_flags(retrieved_flags, retrieval_k_values)
            for k, v in ret.items():
                item[f"retrieval_{k}"] = v

        # Noise ratio
        num_contexts = item.get("num_contexts", None)
        num_noise = item.get("num_noise_chunks", None)
        try:
            item["noise_chunk_ratio"] = safe_div(float(num_noise), float(num_contexts))
        except Exception:
            item["noise_chunk_ratio"] = np.nan

        out.append(item)

    return out


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(values, size=n, replace=True))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # For evaluation sizes in this project this is fine.
    greater = 0
    lower = 0
    for xi in x:
        greater += np.sum(xi > y)
        lower += np.sum(xi < y)
    return float((greater - lower) / (len(x) * len(y)))


def rank_biserial_from_diff(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan
    abs_diff = np.abs(diff)
    ranks = stats.rankdata(abs_diff)
    pos = np.sum(ranks[diff > 0])
    neg = np.sum(ranks[diff < 0])
    denom = pos + neg
    return float((pos - neg) / denom) if denom else np.nan


def normality_test(diff: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 3:
        return {"normality_stat": np.nan, "normality_p": np.nan, "normality_test": "not_enough_pairs"}

    # If all paired differences are identical, a normality test is not meaningful.
    # This also prevents SciPy from spamming:
    # "Input data has range zero. The results may not be accurate."
    if np.nanmax(diff) == np.nanmin(diff):
        return {"normality_stat": np.nan, "normality_p": np.nan, "normality_test": "constant_difference"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if len(diff) <= 5000:
                res = stats.shapiro(diff)
                return {"normality_stat": float(res.statistic), "normality_p": float(res.pvalue), "normality_test": "shapiro"}
            res = stats.normaltest(diff)
            return {"normality_stat": float(res.statistic), "normality_p": float(res.pvalue), "normality_test": "dagostino_k2"}
    except Exception:
        return {"normality_stat": np.nan, "normality_p": np.nan, "normality_test": "failed"}

def paired_test_result(diff: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    out: Dict[str, Any] = {
        "n_pairs": int(len(diff)),
        "mean_diff": np.nan,
        "median_diff": np.nan,
        "std_diff": np.nan,
        "cohen_dz": np.nan,
        "cliffs_delta": np.nan,
        "rank_biserial": np.nan,
        "paired_t_stat": np.nan,
        "paired_t_p": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
        "bootstrap_ci95_low": np.nan,
        "bootstrap_ci95_high": np.nan,
        "normality_stat": np.nan,
        "normality_p": np.nan,
        "normality_test": None,
        "degenerate_diff": False,
        "stat_note": None,
    }
    if len(diff) == 0:
        out["stat_note"] = "no_valid_pairs"
        return out

    out["mean_diff"] = float(np.mean(diff))
    out["median_diff"] = float(np.median(diff))
    out["std_diff"] = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    out["cohen_dz"] = cohen_dz(diff)
    out["rank_biserial"] = rank_biserial_from_diff(diff)

    if x is not None and y is not None:
        out["cliffs_delta"] = cliffs_delta(np.asarray(x, dtype=float), np.asarray(y, dtype=float))

    if len(diff) >= 2:
        diff_range = float(np.nanmax(diff) - np.nanmin(diff))
        diff_sd = float(np.nanstd(diff, ddof=1)) if len(diff) > 1 else 0.0
        is_degenerate = (diff_range <= 1e-12) or (diff_sd <= 1e-12)
        out["degenerate_diff"] = bool(is_degenerate)

        # Bootstrap CI is still useful even if differences are constant.
        lo, hi = bootstrap_ci(diff)
        out["bootstrap_ci95_low"] = lo
        out["bootstrap_ci95_high"] = hi

        if is_degenerate:
            # t-test and normality test are not meaningful when all differences
            # are constant/nearly constant; running them causes precision-loss spam.
            if np.allclose(diff, 0.0, atol=1e-12):
                out["wilcoxon_stat"] = np.nan
                out["wilcoxon_p"] = 1.0
                out["stat_note"] = "all_differences_zero"
            else:
                out["stat_note"] = "constant_or_nearly_constant_difference"
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        warnings.simplefilter("ignore", category=UserWarning)
                        w = stats.wilcoxon(diff)
                        out["wilcoxon_stat"] = float(w.statistic)
                        out["wilcoxon_p"] = float(w.pvalue)
                except Exception:
                    pass
            out["normality_test"] = "skipped_constant_difference"
            return out

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                t = stats.ttest_1samp(diff, popmean=0.0, nan_policy="omit")
                out["paired_t_stat"] = float(t.statistic)
                out["paired_t_p"] = float(t.pvalue)
        except Exception:
            pass
        try:
            if not np.allclose(diff, 0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    warnings.simplefilter("ignore", category=UserWarning)
                    w = stats.wilcoxon(diff)
                    out["wilcoxon_stat"] = float(w.statistic)
                    out["wilcoxon_p"] = float(w.pvalue)
        except Exception:
            pass
        out.update(normality_test(diff))
    return out

def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    p = np.asarray([np.nan if v is None else float(v) for v in p_values], dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return out.tolist()

    ordered = valid[np.argsort(p[valid])]
    m = len(ordered)
    adjusted_sorted = []
    for rank, idx in enumerate(ordered):
        adjusted_sorted.append((m - rank) * p[idx])
    # enforce monotonic non-decreasing in sorted order
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    for idx, adj in zip(ordered, adjusted_sorted):
        out[idx] = adj
    return out.tolist()


def align_groups(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, id_col: str = "paper_id_norm") -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    a = df_a[[id_col, metric]].copy().rename(columns={metric: f"{metric}_a"})
    b = df_b[[id_col, metric]].copy().rename(columns={metric: f"{metric}_b"})
    merged = a.merge(b, on=id_col, how="inner")

    if merged.empty:
        n = min(len(df_a), len(df_b))
        merged = pd.DataFrame({
            id_col: list(range(n)),
            f"{metric}_a": df_a[metric].iloc[:n].to_numpy(),
            f"{metric}_b": df_b[metric].iloc[:n].to_numpy(),
        })

    x = merged[f"{metric}_a"].to_numpy(dtype=float)
    y = merged[f"{metric}_b"].to_numpy(dtype=float)
    diff = x - y
    return diff, x, y, merged



def numeric_series(series: pd.Series) -> pd.Series:
    """Convert numeric/bool/object columns to float safely for aggregation.

    Pandas may keep boolean dtype after pd.to_numeric, and numpy quantile
    cannot interpolate booleans. This helper converts True/False to 1/0 and
    returns float values with NaNs removed.
    """
    if series.dtype == bool:
        return series.astype(float).dropna()
    vals = pd.to_numeric(series, errors="coerce")
    return vals.astype(float).dropna()

def summary_tables(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    group_cols = ["architecture", "train_setting", "model_key", "condition", "test_file"]
    rows = []
    extra_cols = [
        "source_tokens", "target_tokens", "prediction_tokens",
        "source_tokens_model_tokenizer", "target_tokens_model_tokenizer", "prediction_tokens_model_tokenizer",
        "source_words", "target_words", "prediction_words",
        "source_truncated_risk", "target_truncated_risk",
        "source_truncated_risk_model_tokenizer", "target_truncated_risk_model_tokenizer",
        "context_mrr", "retrieval_mrr", "noise_chunk_ratio",
    ]
    all_cols = list(metrics) + [c for c in extra_cols if c in df.columns]

    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(g))
        for col in all_cols:
            if col not in g.columns:
                continue
            vals = numeric_series(g[col])
            if len(vals) == 0:
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{col}_median"] = float(vals.median())
            row[f"{col}_p95"] = float(np.quantile(vals.to_numpy(dtype=float), 0.95))
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_max"] = float(vals.max())
            lo, hi = bootstrap_ci(vals.to_numpy(dtype=float), n_boot=2000)
            row[f"{col}_ci95_low"] = lo
            row[f"{col}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)



def safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))

def compare_noiseaware_vs_clean(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for arch in sorted(df["architecture"].dropna().unique()):
        for test_file in sorted(df["test_file"].dropna().unique()):
            g = df[(df["architecture"] == arch) & (df["test_file"] == test_file)]
            noise = g[g["train_setting"] == "noiseaware"]
            clean = g[g["train_setting"] == "clean_matched"]
            if noise.empty or clean.empty:
                continue
            for metric in metrics:
                if metric not in g.columns:
                    continue
                diff, x, y, _ = align_groups(noise, clean, metric)
                res = paired_test_result(diff, x=x, y=y)
                res.update({
                    "comparison": "noiseaware_minus_clean_matched",
                    "architecture": arch,
                    "test_file": test_file,
                    "condition": condition_from_filename(test_file),
                    "metric": metric,
                    "mean_noiseaware": safe_nanmean(x) if len(x) else np.nan,
                    "mean_clean_matched": safe_nanmean(y) if len(y) else np.nan,
                    "positive_means": "noise-aware higher; check metric_direction",
                    "metric_direction": "lower_is_better" if metric in LOWER_IS_BETTER_METRICS else "higher_is_better",
                })
                rows.append(res)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_p_holm"] = holm_bonferroni(out["paired_t_p"].tolist())
        out["wilcoxon_p_holm"] = holm_bonferroni(out["wilcoxon_p"].tolist())
    return out


def robustness_degradation(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    clean_file = "test_clean.jsonl"

    for model_key in sorted(df["model_key"].dropna().unique()):
        g_model = df[df["model_key"] == model_key]
        clean = g_model[g_model["test_file"] == clean_file]
        if clean.empty:
            continue

        for test_file in sorted(g_model["test_file"].dropna().unique()):
            if test_file == clean_file:
                continue
            noisy = g_model[g_model["test_file"] == test_file]
            if noisy.empty:
                continue

            for metric in metrics:
                if metric not in g_model.columns:
                    continue
                diff, x, y, _ = align_groups(noisy, clean, metric)
                res = paired_test_result(diff, x=x, y=y)
                clean_mean = safe_nanmean(y) if len(y) else np.nan
                noisy_mean = safe_nanmean(x) if len(x) else np.nan
                res.update({
                    "comparison": "noisy_minus_clean_test",
                    "model_key": model_key,
                    "architecture": g_model.iloc[0].get("architecture"),
                    "train_setting": g_model.iloc[0].get("train_setting"),
                    "test_file": test_file,
                    "condition": condition_from_filename(test_file),
                    "metric": metric,
                    "mean_noisy": noisy_mean,
                    "mean_clean": clean_mean,
                    "retention_rate": safe_div(noisy_mean, clean_mean),
                    "absolute_degradation": clean_mean - noisy_mean,
                    "relative_degradation": safe_div(clean_mean - noisy_mean, clean_mean),
                    "negative_mean_diff_means": "performance degradation under noise for higher-is-better metrics",
                    "metric_direction": "lower_is_better" if metric in LOWER_IS_BETTER_METRICS else "higher_is_better",
                })
                rows.append(res)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_p_holm"] = holm_bonferroni(out["paired_t_p"].tolist())
        out["wilcoxon_p_holm"] = holm_bonferroni(out["wilcoxon_p"].tolist())
    return out


def additive_vs_substitutive(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    pairs = [
        ("test_noisy_easy_additive.jsonl", "test_noisy_easy_substitutive.jsonl", "easy_additive_minus_substitutive"),
        ("test_noisy_hard_additive.jsonl", "test_noisy_hard_substitutive.jsonl", "hard_additive_minus_substitutive"),
    ]
    rows = []

    for model_key in sorted(df["model_key"].dropna().unique()):
        g_model = df[df["model_key"] == model_key]
        for file_a, file_b, comp in pairs:
            a = g_model[g_model["test_file"] == file_a]
            b = g_model[g_model["test_file"] == file_b]
            if a.empty or b.empty:
                continue
            for metric in metrics:
                if metric not in g_model.columns:
                    continue
                diff, x, y, _ = align_groups(a, b, metric)
                res = paired_test_result(diff, x=x, y=y)
                res.update({
                    "comparison": comp,
                    "model_key": model_key,
                    "architecture": g_model.iloc[0].get("architecture"),
                    "train_setting": g_model.iloc[0].get("train_setting"),
                    "metric": metric,
                    "mean_additive": safe_nanmean(x) if len(x) else np.nan,
                    "mean_substitutive": safe_nanmean(y) if len(y) else np.nan,
                    "positive_means": "additive higher than substitutive; check metric_direction",
                    "metric_direction": "lower_is_better" if metric in LOWER_IS_BETTER_METRICS else "higher_is_better",
                })
                rows.append(res)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_p_holm"] = holm_bonferroni(out["paired_t_p"].tolist())
        out["wilcoxon_p_holm"] = holm_bonferroni(out["wilcoxon_p"].tolist())
    return out


def retrieval_summary(df: pd.DataFrame) -> pd.DataFrame:
    prefixes = ["context_", "retrieval_"]
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in prefixes) and any(x in c for x in ["hit@", "precision@", "recall@", "ndcg@", "mrr"]):
            cols.append(c)
    group_cols = ["condition", "test_file", "train_setting", "model_key"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(g))
        for c in sorted(cols):
            vals = numeric_series(g[c])
            if len(vals):
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def truncation_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["architecture", "train_setting", "model_key", "condition", "test_file"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(g))
        for c in ["source_tokens", "target_tokens", "prediction_tokens",
                  "source_tokens_model_tokenizer", "target_tokens_model_tokenizer", "prediction_tokens_model_tokenizer",
                  "source_words", "target_words", "prediction_words"]:
            vals = numeric_series(g[c])
            row[f"{c}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_p95"] = float(np.quantile(vals.to_numpy(dtype=float), 0.95)) if len(vals) else np.nan
            row[f"{c}_max"] = float(vals.max()) if len(vals) else np.nan
        row["source_truncated_risk_n"] = int(g["source_truncated_risk"].sum())
        row["source_truncated_risk_rate"] = float(g["source_truncated_risk"].mean())
        row["target_truncated_risk_n"] = int(g["target_truncated_risk"].sum())
        row["target_truncated_risk_rate"] = float(g["target_truncated_risk"].mean())
        if "source_truncated_risk_model_tokenizer" in g.columns:
            row["source_truncated_risk_model_tokenizer_n"] = int(g["source_truncated_risk_model_tokenizer"].sum())
            row["source_truncated_risk_model_tokenizer_rate"] = float(g["source_truncated_risk_model_tokenizer"].mean())
        if "target_truncated_risk_model_tokenizer" in g.columns:
            row["target_truncated_risk_model_tokenizer_n"] = int(g["target_truncated_risk_model_tokenizer"].sum())
            row["target_truncated_risk_model_tokenizer_rate"] = float(g["target_truncated_risk_model_tokenizer"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    out_root = Path(args.out_root) if args.out_root else None

    def path_of(sub: str, explicit: Optional[str]) -> str:
        if explicit:
            return explicit
        if out_root is None:
            raise ValueError(f"Missing model path for {sub}. Provide --out_root or explicit path.")
        return str(out_root / sub)

    return [
        ModelSpec("bart_base_noiseaware", "BART-base", "noiseaware", path_of("01_bart_base_noiseaware", args.bart_noise_dir)),
        ModelSpec("bart_base_clean_matched", "BART-base", "clean_matched", path_of("02_bart_base_clean_matched", args.bart_clean_dir)),
        ModelSpec("t5_base_noiseaware", "T5-base", "noiseaware", path_of("03_t5_base_noiseaware", args.t5_noise_dir)),
        ModelSpec("t5_base_clean_matched", "T5-base", "clean_matched", path_of("04_t5_base_clean_matched", args.t5_clean_dir)),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute rank-B metrics for BART/T5 noise-aware RAG summarization.")

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--out_root", type=str, default=None)

    p.add_argument("--bart_noise_dir", type=str, default=None)
    p.add_argument("--bart_clean_dir", type=str, default=None)
    p.add_argument("--t5_noise_dir", type=str, default=None)
    p.add_argument("--t5_clean_dir", type=str, default=None)

    p.add_argument("--test_files", nargs="*", default=DEFAULT_TEST_FILES)
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--max_source_length", type=int, default=1024)
    p.add_argument("--max_target_length", type=int, default=512)
    p.add_argument("--generation_max_length", type=int, default=320)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)

    p.add_argument("--bertscore_model", type=str, default="roberta-large")
    p.add_argument("--bertscore_batch_size", type=int, default=16)
    p.add_argument("--bertscore_lang", type=str, default="en")
    p.add_argument("--bertscore_rescale", action="store_true")
    p.add_argument("--bertscore_max_length", type=int, default=512)
    p.add_argument("--skip_bertscore", action="store_true")

    p.add_argument("--length_tokenizer", type=str, default="facebook/bart-base")
    p.add_argument("--retrieval_k_values", nargs="*", type=int, default=[1, 3, 5, 10])

    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--overwrite_predictions", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Precision loss.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*range zero.*")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model_specs = build_model_specs(args)

    eval_config = vars(args).copy()
    eval_config["model_specs"] = [m.__dict__ for m in model_specs]
    with (output_dir / "eval_config_rankB.json").open("w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=2, ensure_ascii=False)

    # Generate predictions
    if not args.skip_generation:
        for spec in model_specs:
            if not Path(spec.path).exists():
                raise FileNotFoundError(f"Model directory not found: {spec.path}")
            generate_predictions_for_model(
                model_spec=spec,
                data_dir=data_dir,
                test_files=args.test_files,
                pred_dir=pred_dir,
                max_source_length=args.max_source_length,
                generation_max_length=args.generation_max_length,
                num_beams=args.num_beams,
                batch_size=args.batch_size,
                limit=args.limit,
                overwrite=args.overwrite_predictions,
            )

    # Load predictions
    all_rows: List[Dict[str, Any]] = []
    for spec in model_specs:
        for test_file in args.test_files:
            pred_path = pred_dir / f"{spec.key}__{Path(test_file).stem}.jsonl"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction file: {pred_path}")
            all_rows.extend(read_jsonl(pred_path))

    print(f"\n[metrics] total rows={len(all_rows):,}")

    # Compute metrics
    rows = compute_rouge(all_rows)

    if not args.skip_bertscore:
        rows = add_bertscore(
            rows,
            bertscore_model=args.bertscore_model,
            bertscore_batch_size=args.bertscore_batch_size,
            lang=args.bertscore_lang,
            device="cuda" if torch.cuda.is_available() else "cpu",
            rescale_with_baseline=args.bertscore_rescale,
            bertscore_max_length=args.bertscore_max_length,
        )
    else:
        for r in rows:
            r["bertscore_p"] = np.nan
            r["bertscore_r"] = np.nan
            r["bertscore_f1"] = np.nan

    rows = add_rankB_record_metrics(
        rows,
        length_tokenizer=args.length_tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        retrieval_k_values=args.retrieval_k_values,
    )

    record_df = pd.DataFrame(rows)
    record_path = output_dir / "rankB_metrics_record_level.csv"
    record_df.to_csv(record_path, index=False)

    # Metrics included in summary/stat tests
    metrics = [m for m in CORE_METRICS if m in record_df.columns]
    # Add retrieval metrics dynamically
    for c in record_df.columns:
        if any(x in c for x in ["mrr", "ndcg@", "precision@", "recall@", "hit@"]):
            if c.startswith("context_") or c.startswith("retrieval_"):
                metrics.append(c)
    metrics = list(dict.fromkeys(metrics))

    summary_df = summary_tables(record_df, metrics)
    pair_df = compare_noiseaware_vs_clean(record_df, metrics)
    robust_df = robustness_degradation(record_df, metrics)
    add_sub_df = additive_vs_substitutive(record_df, metrics)
    retr_df = retrieval_summary(record_df)
    trunc_df = truncation_summary(record_df)

    summary_df.to_csv(output_dir / "rankB_metrics_summary.csv", index=False)
    pair_df.to_csv(output_dir / "rankB_paired_noiseaware_vs_clean.csv", index=False)
    robust_df.to_csv(output_dir / "rankB_robustness_degradation.csv", index=False)
    add_sub_df.to_csv(output_dir / "rankB_additive_vs_substitutive.csv", index=False)
    retr_df.to_csv(output_dir / "rankB_retrieval_summary.csv", index=False)
    trunc_df.to_csv(output_dir / "rankB_truncation_summary.csv", index=False)

    # Compact paper table: core metrics means only
    compact_cols = ["architecture", "train_setting", "model_key", "condition", "test_file", "n"]
    for m in ["rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1",
              "entity_f1_ref", "source_supported_entity_rate", "unsupported_entity_rate",
              "number_f1_ref", "unsupported_number_rate",
              "source_token_support_rate", "sentence_support_proxy",
              "distinct_1", "distinct_2", "repetition_2gram_rate"]:
        col = f"{m}_mean"
        if col in summary_df.columns:
            compact_cols.append(col)
    compact_df = summary_df[[c for c in compact_cols if c in summary_df.columns]].copy()
    compact_df.to_csv(output_dir / "rankB_compact_paper_table.csv", index=False)

    xlsx = output_dir / "rankB_all_metrics_tables.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        compact_df.to_excel(writer, sheet_name="compact_paper_table", index=False)
        summary_df.to_excel(writer, sheet_name="summary_all_metrics", index=False)
        pair_df.to_excel(writer, sheet_name="noise_vs_clean_tests", index=False)
        robust_df.to_excel(writer, sheet_name="robustness", index=False)
        add_sub_df.to_excel(writer, sheet_name="add_vs_sub", index=False)
        retr_df.to_excel(writer, sheet_name="retrieval", index=False)
        trunc_df.to_excel(writer, sheet_name="truncation", index=False)
        record_df.head(50000).to_excel(writer, sheet_name="record_level_head", index=False)

    print("\nDONE")
    print(f"  Record-level metrics : {record_path}")
    print(f"  Summary              : {output_dir / 'rankB_metrics_summary.csv'}")
    print(f"  Noise vs clean tests : {output_dir / 'rankB_paired_noiseaware_vs_clean.csv'}")
    print(f"  Robustness           : {output_dir / 'rankB_robustness_degradation.csv'}")
    print(f"  Additive vs substit. : {output_dir / 'rankB_additive_vs_substitutive.csv'}")
    print(f"  Retrieval summary    : {output_dir / 'rankB_retrieval_summary.csv'}")
    print(f"  Truncation summary   : {output_dir / 'rankB_truncation_summary.csv'}")
    print(f"  Excel workbook       : {xlsx}")


if __name__ == "__main__":
    main()
