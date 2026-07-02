import os
import json
import random
import argparse
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from datasets import load_from_disk

from rulebase_chunkforpdf import process_document
from retrieval_tokenizer import DenseEncoder, Document
from summarized import T5Summarizer

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ARXIV_DIR = "./dataset/arxiv"

DEFAULT_QUERY_TEMPLATES = [
    "Summarize the main contribution of this paper",
    "Summarize the proposed method and main findings of this paper",
    "What problem does this paper address and how does it solve it?",
]

# K values saved into every JSONL record for retrieval/context evaluation.
# You can change this list if you need other cutoffs, for example [1, 3, 5].
DEFAULT_RETRIEVAL_K_VALUES = [1, 3, 5, 10]

# Robust fallback chunking.
# If rulebase_chunkforpdf.process_document returns too few chunks,
# re-chunk the raw article with a simple word-level sliding window.
# This avoids losing almost all papers at the min_chunks=3 filter.
FALLBACK_MAX_CHUNK_WORDS = 150
FALLBACK_OVERLAP_WORDS = 30
FALLBACK_MIN_CHUNK_WORDS = 40

# Target/reference summary filter. This is important for BART/LED training:
# arXiv abstracts should normally be short summaries. Extremely long targets
# usually indicate that full article text leaked into target_text.
DEFAULT_MIN_TARGET_WORDS = 20
DEFAULT_MAX_TARGET_WORDS = 512

# Hard cap each context chunk before writing input_text.
# This prevents one bad section-aware chunk from creating 10k+ word inputs.
MAX_CONTEXT_CHUNK_WORDS = 180


def _word_count_simple(text: Any) -> int:
    if text is None:
        return 0
    return len(str(text).replace("\n", " ").split())


def filter_papers_by_target_length(
    papers: List[Dict[str, str]],
    split_name: str,
    min_target_words: int = DEFAULT_MIN_TARGET_WORDS,
    max_target_words: int = DEFAULT_MAX_TARGET_WORDS,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Remove empty, too-short, and abnormally long abstract targets.

    The same filter is applied before selecting train/validation/test targets and
    before constructing held-out distractor pools. This keeps BART and LED
    comparable because both models see the same cleaned JSONL rows.
    """
    kept: List[Dict[str, str]] = []
    dropped_short = 0
    dropped_long = 0
    dropped_empty_article = 0
    dropped_empty_target = 0
    lengths_kept: List[int] = []
    lengths_dropped: List[int] = []

    for paper in papers:
        article = str(paper.get("article", "")).strip()
        abstract = str(paper.get("abstract", "")).strip()
        tw = _word_count_simple(abstract)

        if not article:
            dropped_empty_article += 1
            lengths_dropped.append(tw)
            continue
        if not abstract:
            dropped_empty_target += 1
            lengths_dropped.append(tw)
            continue
        if tw < int(min_target_words):
            dropped_short += 1
            lengths_dropped.append(tw)
            continue
        if tw > int(max_target_words):
            dropped_long += 1
            lengths_dropped.append(tw)
            continue

        kept.append(paper)
        lengths_kept.append(tw)

    def _stats(vals: List[int]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0, "min": None, "mean": None, "max": None}
        return {
            "n": len(vals),
            "min": int(min(vals)),
            "mean": float(sum(vals) / len(vals)),
            "max": int(max(vals)),
        }

    meta = {
        "split": split_name,
        "input_records": len(papers),
        "kept_records": len(kept),
        "dropped_records": len(papers) - len(kept),
        "min_target_words": int(min_target_words),
        "max_target_words": int(max_target_words),
        "dropped_empty_article": int(dropped_empty_article),
        "dropped_empty_target": int(dropped_empty_target),
        "dropped_too_short": int(dropped_short),
        "dropped_too_long": int(dropped_long),
        "kept_target_word_stats": _stats(lengths_kept),
        "dropped_target_word_stats": _stats(lengths_dropped),
    }
    return kept, meta


def write_dataset_manifest(output_dir: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dataset_build_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def validate_jsonl_target_lengths(path: str) -> Dict[str, Any]:
    """Quick post-write check for JSONL rows used by BART/LED training."""
    n = 0
    empty_input = 0
    empty_target = 0
    lens: List[int] = []
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            inp = str(r.get("input_text", "")).strip()
            tar = str(r.get("target_text", "")).strip()
            n += 1
            if not inp:
                empty_input += 1
            if not tar:
                empty_target += 1
            lens.append(_word_count_simple(tar))
    return {
        "path": path,
        "exists": True,
        "n": n,
        "empty_input": empty_input,
        "empty_target": empty_target,
        "target_words_min": int(min(lens)) if lens else None,
        "target_words_mean": float(sum(lens) / len(lens)) if lens else None,
        "target_words_max": int(max(lens)) if lens else None,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _clean_field(x: Any, join_with: str = "\n") -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return join_with.join(str(i).strip() for i in x if str(i).strip())
    return str(x).replace("\\n", "\n").replace("/n", "\n").strip()


def load_pubmed_txt(path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            article, abstract = parts[0].strip(), parts[1].strip()
            if article and abstract:
                records.append({"article": article, "abstract": abstract})
    return records


def load_arxiv_arrow(split_dir: str) -> List[Dict[str, str]]:
    split_info = os.path.join(split_dir, "dataset_info.json")
    if os.path.isfile(split_info):
        ds = load_from_disk(split_dir)
    else:
        split_name = os.path.basename(split_dir.rstrip("/\\"))
        root_dir   = os.path.dirname(split_dir.rstrip("/\\"))
        dataset    = load_from_disk(root_dir)
        from datasets import DatasetDict
        ds = dataset[split_name] if isinstance(dataset, DatasetDict) else dataset

    records: List[Dict[str, str]] = []
    for sample in ds:
        article  = _clean_field(sample.get("article",  ""), join_with="\n")
        abstract = _clean_field(sample.get("abstract", ""), join_with=" ")
        if article and abstract:
            records.append({"article": article, "abstract": abstract})
    return records


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def _truncate_words(text: Any, max_words: int = MAX_CONTEXT_CHUNK_WORDS) -> str:
    """Keep generated JSONL inputs bounded for models without safe truncation."""
    words = str(text or "").replace("\n", " ").split()
    if max_words is not None and max_words > 0 and len(words) > max_words:
        return " ".join(words[:max_words])
    return " ".join(words)


def make_documents_from_chunks(chunks: List[Dict[str, Any]]) -> List[Document]:
    documents: List[Document] = []
    for c in chunks:
        raw_text = c.get("text", "")
        text = _truncate_words(raw_text, MAX_CONTEXT_CHUNK_WORDS)
        if not text.strip():
            continue

        metadata = dict(c)
        original_wc = _word_count_simple(raw_text)
        metadata["original_word_count"] = int(original_wc)
        metadata["word_count"] = int(_word_count_simple(text))
        metadata["truncated_for_model"] = bool(original_wc > MAX_CONTEXT_CHUNK_WORDS)

        documents.append(
            Document(
                # globally unique id = source_doc_id + chunk_id to avoid cross-paper collisions
                id=f"{c.get('source_doc_id', 'unk')}_{c['chunk_id']}",
                text=text,
                metadata=metadata,
            )
        )
    return documents

def _get_source_doc_id(doc: Document) -> Optional[str]:
    if not hasattr(doc, "metadata") or doc.metadata is None:
        return None
    return doc.metadata.get("source_doc_id")


def _count_unique_source_docs(docs: List[Document]) -> int:
    source_ids = {
        _get_source_doc_id(doc)
        for doc in docs
        if _get_source_doc_id(doc) is not None
    }
    return len(source_ids)


def _get_doc_id(doc: Document) -> str:
    return str(getattr(doc, "id", ""))


def _get_relevant_chunk_ids(
    documents: List[Document],
    source_doc_id: str,
) -> List[str]:
    """Gold relevant chunks for one query/paper.

    In this dataset builder, the paper abstract is the target summary.
    Therefore, every chunk whose ``source_doc_id`` equals the current
    ``paper_id`` is treated as relevant for retrieval evaluation.
    """
    return [
        _get_doc_id(doc)
        for doc in documents
        if _get_source_doc_id(doc) == source_doc_id
    ]


def compute_precision_recall_at_k(
    ranked_docs: List[Document],
    relevant_chunk_ids: List[str],
    k_values: Optional[List[int]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Compute hits@K, precision@K and recall@K from ranked documents.

    Notes
    -----
    - ``ranked_docs`` must already be ordered from most relevant to least relevant.
    - If fewer than K documents are available, precision is divided by the
      number of actually available top-K documents to avoid artificial zeros
      on short papers.
    - Recall is divided by the number of gold relevant chunks.
    """
    k_values = k_values or DEFAULT_RETRIEVAL_K_VALUES
    relevant_set = set(relevant_chunk_ids)
    ranked_ids = [_get_doc_id(doc) for doc in ranked_docs]
    total_relevant = len(relevant_set)

    metrics: Dict[str, Any] = {}
    for k in k_values:
        top_k = ranked_ids[:k]
        retrieved_at_k = len(top_k)
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)

        metrics[f"{prefix}retrieved@{k}"] = retrieved_at_k
        metrics[f"{prefix}hits@{k}"] = hits
        metrics[f"{prefix}precision@{k}"] = hits / max(retrieved_at_k, 1)
        metrics[f"{prefix}recall@{k}"] = hits / max(total_relevant, 1)

    return metrics


def build_retrieval_eval_payload(
    retrieved_docs: List[Document],
    retrieved_indices: List[int],
    all_scores: np.ndarray,
    relevant_chunk_ids: List[str],
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Metadata and metrics for the retriever output before noise injection."""
    scores_arr = np.asarray(np.atleast_1d(all_scores), dtype=np.float32)
    relevant_set = set(relevant_chunk_ids)

    retrieved_scores: List[Optional[float]] = []
    for idx in retrieved_indices:
        if 0 <= int(idx) < len(scores_arr):
            retrieved_scores.append(float(scores_arr[int(idx)]))
        else:
            retrieved_scores.append(None)

    payload: Dict[str, Any] = {
        "retrieved_chunk_ids": [_get_doc_id(doc) for doc in retrieved_docs],
        "retrieved_source_doc_ids": [_get_source_doc_id(doc) for doc in retrieved_docs],
        "retrieved_scores": retrieved_scores,
        "retrieved_hit_flags": [
            1 if _get_doc_id(doc) in relevant_set else 0
            for doc in retrieved_docs
        ],
        "num_retrieved_chunks": len(retrieved_docs),
        "num_relevant_chunks": len(relevant_chunk_ids),
        "relevant_chunk_ids": relevant_chunk_ids,
    }
    payload.update(
        compute_precision_recall_at_k(
            ranked_docs=retrieved_docs,
            relevant_chunk_ids=relevant_chunk_ids,
            k_values=k_values,
            prefix="retrieval_",
        )
    )
    return payload


def build_context_eval_payload(
    context_docs: List[Document],
    relevant_chunk_ids: List[str],
    k_values: Optional[List[int]] = None,
    include_unprefixed_metrics: bool = True,
) -> Dict[str, Any]:
    """Metadata and metrics for the final contexts fed into the summarizer.

    For clean samples, this is the same as the retrieved context.
    For noisy samples, this includes retrieved chunks plus injected noise chunks,
    in the actual order used in ``input_text``.
    """
    relevant_set = set(relevant_chunk_ids)
    payload: Dict[str, Any] = {
        "context_chunk_ids": [_get_doc_id(doc) for doc in context_docs],
        "context_source_doc_ids": [_get_source_doc_id(doc) for doc in context_docs],
        "context_hit_flags": [
            1 if _get_doc_id(doc) in relevant_set else 0
            for doc in context_docs
        ],
        "num_context_chunks_eval": len(context_docs),
    }

    # Prefixed metrics describe the final context actually passed to the model.
    context_metrics = compute_precision_recall_at_k(
        ranked_docs=context_docs,
        relevant_chunk_ids=relevant_chunk_ids,
        k_values=k_values,
        prefix="context_",
    )
    payload.update(context_metrics)

    # Unprefixed aliases make quick analysis easier when reading JSONL.
    if include_unprefixed_metrics:
        payload.update(
            compute_precision_recall_at_k(
                ranked_docs=context_docs,
                relevant_chunk_ids=relevant_chunk_ids,
                k_values=k_values,
                prefix="",
            )
        )

    return payload


def choose_query(paper_id: str, use_multiple: bool = True) -> str:
    if not use_multiple:
        return DEFAULT_QUERY_TEMPLATES[0]
    return random.Random(paper_id).choice(DEFAULT_QUERY_TEMPLATES)


# ---------------------------------------------------------------------------
# Lightweight example builder (no tokenizer/model dependency)
# ---------------------------------------------------------------------------
SAFE_MAX_CONTEXT_WORDS_EACH = 180
SAFE_MAX_INPUT_WORDS = 950
SAFE_MAX_TARGET_WORDS = 512


def _truncate_words(text: Any, max_words: int) -> str:
    """Whitespace word-level truncation for portable JSONL building.

    The dataset builder should not depend on a seq2seq tokenizer/model. Actual
    tokenization is done later by the training script for BART/PEGASUS/T5/LED/etc.
    This function keeps the JSONL safe enough so downstream tokenizers are not
    forced to handle extreme outliers.
    """
    clean = str(text or "").replace("\\n", " ").replace("/n", " ")
    clean = " ".join(clean.split())
    if not clean:
        return ""
    words = clean.split()
    if len(words) > int(max_words):
        words = words[: int(max_words)]
    return " ".join(words).strip()


def build_training_example_safe(
    query: str,
    contexts: List[str],
    target_text: str,
    max_contexts: Optional[int] = None,
    task_prefix: str = "summarize",
    max_context_words_each: int = SAFE_MAX_CONTEXT_WORDS_EACH,
    max_input_words: int = SAFE_MAX_INPUT_WORDS,
    max_target_words: int = SAFE_MAX_TARGET_WORDS,
) -> Dict[str, str]:
    """Build {input_text, target_text} without loading T5/BART/PEGASUS.

    This fixes the zero-row failure caused when ``summarizer.build_training_example``
    throws inside the assembly loop and the exception is swallowed. It also makes
    the generated JSONL model-agnostic: BART, PEGASUS, T5, LED, and causal-LM
    conversion scripts can tokenize later with their own limits.
    """
    q = _truncate_words(query, 80)
    selected = list(contexts or [])
    if max_contexts is not None:
        selected = selected[: max(int(max_contexts), 0)]
    clean_contexts = [
        _truncate_words(c, max_context_words_each)
        for c in selected
        if _truncate_words(c, max_context_words_each)
    ]
    target = _truncate_words(target_text, max_target_words)
    if not target:
        raise ValueError("target_text is empty after cleaning/truncation")
    if not clean_contexts:
        raise ValueError("contexts are empty after cleaning/truncation")
    source = f"{task_prefix}: question: {q} context: " + " ".join(clean_contexts)
    source = _truncate_words(source, max_input_words)
    if not source:
        raise ValueError("input_text is empty after cleaning/truncation")
    return {"input_text": source, "target_text": target}



# ---------------------------------------------------------------------------
# Robust fallback chunking
# ---------------------------------------------------------------------------

def _simple_word_window_chunks(
    text: str,
    source_doc_id: str,
    max_words: int = FALLBACK_MAX_CHUNK_WORDS,
    overlap_words: int = FALLBACK_OVERLAP_WORDS,
    min_words: int = FALLBACK_MIN_CHUNK_WORDS,
) -> List[Dict[str, Any]]:
    """
    Fallback chunker used when section-aware chunking returns too few chunks.

    This keeps the experiment stable on arXiv long documents. If the rule-based
    parser fails to detect sections or returns only 1-2 chunks for a long paper,
    this function creates multiple chunks using a sliding word window.
    """
    if not text or not text.strip():
        return []

    words = text.replace("\n", " ").split()
    if not words:
        return []

    step = max(1, max_words - overlap_words)
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for start in range(0, len(words), step):
        end = min(start + max_words, len(words))
        window = words[start:end]

        if len(window) < min_words and chunks:
            break

        chunk_text = " ".join(window).strip()
        if not chunk_text:
            continue

        chunks.append({
            "chunk_id": chunk_id,
            "section": "Body",
            "text": chunk_text,
            "word_count": len(window),
            "section_index": 0,
            "chunk_index_in_section": chunk_id,
            "source_doc_id": source_doc_id,
            "chunking_method": "fallback_word_window",
        })
        chunk_id += 1

        if end >= len(words):
            break

    return chunks


def _need_fallback_chunking(
    chunks: List[Dict[str, Any]],
    article: str,
    min_required_chunks: int = 3,
) -> bool:
    """
    Use fallback only when a long article receives too few chunks.
    """
    if len(chunks) >= min_required_chunks:
        return False

    n_words = len(article.replace("\n", " ").split())
    approx_needed_words = FALLBACK_MIN_CHUNK_WORDS * min_required_chunks
    return n_words >= approx_needed_words

# ---------------------------------------------------------------------------
# Stage 1 — CPU-parallel chunking
# ---------------------------------------------------------------------------

def _chunk_one_paper(task: Tuple[int, Dict[str, str], str]) -> Optional[Dict[str, Any]]:
    idx, paper, split_name = task
    article  = paper.get("article", "")
    abstract = paper.get("abstract", "")

    if not article or not abstract:
        print(f"[DEBUG] skip {split_name}_{idx}: empty article/abstract")
        return None

    source_doc_id = f"{split_name}_{idx}"

    try:
        processed = process_document(article, source_doc_id=source_doc_id)
        chunks = processed.get("chunks", [])

        # Important fix:
        # If rule-based section-aware chunking returns fewer than 3 chunks for a
        # long article, do not let the later min_chunks=3 filter remove the paper.
        # Re-chunk the raw article with a robust sliding word window instead.
        if _need_fallback_chunking(chunks, article, min_required_chunks=3):
            fallback_chunks = _simple_word_window_chunks(
                article,
                source_doc_id=source_doc_id,
            )
            if len(fallback_chunks) > len(chunks):
                chunks = fallback_chunks

        if not chunks:
            print(f"[DEBUG] no chunks for {source_doc_id} | article_len={len(article)}")
            return None

        return {
            "paper_id": source_doc_id,
            "abstract": abstract,
            "chunks": chunks,
        }

    except Exception as e:
        print(f"[DEBUG] chunk error {source_doc_id}: {e}")

        # Last-resort fallback: even if process_document crashes, still try to
        # build chunks from the raw article. This prevents losing long papers.
        fallback_chunks = _simple_word_window_chunks(
            article,
            source_doc_id=source_doc_id,
        )
        if fallback_chunks:
            return {
                "paper_id": source_doc_id,
                "abstract": abstract,
                "chunks": fallback_chunks,
            }

        return None

def batch_chunk_papers(
    papers:      List[Dict[str, str]],
    split_name:  str,
    num_workers: int = 4,
    limit:       Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Chunk all papers in parallel using threads.
    ThreadPoolExecutor (not Process) avoids Windows spawn overhead
    where each process would re-import torch + transformers (~30-60s).
    Chunking is I/O + regex heavy → threads work well here.
    """
    papers = papers[:limit] if limit is not None else papers
    tasks  = [(idx, p, split_name) for idx, p in enumerate(papers)]

    print(f"  [chunk:{split_name}] {len(tasks)} papers  |  workers={num_workers}")
    t0 = time.time()

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_chunk_one_paper, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"  chunk [{split_name}]", unit="paper"):
            res = fut.result()
            if res is not None:
                results.append(res)

    results.sort(key=lambda r: int(r["paper_id"].split("_")[-1]))

    # Diagnostic summary: helps detect chunking collapse early.
    if results:
        chunk_counts = [len(r.get("chunks", [])) for r in results]
        n_ge3 = sum(1 for x in chunk_counts if x >= 3)
        n_ge1 = sum(1 for x in chunk_counts if x >= 1)
        print(
            f"  [chunk:{split_name}] chunk-count summary | "
            f"min={min(chunk_counts)} median={float(np.median(chunk_counts)):.1f} "
            f"mean={float(np.mean(chunk_counts)):.1f} max={max(chunk_counts)} "
            f"| >=1={n_ge1} >=3={n_ge3}"
        )

    print(f"  [chunk:{split_name}] {len(results)}/{len(tasks)} valid  "
          f"| {time.time()-t0:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Stage 2 — GPU batch encoding  (VRAM-safe for 6GB cards)
# ---------------------------------------------------------------------------

def batch_encode_all_chunks(
    chunked_papers:  List[Dict[str, Any]],
    encoder:         DenseEncoder,
    paper_batch:     int = 500,   # encode this many papers at a time
                                  # 500 papers × ~15 chunks × 384-dim ≈ 400MB RAM
) -> Dict[str, Dict[str, Any]]:
    """
    Encode chunks in groups of `paper_batch` papers to keep
    CPU RAM and GPU VRAM under control on 6 GB cards.

    Returns:
        { paper_id: {"chunks": [...], "embeddings": np.ndarray, "abstract": str} }
    """
    result: Dict[str, Dict[str, Any]] = {}
    total  = len(chunked_papers)

    for batch_start in range(0, total, paper_batch):
        group = chunked_papers[batch_start : batch_start + paper_batch]

        # Flatten texts for this group
        all_texts:    List[str]                = []
        paper_slices: Dict[str, Tuple[int,int]] = {}

        for cp in group:
            pid   = cp["paper_id"]
            texts = [c["text"] for c in cp["chunks"]]
            s     = len(all_texts)
            all_texts.extend(texts)
            paper_slices[pid] = (s, s + len(texts))

        batch_end = min(batch_start + paper_batch, total)
        print(f"  [encode] papers {batch_start+1}-{batch_end}/{total} "
              f"| {len(all_texts)} chunks", flush=True)

        all_embs = encoder.encode(all_texts)   # GPU batched

        # Redistribute embeddings back
        for cp in group:
            pid  = cp["paper_id"]
            s, e = paper_slices[pid]
            result[pid] = {
                "chunks":     cp["chunks"],
                "embeddings": all_embs[s:e],
                "abstract":   cp["abstract"],
            }

        # Free intermediate arrays explicitly
        del all_texts, all_embs, paper_slices

    return result


# ---------------------------------------------------------------------------
# Stage 3 helpers — per-paper retrieval using pre-built embeddings
# ---------------------------------------------------------------------------

class _NumpyIndexFlatIP:
    """Small FAISS-compatible fallback for laptops where faiss-cpu is unavailable.

    It is slower than FAISS but good enough for per-paper retrieval because each
    paper only has a limited number of chunks.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.embeddings: Optional[np.ndarray] = None

    def add(self, embeddings: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
        q = np.asarray(query_emb, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        scores = (self.embeddings @ q.T).squeeze()
        scores = np.asarray(np.atleast_1d(scores), dtype=np.float32)
        k = min(int(k), len(scores))
        idxs = np.argsort(-scores)[:k].astype(np.int64)
        return scores[idxs].reshape(1, -1), idxs.reshape(1, -1)


def _build_faiss_index(embeddings: np.ndarray) -> Any:
    dim = embeddings.shape[1]
    if faiss is not None:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index
    index = _NumpyIndexFlatIP(dim)
    index.add(embeddings)
    return index


def _mmr_select(
    q_emb:       np.ndarray,     # (1, D)
    cand_embs:   np.ndarray,     # (C, D)
    cand_scores: List[float],
    k:           int,
    mmr_lambda:  float = 0.5,
) -> List[int]:
    """Pure-numpy MMR — no encoder call needed."""
    sim_matrix = cand_embs @ cand_embs.T
    selected:  List[int] = []
    remaining: set       = set(range(len(cand_scores)))

    for _ in range(min(k, len(cand_scores))):
        best_pos, best_score = -1, float("-inf")
        for pos in remaining:
            rel = cand_scores[pos]
            red = float(np.max(sim_matrix[pos, selected])) if selected else 0.0
            score = mmr_lambda * rel - (1 - mmr_lambda) * red
            if score > best_score:
                best_score, best_pos = score, pos
        if best_pos == -1:
            break
        selected.append(best_pos)
        remaining.remove(best_pos)
    return selected


def retrieve_clean(
    query_emb:   np.ndarray,
    documents:   List[Document],
    doc_embs:    np.ndarray,
    index:       Any,
    final_k:     int,
    noise_k:     int,
    mmr_lambda:  float = 0.5,
) -> Tuple[List[Document], List[int], np.ndarray]:
    """
    Fast retrieval using pre-built FAISS index + numpy MMR.
    Returns (retrieved_docs, retrieved_indices, all_scores).
    all_scores: cosine similarity of ALL docs against the query — used for easy/hard noise selection.
    """
    num_docs    = len(documents)
    effective_k = min(final_k, num_docs)
    candidate_k = min(max(effective_k * 5, 50), num_docs)

    scores, idxs = index.search(query_emb, candidate_k)
    cand_indices = idxs[0].tolist()
    cand_scores  = scores[0].tolist()
    cand_embs    = doc_embs[cand_indices]

    selected_pos = _mmr_select(query_emb, cand_embs, cand_scores,
                               k=effective_k, mmr_lambda=mmr_lambda)

    retrieved_docs    = [documents[cand_indices[p]] for p in selected_pos]
    retrieved_indices = [cand_indices[p] for p in selected_pos]

    # Score all docs in this paper's corpus (used for easy/hard noise classification)
    all_scores = (doc_embs @ query_emb.T).squeeze()   # (N,)

    return retrieved_docs, retrieved_indices, all_scores


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------
def _select_noise_candidates_by_similarity(
    candidate_pool: List[Tuple[int, Document]],
    sims: np.ndarray,
    noise_mode: str,
) -> List[Tuple[int, Document]]:
    """
    Select cross-document noise candidates based on similarity to the query.

    Modes:
        - cross_doc_easy: choose semantically similar distractors
        - cross_doc_hard: choose semantically dissimilar distractors
    """
    sims = np.asarray(sims, dtype=np.float32)
    sims = np.atleast_1d(sims)
    sims = np.nan_to_num(sims, nan=-1.0)

    if len(candidate_pool) != len(sims):
        n = min(len(candidate_pool), len(sims))
        candidate_pool = candidate_pool[:n]
        sims = sims[:n]

    if len(candidate_pool) == 0:
        return []

    if noise_mode == "cross_doc_easy":
        # More confusing distractors: high similarity, but not the absolute top tail
        lo = float(np.percentile(sims, 70))
        hi = float(np.percentile(sims, 95))
        mask = (sims >= lo) & (sims <= hi)
        filtered = [candidate_pool[i] for i in range(len(candidate_pool)) if mask[i]]

        # Controlled relaxation if too few candidates remain
        if len(filtered) == 0:
            lo = float(np.percentile(sims, 60))
            hi = float(np.percentile(sims, 98))
            mask = (sims >= lo) & (sims <= hi)
            filtered = [candidate_pool[i] for i in range(len(candidate_pool)) if mask[i]]

        return filtered

    if noise_mode == "cross_doc_hard":
        # Truly irrelevant distractors: bottom similarity region
        threshold = float(np.percentile(sims, 10))
        mask = sims <= threshold
        filtered = [candidate_pool[i] for i in range(len(candidate_pool)) if mask[i]]

        # Controlled relaxation if too few candidates remain
        if len(filtered) == 0:
            threshold = float(np.percentile(sims, 20))
            mask = sims <= threshold
            filtered = [candidate_pool[i] for i in range(len(candidate_pool)) if mask[i]]

        return filtered

    return []

def make_noisy_context(
    clean_docs: List[Document],
    noise_k: int = 2,
    rng: Any = None,
    shuffle: bool = True,
    global_noise_pool: Optional[List[Document]] = None,
    global_pool_embs: Optional[np.ndarray] = None,
    query_emb: Optional[np.ndarray] = None,
    current_paper_id: Optional[str] = None,
    noise_mode: str = "cross_doc_easy",
) -> Dict[str, Any]:
    """
    Build a noisy context bundle by mixing retrieved clean chunks with
    cross-document noise chunks.

    Parameters
    ----------
    clean_docs:
        Retrieved clean chunks from the current paper.
    noise_k:
        Number of noise chunks to inject.
    rng:
        Random generator. Falls back to Python's random module if None.
    shuffle:
        Whether to shuffle clean and noise chunks after mixing.
    global_noise_pool:
        Cross-document chunk pool used as the source of distractors.
    global_pool_embs:
        Embeddings aligned with global_noise_pool.
    query_emb:
        Query embedding used to score similarity for easy/hard noise.
    current_paper_id:
        Source document id of the current paper. Used to exclude same-paper chunks.
    noise_mode:
        One of:
            - "cross_doc_easy": semantically similar distractors
            - "cross_doc_hard": semantically dissimilar distractors
            - "cross_document": random cross-document distractors

    Returns
    -------
    dict with:
        clean_contexts:
            List of original clean texts.
        noisy_contexts:
            List of mixed texts (clean + noise) if noise is available,
            otherwise the clean-only list.
        noise_docs:
            The actual sampled noise Document objects.
        has_true_noise:
            True only when real cross-document noise was added.
        noise_source:
            A short status label for debugging / analysis.
        n_noise:
            Number of injected noise chunks.

    Notes
    -----
    - The function never uses chunks from the same paper as noise.
    - Already retrieved clean chunks are excluded from the candidate pool.
    - Easy/hard modes do not fall back to fully random sampling, because that
      would break the intended noise label.
    - If no valid noise can be selected, the function safely returns clean-only output.
    """
    rng = rng or random

    clean_contexts = [doc.text for doc in clean_docs]
    clean_ids = {doc.id for doc in clean_docs}

    # Helper: standard clean-only fallback to keep the return structure consistent.
    def _clean_only_result(source: str) -> Dict[str, Any]:
        return {
            "clean_contexts": clean_contexts,
            "noisy_contexts": clean_contexts,
            "clean_docs": list(clean_docs),
            "noise_docs": [],
            "noisy_docs": list(clean_docs),
            "has_true_noise": False,
            "noise_source": source,
            "n_noise": 0,
        }

    # No global pool means no cross-document noise can be created.
    if global_noise_pool is None or len(global_noise_pool) == 0:
        return _clean_only_result("none")

    # Keep only valid cross-document candidates:
    # 1) exclude already retrieved clean chunks
    # 2) exclude chunks from the current paper
    candidate_pool: List[Tuple[int, Document]] = [
        (i, doc)
        for i, doc in enumerate(global_noise_pool)
        if doc.id not in clean_ids
        and _get_source_doc_id(doc) != current_paper_id
    ]

    if len(candidate_pool) == 0:
        return _clean_only_result("none")

    noise_docs: List[Document] = []
    noise_source = noise_mode

    if noise_mode in ("cross_doc_easy", "cross_doc_hard"):
        # Easy/hard modes require embeddings to compute similarity-based filtering.
        if global_pool_embs is None or query_emb is None:
            return _clean_only_result(f"{noise_mode}_missing_embeddings")

        pool_indices = [i for i, _ in candidate_pool]
        pool_embs = global_pool_embs[pool_indices]

        qe = query_emb.reshape(1, -1) if query_emb.ndim == 1 else query_emb
        sims = (pool_embs @ qe.T).squeeze()
        sims = np.asarray(np.atleast_1d(sims), dtype=np.float32)

        # Guard against invalid similarity output.
        if len(sims) == 0 or np.isnan(sims).all():
            return _clean_only_result(f"{noise_mode}_empty_similarity")

        # Select candidate band based on similarity profile.
        filtered = _select_noise_candidates_by_similarity(
            candidate_pool=candidate_pool,
            sims=sims,
            noise_mode=noise_mode,
        )

        if len(filtered) == 0:
            return _clean_only_result(f"{noise_mode}_insufficient_candidates")

        sampled = rng.sample(filtered, min(noise_k, len(filtered)))
        noise_docs = [doc for _, doc in sampled]

        # Mark partial injection if the filtered band is smaller than noise_k.
        if len(noise_docs) < noise_k:
            noise_source = f"{noise_mode}_partial"

    elif noise_mode == "cross_document":
        # Baseline random cross-document noise without similarity filtering.
        sampled = rng.sample(candidate_pool, min(noise_k, len(candidate_pool)))
        noise_docs = [doc for _, doc in sampled]

        if len(noise_docs) < noise_k:
            noise_source = "cross_document_partial"

    else:
        return _clean_only_result(f"invalid_noise_mode:{noise_mode}")

    # If no real noise was sampled, return clean-only output.
    if len(noise_docs) == 0:
        return _clean_only_result("none")

    # Merge clean chunks with noise chunks.
    noisy_docs = list(clean_docs) + noise_docs

    # Optional shuffle prevents the model from always seeing noise at the tail.
    if shuffle:
        rng.shuffle(noisy_docs)

    return {
        "clean_contexts": clean_contexts,
        "noisy_contexts": [doc.text for doc in noisy_docs],
        "clean_docs": list(clean_docs),
        "noise_docs": noise_docs,
        "noisy_docs": noisy_docs,
        "has_true_noise": True,
        "noise_source": noise_source,
        "n_noise": len(noise_docs),
    }
# ---------------------------------------------------------------------------
# Stage 3 — per-paper example assembly (fast, no encoder calls)
# ---------------------------------------------------------------------------

def _assemble_one_paper(
    pid: str,
    paper_data: Dict[str, Any],
    query_emb: np.ndarray,        # shape: (1, D)
    query: str,
    summarizer: T5Summarizer,
    split_name: str,
    final_k: int,
    noise_k: int,
    add_noisy: bool,
    noisy_prob: float,
    shuffle_noisy: bool,
    noise_mode: str,
    global_noise_pool: Optional[List[Document]],
    global_pool_embs: Optional[np.ndarray],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Assemble training/evaluation examples for a single paper.

    Output behavior
    ---------------
    - Always try to create one clean example.
    - Optionally create one noisy example, depending on `add_noisy`
      and `noisy_prob`.
    - The noisy example is created only when real cross-document noise
      is successfully injected.

    Notes
    -----
    - Clean retrieval is performed only once for this paper.
    - The noisy example reuses the same retrieved clean chunks and only
      appends sampled cross-document distractors.
    - A debug check is included to verify that noisy contexts actually
      contain more than one source document.
    """
    chunks = paper_data["chunks"]
    doc_embs = paper_data["embeddings"]
    abstract = paper_data["abstract"]

    documents = make_documents_from_chunks(chunks)
    if not documents:
        return []

    index = _build_faiss_index(doc_embs)

    retrieved_docs, retrieved_indices, all_scores = retrieve_clean(
        query_emb=query_emb,
        documents=documents,
        doc_embs=doc_embs,
        index=index,
        final_k=final_k,
        noise_k=noise_k if add_noisy else 0,
    )
    if not retrieved_docs:
        return []

    records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Build clean example
    # ------------------------------------------------------------------
    clean_contexts = [doc.text for doc in retrieved_docs]
    relevant_chunk_ids = _get_relevant_chunk_ids(documents, pid)
    retrieval_eval_payload = build_retrieval_eval_payload(
        retrieved_docs=retrieved_docs,
        retrieved_indices=retrieved_indices,
        all_scores=all_scores,
        relevant_chunk_ids=relevant_chunk_ids,
    )
    clean_context_eval_payload = build_context_eval_payload(
        context_docs=retrieved_docs,
        relevant_chunk_ids=relevant_chunk_ids,
    )

    clean_ex = build_training_example_safe(
        query=query,
        contexts=clean_contexts,
        target_text=abstract,
        max_contexts=final_k,
    )
    clean_ex.update({
        "paper_id": pid,
        "split": split_name,
        "sample_type": "clean",
        "noise_mode": "clean",
        "noise_source": "none",
        "num_noise_chunks": 0,
        "noise_chunk_ids": [],
        "noise_source_doc_ids": [],
        "num_contexts": len(clean_contexts),
        "num_chunks_total_in_paper": len(chunks),
        "num_documents_in_context": _count_unique_source_docs(retrieved_docs),
        **retrieval_eval_payload,
        **clean_context_eval_payload,
    })
    records.append(clean_ex)

    # ------------------------------------------------------------------
    # Optionally build noisy example
    # ------------------------------------------------------------------
    if add_noisy and rng.random() < noisy_prob:
        # For validation we usually want only ONE noisy file, but that file
        # should still represent both easy and hard retrieval noise.
        # mixed_easy_hard therefore creates one noisy example per paper and
        # randomly chooses either easy or hard noise for that example.
        actual_noise_mode = noise_mode
        if noise_mode == "mixed_easy_hard":
            actual_noise_mode = "cross_doc_hard" if rng.random() < 0.5 else "cross_doc_easy"

        bundle = make_noisy_context(
            clean_docs=retrieved_docs,
            noise_k=noise_k,
            rng=rng,
            shuffle=shuffle_noisy,
            global_noise_pool=global_noise_pool,
            global_pool_embs=global_pool_embs,
            query_emb=query_emb,
            current_paper_id=pid,
            noise_mode=actual_noise_mode,
        )

        if bundle["has_true_noise"]:
            noisy_ex = build_training_example_safe(
                query=query,
                contexts=bundle["noisy_contexts"],
                target_text=abstract,
                max_contexts=final_k + noise_k,
            )

            context_docs = list(bundle.get("noisy_docs", []))
            if not context_docs:
                context_docs = list(retrieved_docs) + list(bundle.get("noise_docs", []))
            noise_docs = list(bundle.get("noise_docs", []))
            unique_docs = _count_unique_source_docs(context_docs)
            noisy_context_eval_payload = build_context_eval_payload(
                context_docs=context_docs,
                relevant_chunk_ids=relevant_chunk_ids,
            )

            # Debug safeguard: a noisy example should contain at least
            # two distinct source documents.
            if unique_docs <= 1:
                print(f"[BUG] noisy but only 1 doc: {pid}")

            noisy_ex.update({
                "paper_id": pid,
                "split": split_name,
                "sample_type": "noisy",
                "num_contexts": len(bundle["noisy_contexts"]),
                "noise_mode": actual_noise_mode,
                "noise_source": bundle["noise_source"],
                "num_noise_chunks": bundle.get("n_noise", 0),
                "noise_chunk_ids": [_get_doc_id(doc) for doc in noise_docs],
                "noise_source_doc_ids": [_get_source_doc_id(doc) for doc in noise_docs],
                "num_chunks_total_in_paper": len(chunks),
                "num_documents_in_context": unique_docs,
                **retrieval_eval_payload,
                **noisy_context_eval_payload,
            })
            records.append(noisy_ex)
        else:
            print(f"[WARN] no noise for {pid} ({actual_noise_mode})")
    return records

# ---------------------------------------------------------------------------
# Full split builder — 3-stage
# ---------------------------------------------------------------------------

def build_split_fast(
    papers:               List[Dict[str, str]],
    split_name:           str,
    encoder:              DenseEncoder,
    summarizer:           T5Summarizer,
    limit:                Optional[int],
    final_k:              int,
    noise_k:              int,
    min_chunks:           int,
    use_multiple_queries: bool,
    add_noisy:            bool,
    noisy_prob:           float,
    shuffle_noisy:        bool,
    noise_mode:           str,
    global_noise_pool:    Optional[List[Document]],
    global_pool_embs:     Optional[np.ndarray],
    rng:                  random.Random,
    num_chunk_workers:    int = 4,
    paper_batch:          int = 500,
) -> List[Dict[str, Any]]:

    print(f"\n── {split_name.upper()} [{noise_mode}] ──────────────────────")

    # 1) Chunk papers
    chunked = batch_chunk_papers(
        papers,
        split_name,
        num_workers=num_chunk_workers,
        limit=limit,
    )
    print(f"  [{split_name}] chunked papers before filter: {len(chunked)}")

    # 2) Filter papers by minimum number of chunks
    chunked = [c for c in chunked if len(c.get("chunks", [])) >= min_chunks]
    print(f"  [{split_name}] papers after filter(min_chunks={min_chunks}): {len(chunked)}")

    if not chunked:
        print(f"  [{split_name}] no valid papers after chunk filtering")
        return []

    # 3) Encode all chunks
    paper_map = batch_encode_all_chunks(
        chunked,
        encoder,
        paper_batch=paper_batch,
    )

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 4) Cache query embeddings
    query_emb_cache = {q: encoder.encode(q) for q in DEFAULT_QUERY_TEMPLATES}

    # 5) Assemble examples
    records: List[Dict[str, Any]] = []
    total = len(paper_map)

    for i, (pid, pdata) in enumerate(
        tqdm(paper_map.items(), desc=f"  assemble [{split_name}]", unit="paper")
    ):
        try:
            query = choose_query(pid, use_multiple_queries)
            query_emb = query_emb_cache[query]

            examples = _assemble_one_paper(
                pid=pid,
                paper_data=pdata,
                query_emb=query_emb,
                query=query,
                summarizer=summarizer,
                split_name=split_name,
                final_k=final_k,
                noise_k=noise_k if add_noisy else 0,
                add_noisy=add_noisy,
                noisy_prob=noisy_prob,
                shuffle_noisy=shuffle_noisy,
                noise_mode=noise_mode,
                global_noise_pool=global_noise_pool,
                global_pool_embs=global_pool_embs,
                rng=rng,
            )
            records.extend(examples)

        except Exception as e:
            print(f"[WARN] {split_name}::{pid}: {type(e).__name__}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  [{split_name}] {i+1}/{total} -> {len(records)} records")

    return records


def build_test_split_fast(
    papers: List[Dict[str, str]],
    encoder: DenseEncoder,
    summarizer: T5Summarizer,
    limit: Optional[int],
    final_k: int,
    noise_k: int,
    min_chunks: int,
    use_multiple_queries: bool,
    shuffle_noisy: bool,
    global_noise_pool: Optional[List[Document]],
    global_pool_embs: Optional[np.ndarray],
    rng: random.Random,
    test_easy_prob: float,
    test_hard_prob: float,
    num_chunk_workers: int = 4,
    paper_batch: int = 500,
    eval_split_name: str = "test",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Chunk and encode an evaluation split once, then generate:
        - clean examples
        - easy-noise examples
        - hard-noise examples

    This function is used for both validation and test. For this project,
    validation should also have noisy versions because the research question is
    about the impact of retrieval noise on RAG summarization.
    """
    split_label = eval_split_name
    print(
        f"\n── {split_label.upper()} "
        f"[clean + easy + hard noise — single encode pass] ──────────────────────"
    )

    # 1) Chunk papers
    chunked = batch_chunk_papers(
        papers,
        split_label,
        num_workers=num_chunk_workers,
        limit=limit,
    )
    print(f"  [{split_label}] chunked papers before filter: {len(chunked)}")

    # 2) Filter papers by minimum number of chunks
    chunked = [c for c in chunked if len(c.get("chunks", [])) >= min_chunks]
    print(f"  [{split_label}] papers after filter(min_chunks={min_chunks}): {len(chunked)}")

    if not chunked:
        print(f"  [{split_label}] no valid papers after chunk filtering")
        return [], [], []

    # 3) Encode all chunks once
    paper_map = batch_encode_all_chunks(
        chunked,
        encoder,
        paper_batch=paper_batch,
    )

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 4) Cache query embeddings
    query_emb_cache = {q: encoder.encode(q) for q in DEFAULT_QUERY_TEMPLATES}

    clean_records: List[Dict[str, Any]] = []
    easy_records:  List[Dict[str, Any]] = []
    hard_records:  List[Dict[str, Any]] = []

    for i, (pid, pdata) in enumerate(
        tqdm(paper_map.items(), desc=f"  assemble [{split_label}]", unit="paper")
    ):
        try:
            query = choose_query(pid, use_multiple_queries)
            query_emb = query_emb_cache[query]

            chunks   = pdata["chunks"]
            doc_embs = pdata["embeddings"]
            abstract = pdata["abstract"]

            documents = make_documents_from_chunks(chunks)
            if not documents:
                continue

            index = _build_faiss_index(doc_embs)

            retrieved_docs, retrieved_indices, all_scores = retrieve_clean(
                query_emb=query_emb,
                documents=documents,
                doc_embs=doc_embs,
                index=index,
                final_k=final_k,
                noise_k=noise_k,
            )
            if not retrieved_docs:
                continue

            # ----- clean example -----
            clean_contexts = [d.text for d in retrieved_docs]
            relevant_chunk_ids = _get_relevant_chunk_ids(documents, pid)
            retrieval_eval_payload = build_retrieval_eval_payload(
                retrieved_docs=retrieved_docs,
                retrieved_indices=retrieved_indices,
                all_scores=all_scores,
                relevant_chunk_ids=relevant_chunk_ids,
            )
            clean_context_eval_payload = build_context_eval_payload(
                context_docs=retrieved_docs,
                relevant_chunk_ids=relevant_chunk_ids,
            )
            clean_ex = build_training_example_safe(
                query=query,
                contexts=clean_contexts,
                target_text=abstract,
                max_contexts=final_k,
            )
            clean_ex.update({
                "paper_id": pid,
                "split": split_label,
                "sample_type": "clean",
                "noise_mode": "clean",
                "noise_source": "none",
                "num_noise_chunks": 0,
                "noise_chunk_ids": [],
                "noise_source_doc_ids": [],
                "num_contexts": len(clean_contexts),
                "num_chunks_total_in_paper": len(chunks),
                "num_documents_in_context": _count_unique_source_docs(retrieved_docs),
                **retrieval_eval_payload,
                **clean_context_eval_payload,
            })
            clean_records.append(clean_ex)

            # ----- easy noise -----
            if rng.random() < test_easy_prob:
                bundle_easy = make_noisy_context(
                    clean_docs=retrieved_docs,
                    noise_k=noise_k,
                    rng=rng,
                    shuffle=shuffle_noisy,
                    global_noise_pool=global_noise_pool,
                    global_pool_embs=global_pool_embs,
                    query_emb=query_emb,
                    current_paper_id=pid,
                    noise_mode="cross_doc_easy",
                )
                if bundle_easy["has_true_noise"]:
                    easy_ex = build_training_example_safe(
                        query=query,
                        contexts=bundle_easy["noisy_contexts"],
                        target_text=abstract,
                        max_contexts=final_k + noise_k,
                    )
                    easy_context_docs = list(bundle_easy.get("noisy_docs", []))
                    if not easy_context_docs:
                        easy_context_docs = list(retrieved_docs) + list(bundle_easy.get("noise_docs", []))
                    easy_noise_docs = list(bundle_easy.get("noise_docs", []))
                    easy_context_eval_payload = build_context_eval_payload(
                        context_docs=easy_context_docs,
                        relevant_chunk_ids=relevant_chunk_ids,
                    )

                    easy_ex.update({
                        "paper_id": pid,
                        "split": split_label,
                        "sample_type": "noisy",
                        "noise_mode": "cross_doc_easy",
                        "noise_source": bundle_easy["noise_source"],
                        "num_contexts": len(bundle_easy["noisy_contexts"]),
                        "num_noise_chunks": bundle_easy.get("n_noise", 0),
                        "noise_chunk_ids": [_get_doc_id(doc) for doc in easy_noise_docs],
                        "noise_source_doc_ids": [_get_source_doc_id(doc) for doc in easy_noise_docs],
                        "num_chunks_total_in_paper": len(chunks),
                        "num_documents_in_context": _count_unique_source_docs(easy_context_docs),
                        **retrieval_eval_payload,
                        **easy_context_eval_payload,
                    })
                    easy_records.append(easy_ex)

            # ----- hard noise -----
            if rng.random() < test_hard_prob:
                bundle_hard = make_noisy_context(
                    clean_docs=retrieved_docs,
                    noise_k=noise_k,
                    rng=rng,
                    shuffle=shuffle_noisy,
                    global_noise_pool=global_noise_pool,
                    global_pool_embs=global_pool_embs,
                    query_emb=query_emb,
                    current_paper_id=pid,
                    noise_mode="cross_doc_hard",
                )
                if bundle_hard["has_true_noise"]:
                    hard_ex = build_training_example_safe(
                        query=query,
                        contexts=bundle_hard["noisy_contexts"],
                        target_text=abstract,
                        max_contexts=final_k + noise_k,
                    )
                    hard_context_docs = list(bundle_hard.get("noisy_docs", []))
                    if not hard_context_docs:
                        hard_context_docs = list(retrieved_docs) + list(bundle_hard.get("noise_docs", []))
                    hard_noise_docs = list(bundle_hard.get("noise_docs", []))
                    hard_context_eval_payload = build_context_eval_payload(
                        context_docs=hard_context_docs,
                        relevant_chunk_ids=relevant_chunk_ids,
                    )
                    hard_ex.update({
                        "paper_id": pid,
                        "split": split_label,
                        "sample_type": "noisy",
                        "noise_mode": "cross_doc_hard",
                        "noise_source": bundle_hard["noise_source"],
                        "num_contexts": len(bundle_hard["noisy_contexts"]),
                        "num_noise_chunks": bundle_hard.get("n_noise", 0),
                        "noise_chunk_ids": [_get_doc_id(doc) for doc in hard_noise_docs],
                        "noise_source_doc_ids": [_get_source_doc_id(doc) for doc in hard_noise_docs],
                        "num_chunks_total_in_paper": len(chunks),
                        "num_documents_in_context": _count_unique_source_docs(hard_context_docs),
                        **retrieval_eval_payload,
                        **hard_context_eval_payload,
                    })
                    hard_records.append(hard_ex)

        except Exception as e:
            print(f"[WARN] {split_label}::{pid}: {type(e).__name__}: {e}")

        if (i + 1) % 500 == 0:
            print(
                f"  [{split_label}] {i+1}/{len(paper_map)} "
                f"clean={len(clean_records)} "
                f"easy={len(easy_records)} "
                f"hard={len(hard_records)}"
            )

    return clean_records, easy_records, hard_records

# ---------------------------------------------------------------------------
# Noise pool builder (also benefits from batch encode)
# ---------------------------------------------------------------------------

def build_global_noise_pool(
    papers:      List[Dict[str, str]],
    encoder:     DenseEncoder,
    limit:       int = 300,
    min_chunks:  int = 1,
    num_workers: int = 4,
    paper_batch: int = 500,
) -> Tuple[List[Document], np.ndarray]:
    """
    Build cross-document noise pool from a subset of papers.

    Returns:
        pool_docs: List[Document]
        pool_embs: np.ndarray of shape (N_chunks, D)
    """
    print("\nBuilding cross-document noise pool...")

    # 1) Chunk papers
    chunked = batch_chunk_papers(
        papers,
        "noise_pool",
        num_workers=num_workers,
        limit=limit,
    )

    print(f"  [noise_pool] chunked papers before filter: {len(chunked)}")

    # 2) Keep papers that have at least 1 chunk
    chunked = [c for c in chunked if len(c.get("chunks", [])) >= min_chunks]

    print(f"  [noise_pool] papers after filter(min_chunks={min_chunks}): {len(chunked)}")

    if not chunked:
        print("  [noise_pool] no valid papers after chunk filtering")
        return [], np.empty((0, 384), dtype=np.float32)

    # Optional debug: inspect first few chunk counts
    sample_counts = [len(c["chunks"]) for c in chunked[:10]]
    print(f"  [noise_pool] sample chunk counts: {sample_counts}")

    # 3) Encode all chunks
    paper_map = batch_encode_all_chunks(
        chunked,
        encoder,
        paper_batch=paper_batch,
    )

    # 4) Flatten documents + embeddings
    pool_docs: List[Document] = []
    pool_embs_list: List[np.ndarray] = []

    for pid, pdata in paper_map.items():
        docs = make_documents_from_chunks(pdata["chunks"])
        embs = pdata["embeddings"]

        if not docs or embs is None or len(embs) == 0:
            continue

        # Safety check: docs and embeddings must align
        if len(docs) != len(embs):
            print(f"[WARN] noise_pool {pid}: len(docs)={len(docs)} != len(embs)={len(embs)}")
            n = min(len(docs), len(embs))
            docs = docs[:n]
            embs = embs[:n]

        pool_docs.extend(docs)
        pool_embs_list.append(np.asarray(embs, dtype=np.float32))

    if not pool_docs or not pool_embs_list:
        print("  [noise_pool] no documents or embeddings collected")
        return [], np.empty((0, 384), dtype=np.float32)

    pool_embs = np.vstack(pool_embs_list).astype(np.float32)

    print(f"  [noise_pool] final: {len(pool_docs)} chunks from {len(chunked)} papers")
    return pool_docs, pool_embs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _default_value_for_jsonl_key(key: str) -> Any:
    """
    Return stable default values so every JSONL row has a compatible schema.

    Why this matters:
    HuggingFace `load_dataset("json")` may fail when clean rows do not have
    fields that noisy rows have, e.g. `noise_mode`, `noise_chunk_ids`.
    """
    list_keys = {
        "retrieved_chunk_ids",
        "retrieved_source_doc_ids",
        "retrieved_scores",
        "retrieved_hit_flags",
        "relevant_chunk_ids",
        "context_chunk_ids",
        "context_source_doc_ids",
        "context_hit_flags",
        "noise_chunk_ids",
        "noise_source_doc_ids",
    }
    string_defaults = {
        "noise_mode": "clean",
        "noise_source": "none",
    }
    int_defaults = {
        "num_noise_chunks": 0,
    }

    if key in list_keys:
        return []
    if key in string_defaults:
        return string_defaults[key]
    if key in int_defaults:
        return int_defaults[key]
    return None


def normalize_jsonl_schema(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure all records share the same JSONL keys before writing.

    This prevents schema errors such as:
        KeyError: 'noise_mode'

    when `train.jsonl` or `valid.jsonl` mixes clean/easy/hard samples.
    """
    required_keys = {
        "sample_type",
        "noise_mode",
        "noise_source",
        "num_noise_chunks",
        "noise_chunk_ids",
        "noise_source_doc_ids",
    }

    all_keys = set(required_keys)
    for record in records:
        all_keys.update(record.keys())

    normalized: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)

        # Clean samples must still declare noise fields.
        if item.get("sample_type") == "clean":
            item.setdefault("noise_mode", "clean")
            item.setdefault("noise_source", "none")
            item.setdefault("num_noise_chunks", 0)
            item.setdefault("noise_chunk_ids", [])
            item.setdefault("noise_source_doc_ids", [])

        for key in all_keys:
            if key not in item:
                item[key] = _default_value_for_jsonl_key(key)

        normalized.append(item)

    return normalized


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    # Ensure the output directory exists before writing JSONL files.
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    records = normalize_jsonl_schema(records)

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build conference-ready RAG summarization data from the full arXiv split. "
            "This held-out version builds the distractor/noise pool from papers "
            "that are not used as train/validation/test target examples. "
            "Train and validation are saved as clean+easy+hard mixed JSONL files; "
            "the script also exports clean-matched training/validation JSONL files "
            "with the same number of rows as the noise-aware files. "
            "test is saved as clean, additive easy/hard, and substitutive easy/hard files."
        )
    )
    parser.add_argument("--source",      type=str, default="arxiv", choices=["arxiv"])
    parser.add_argument("--arxiv_dir",   type=str, default=ARXIV_DIR)
    parser.add_argument("--output_dir",  type=str, default="./prepared_data_heldout_noise")

    parser.add_argument("--min_target_words", type=int, default=DEFAULT_MIN_TARGET_WORDS,
                        help="Drop examples whose target/abstract has fewer words than this. Default 20.")
    parser.add_argument("--max_target_words", type=int, default=DEFAULT_MAX_TARGET_WORDS,
                        help="Drop examples whose target/abstract has more words than this. Default 512.")
    parser.add_argument("--laptop_safe", action="store_true",
                        help="Print laptop-safe guidance. Use smaller --encode_batch_size/--paper_batch/--num_workers on Windows laptops.")

    parser.add_argument("--train_limit", type=int, default=20000, help="Number of target training papers. Default 20000 so the held-out noise pool can start at train[20000]. Use None only with an explicit --noise_pool_offset.")
    parser.add_argument("--valid_limit", type=int, default=1000)
    parser.add_argument("--test_limit",  type=int, default=500, help="Number of target TEST papers/examples. Default 500 so each test JSONL is capped to about 500 samples for laptop-friendly evaluation.")

    parser.add_argument("--final_k",  type=int, default=3)
    parser.add_argument("--noise_k",  type=int, default=2)
    parser.add_argument("--min_chunks", type=int, default=3)
    parser.add_argument("--single_query", action="store_true")
    parser.add_argument("--no_shuffle_noisy", action="store_true")
    parser.add_argument(
        "--no_shuffle_train_records",
        action="store_true",
        help="Do not shuffle the combined train.jsonl records after merging clean/easy/hard."
    )

    parser.add_argument(
        "--clean_matched_copies",
        type=int,
        default=3,
        help=(
            "Only used when --clean_control_mode repeat. Number of clean copies per target paper "
            "for the old repeated clean-control file."
        ),
    )
    parser.add_argument(
        "--clean_control_mode",
        type=str,
        default="unique",
        choices=["unique", "repeat"],
        help=(
            "unique: train_clean_matched/valid_clean_matched are filled with different clean papers, "
            "so paper_id is not repeated. repeat: old behavior, repeat the same clean rows to match the "
            "noise-aware row count."
        ),
    )
    parser.add_argument(
        "--clean_extra_offset",
        type=int,
        default=None,
        help=(
            "Start index inside the filtered arXiv train split for extra unique clean-control papers. "
            "If omitted, the script starts after train targets, train/valid noise pool, and test noise pool."
        ),
    )
    parser.add_argument(
        "--valid_clean_extra_offset",
        type=int,
        default=None,
        help=(
            "Start index inside the filtered validation split for extra unique validation clean-control papers. "
            "If omitted, the script starts at valid_limit."
        ),
    )
    parser.add_argument(
        "--clean_extra_oversample_factor",
        type=float,
        default=1.35,
        help=(
            "When building unique clean-control rows, request extra papers by this factor because some papers "
            "may be dropped by chunk filtering. Default 1.35."
        ),
    )

    # Pool large enough to supply both easy and hard noise candidates.
    parser.add_argument("--noise_pool_limit", type=int, default=10000)
    parser.add_argument(
        "--noise_pool_strategy",
        type=str,
        default="heldout_train_tail",
        choices=["heldout_train_tail", "legacy_train_pool"],
        help=(
            "heldout_train_tail: use train papers after the target train_limit as the "
            "distractor pool. This is the recommended setting for rank-B style validity. "
            "legacy_train_pool: reproduce the old behavior by using the beginning of the "
            "training split as the noise pool."
        ),
    )
    parser.add_argument(
        "--noise_pool_offset",
        type=int,
        default=None,
        help=(
            "Start index for the held-out noise pool inside the training split. "
            "If omitted, the script uses train_limit as the offset. "
            "Example: train_limit=20000 -> noise pool starts at train[20000]."
        ),
    )

    parser.add_argument(
        "--test_noise_pool_limit",
        type=int,
        default=10000,
        help=(
            "Number of held-out papers used only as the TEST distractor pool. "
            "Default 10000. Recommended: train target [0:20000], train/valid noise [20000:30000], test noise [30000:40000]."
        ),
    )
    parser.add_argument(
        "--test_noise_pool_offset",
        type=int,
        default=30000,
        help=(
            "Start index for the independent test distractor pool inside the training split. "
            "Default 30000 so it is disjoint from train target and train-noise pool."
        ),
    )
    parser.add_argument(
        "--substitutive_clean_k",
        type=int,
        default=1,
        help=(
            "Number of clean target-document chunks kept in substitutive-noise test. "
            "Default 1 with noise_k=2 gives 1 clean + 2 noise = 3 chunks, matching clean context length in chunk count."
        ),
    )

    # By default, every train and validation paper creates three versions:
    # clean + easy-noise + hard-noise. If noise candidates cannot be found,
    # that noisy version is skipped and reported in the summary.
    parser.add_argument(
        "--train_easy_prob",
        type=float,
        default=1.0,
        help="Probability of creating an easy-noise training sample for each training paper."
    )
    parser.add_argument(
        "--train_hard_prob",
        type=float,
        default=1.0,
        help="Probability of creating a hard-noise training sample for each training paper."
    )
    parser.add_argument(
        "--valid_easy_prob",
        type=float,
        default=1.0,
        help="Probability of creating an easy-noise validation sample for each validation paper."
    )
    parser.add_argument(
        "--valid_hard_prob",
        type=float,
        default=1.0,
        help="Probability of creating a hard-noise validation sample for each validation paper."
    )
    parser.add_argument(
        "--test_easy_prob",
        type=float,
        default=1.0,
        help="Probability of creating an easy-noise test sample for each test paper."
    )
    parser.add_argument(
        "--test_hard_prob",
        type=float,
        default=1.0,
        help="Probability of creating a hard-noise test sample for each test paper."
    )

    # Performance
    parser.add_argument("--num_workers",       type=int, default=2)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--paper_batch",       type=int, default=100)
    parser.add_argument("--seed",              type=int, default=42)
    return parser.parse_args()


def select_noise_pool_papers(
    args: argparse.Namespace,
    train_papers: List[Dict[str, str]],
    valid_papers: List[Dict[str, str]],
    test_papers: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Select papers for the distractor/noise pool.

    Recommended strategy: ``heldout_train_tail``.

    If the target training set is ``train_papers[:train_limit]``, the noise pool
    is taken from ``train_papers[train_limit : train_limit + noise_pool_limit]``.
    Therefore, the distractor documents are not used as target examples in
    train/validation/test. Validation and test targets already come from separate
    dataset splits.

    This does not create a new model architecture. It only fixes the evaluation
    design so that cross-document distractors come from a held-out document pool.
    """
    strategy = args.noise_pool_strategy

    if strategy == "legacy_train_pool":
        pool = train_papers[: args.noise_pool_limit]
        meta = {
            "noise_pool_strategy": "legacy_train_pool",
            "description": (
                "Legacy behavior: distractors are selected from the beginning of "
                "the training split. This is not a held-out distractor pool."
            ),
            "train_target_range": [0, args.train_limit if args.train_limit is not None else len(train_papers)],
            "validation_target_split": "validation",
            "test_target_split": "test",
            "noise_pool_range": [0, len(pool)],
            "noise_pool_size_requested": args.noise_pool_limit,
            "noise_pool_size_actual": len(pool),
            "is_held_out_from_target_train_subset": False,
        }
        return pool, meta

    if args.noise_pool_offset is not None:
        start = int(args.noise_pool_offset)
    else:
        if args.train_limit is None:
            raise ValueError(
                "heldout_train_tail requires --train_limit or an explicit "
                "--noise_pool_offset. In this fixed version the default is "
                "--train_limit 20000. If you changed it to None, run with "
                "--train_limit 20000 --noise_pool_limit 10000."
            )
        start = int(args.train_limit)

    if start < 0:
        raise ValueError("--noise_pool_offset must be >= 0")
    if start >= len(train_papers):
        raise ValueError(
            f"Held-out noise pool offset {start} is outside the training split "
            f"size ({len(train_papers)})."
        )

    end = min(start + int(args.noise_pool_limit), len(train_papers))
    pool = train_papers[start:end]
    if len(pool) == 0:
        raise ValueError("Held-out noise pool is empty. Increase dataset size or reduce offset.")

    train_target_end = args.train_limit if args.train_limit is not None else start
    overlap_with_target_train = not (start >= int(train_target_end))
    if overlap_with_target_train:
        raise ValueError(
            f"Held-out noise pool overlaps target train range: "
            f"target=[0,{int(train_target_end)}), noise=[{start},{end}). "
            "Use --noise_pool_offset >= --train_limit or --noise_pool_strategy legacy_train_pool."
        )

    meta = {
        "noise_pool_strategy": "heldout_train_tail",
        "description": (
            "Distractors are selected from a held-out tail of the arXiv training split. "
            "These documents are not used as target examples in the configured train, "
            "validation, or test target sets."
        ),
        "train_target_range": [0, int(train_target_end)],
        "validation_target_split": "validation",
        "test_target_split": "test",
        "noise_pool_range": [int(start), int(end)],
        "noise_pool_size_requested": int(args.noise_pool_limit),
        "noise_pool_size_actual": len(pool),
        "is_held_out_from_target_train_subset": not overlap_with_target_train,
        "important_note": (
            "This is a held-out distractor document pool for retrieval-noise injection. "
            "It is still drawn from the same arXiv dataset distribution, but not from "
            "the configured target examples."
        ),
    }
    return pool, meta


def make_clean_matched_records(
    clean_records: List[Dict[str, Any]],
    copies: int = 3,
    split_name: str = "train",
    rng: Optional[random.Random] = None,
    shuffle: bool = True,
) -> List[Dict[str, Any]]:
    """Create a clean-only control file with the same row count as noise-aware data.

    Noise-aware training uses three variants per target paper: clean, high-similarity
    noise, and low-similarity noise. If the clean model is trained with only one
    clean row per paper, a reviewer can argue that any gain may come from seeing
    more input variants/rows rather than from noise exposure itself.

    This helper creates ``copies`` clean rows per target paper. With the default
    ``copies=3``, the clean-control train file has approximately the same number
    of rows as the noise-aware train file, while every row still uses clean context
    only. This controls the training-sample-count comparison, but it does not
    create new unique target papers.
    """
    copies = max(int(copies), 1)
    matched: List[Dict[str, Any]] = []

    for rec in clean_records:
        for copy_id in range(copies):
            item = dict(rec)
            item["split"] = split_name
            item["sample_type"] = "clean"
            item["noise_mode"] = "clean"
            item["noise_source"] = "none"
            item["num_noise_chunks"] = 0
            item["noise_chunk_ids"] = []
            item["noise_source_doc_ids"] = []
            item["clean_control_copy_id"] = copy_id
            item["clean_control_num_copies"] = copies
            item["train_setting"] = "clean_matched"
            item["matched_to_noiseaware_rows"] = True
            matched.append(item)

    if shuffle and rng is not None:
        rng.shuffle(matched)
    return matched


def make_unique_clean_matched_records(
    primary_clean_records: List[Dict[str, Any]],
    extra_clean_records: List[Dict[str, Any]],
    desired_count: int,
    split_name: str = "train",
    rng: Optional[random.Random] = None,
    shuffle: bool = True,
) -> List[Dict[str, Any]]:
    """Create a clean-control file without repeating the same paper_id.

    This is the non-duplicated alternative to the old clean-matched control.
    It first uses the clean records from the main target set, then fills the
    remaining rows with extra clean-only papers from a disjoint tail of the
    same raw split. The result can match the noise-aware row count while keeping
    one clean row per paper_id.

    Important: this removes exact duplicate clean inputs, but the clean-control
    model now sees more unique target papers than the noise-aware model. Report
    this as a clean-diversity control rather than a perfect paired control.
    """
    desired_count = int(desired_count)
    matched: List[Dict[str, Any]] = []
    seen_paper_ids = set()

    def _add_records(records: List[Dict[str, Any]], source_label: str) -> None:
        nonlocal matched
        for rec in records:
            if len(matched) >= desired_count:
                break
            pid = str(rec.get("paper_id", "")).strip()
            if not pid or pid in seen_paper_ids:
                continue
            item = dict(rec)
            item["split"] = split_name
            item["sample_type"] = "clean"
            item["noise_mode"] = "clean"
            item["noise_source"] = "none"
            item["num_noise_chunks"] = 0
            item["noise_chunk_ids"] = []
            item["noise_source_doc_ids"] = []
            item["clean_control_copy_id"] = 0
            item["clean_control_num_copies"] = 1
            item["train_setting"] = "clean_matched_unique"
            item["matched_to_noiseaware_rows"] = True
            item["clean_control_mode"] = "unique_no_repeat"
            item["clean_control_source"] = source_label
            item["clean_control_uses_unique_paper_ids"] = True
            matched.append(item)
            seen_paper_ids.add(pid)

    _add_records(primary_clean_records, "primary_target_clean")
    _add_records(extra_clean_records, "extra_unique_clean")

    if len(matched) < desired_count:
        raise ValueError(
            f"Not enough unique clean rows for {split_name}: "
            f"needed {desired_count}, got {len(matched)}. "
            f"Increase --clean_extra_oversample_factor or lower train/valid limits, "
            f"or use --clean_control_mode repeat to reproduce the old duplicated control."
        )

    if shuffle and rng is not None:
        rng.shuffle(matched)
    return matched


def _compute_unique_clean_request(needed: int, oversample_factor: float) -> int:
    if needed <= 0:
        return 0
    return max(int(needed * max(float(oversample_factor), 1.0)) + 1000, needed)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    print("\nFull-dataset / conference-ready settings:")
    print(f"  train_limit={args.train_limit}  (default 20000 for held-out noise pool)")
    print(f"  valid_limit={args.valid_limit}  (default 1000)")
    print(f"  test_limit ={args.test_limit}  (default 500; test files are kept at 500 samples)")
    print(f"  min_chunks={args.min_chunks}")
    print(f"  noise_pool_limit={args.noise_pool_limit:,} papers")

    # ------------------------------------------------------------------
    # Load raw papers
    # ------------------------------------------------------------------
    print("Loading dataset...")
    train_papers = load_arxiv_arrow(os.path.join(args.arxiv_dir, "train"))
    valid_papers = load_arxiv_arrow(os.path.join(args.arxiv_dir, "validation"))
    test_papers  = load_arxiv_arrow(os.path.join(args.arxiv_dir, "test"))
    print(f"  raw train={len(train_papers):,}  valid={len(valid_papers):,}  "
          f"test={len(test_papers):,}")

    # ------------------------------------------------------------------
    # Target/reference filter for BART + LED
    # ------------------------------------------------------------------
    train_papers, train_filter_meta = filter_papers_by_target_length(
        train_papers, "train", args.min_target_words, args.max_target_words
    )
    valid_papers, valid_filter_meta = filter_papers_by_target_length(
        valid_papers, "validation", args.min_target_words, args.max_target_words
    )
    test_papers, test_filter_meta = filter_papers_by_target_length(
        test_papers, "test", args.min_target_words, args.max_target_words
    )

    target_filter_meta = {
        "purpose": "Remove target_text/abstract outliers before building BART/LED JSONL data.",
        "min_target_words": int(args.min_target_words),
        "max_target_words": int(args.max_target_words),
        "splits": {
            "train": train_filter_meta,
            "validation": valid_filter_meta,
            "test": test_filter_meta,
        },
    }
    print("\nTarget/reference filter for BART + LED:")
    print(json.dumps(target_filter_meta, indent=2, ensure_ascii=False))

    with open(os.path.join(args.output_dir, "target_filter_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(target_filter_meta, f, indent=2, ensure_ascii=False)

    print(f"  filtered train={len(train_papers):,}  valid={len(valid_papers):,}  "
          f"test={len(test_papers):,}")

    if args.laptop_safe:
        print("\n[LAPTOP SAFE] Recommended: --encode_batch_size 16 --paper_batch 100 --num_workers 2")
        if faiss is None:
            print("[LAPTOP SAFE] faiss is not installed; using numpy search fallback.")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    print("\nLoading encoder & summarizer...")
    encoder              = DenseEncoder(batch_size=args.encode_batch_size)
    summarizer           = None  # v2: JSONL build uses lightweight builder; no tokenizer/model load
    use_multiple_queries = not args.single_query
    shuffle_noisy        = not args.no_shuffle_noisy

    # ------------------------------------------------------------------
    # Build noise pool ONCE — shared by train, validation, and test.
    # Recommended setting: held-out distractor pool.
    # ------------------------------------------------------------------
    print(f"\n  noise_pool_limit={args.noise_pool_limit}")
    print(f"  noise_pool_strategy={args.noise_pool_strategy}")
    noise_pool_papers, noise_pool_meta = select_noise_pool_papers(
        args=args,
        train_papers=train_papers,
        valid_papers=valid_papers,
        test_papers=test_papers,
    )
    print("  noise_pool_metadata:")
    print(json.dumps(noise_pool_meta, indent=2, ensure_ascii=False))

    # Save metadata early so the paper can report the exact held-out range.
    noise_meta_path = os.path.join(args.output_dir, "noise_pool_metadata.json")
    with open(noise_meta_path, "w", encoding="utf-8") as f:
        json.dump(noise_pool_meta, f, indent=2, ensure_ascii=False)

    global_noise_pool, global_pool_embs = build_global_noise_pool(
        papers=noise_pool_papers,
        encoder=encoder,
        limit=None,
        min_chunks=1,
        num_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # ------------------------------------------------------------------
    # Build an independent TEST distractor pool.
    # This pool is used only for final test additive/substitutive noisy
    # evaluation. It is intentionally disjoint from the train target subset
    # and, by default, also disjoint from the train/validation noise pool:
    #   train targets       : train[0:20000]
    #   train/valid noise   : train[20000:30000]
    #   independent test noise: train[30000:40000]
    # ------------------------------------------------------------------
    test_start = int(args.test_noise_pool_offset)
    if test_start < 0:
        raise ValueError("--test_noise_pool_offset must be >= 0")
    if test_start >= len(train_papers):
        raise ValueError(
            f"TEST noise pool offset {test_start} is outside the training split "
            f"size ({len(train_papers)})."
        )
    test_end = min(test_start + int(args.test_noise_pool_limit), len(train_papers))
    test_noise_pool_papers = train_papers[test_start:test_end]
    if len(test_noise_pool_papers) == 0:
        raise ValueError("TEST noise pool is empty. Reduce --test_noise_pool_offset or increase dataset size.")

    train_target_end = int(args.train_limit) if args.train_limit is not None else 0
    train_noise_range = noise_pool_meta.get("noise_pool_range", [None, None])
    train_noise_start, train_noise_end = train_noise_range
    disjoint_from_train_targets = test_start >= train_target_end
    disjoint_from_train_noise = True
    if train_noise_start is not None and train_noise_end is not None:
        disjoint_from_train_noise = (test_end <= int(train_noise_start)) or (test_start >= int(train_noise_end))
    if not disjoint_from_train_targets or not disjoint_from_train_noise:
        raise ValueError(
            f"TEST noise pool is not disjoint: test_noise=[{test_start},{test_end}), "
            f"train_target=[0,{train_target_end}), train_valid_noise={train_noise_range}."
        )

    test_noise_meta = {
        "noise_pool_role": "independent_test_distractor_pool",
        "description": (
            "Distractors for final test noisy conditions are selected from a separate "
            "held-out tail of the arXiv training split. This pool is not used as target "
            "training examples and is separate from the train/validation noise pool by default."
        ),
        "test_noise_pool_range": [int(test_start), int(test_end)],
        "test_noise_pool_size_requested": int(args.test_noise_pool_limit),
        "test_noise_pool_size_actual": len(test_noise_pool_papers),
        "train_target_range": [0, train_target_end],
        "train_valid_noise_pool_range": train_noise_range,
        "is_disjoint_from_train_target_subset": bool(disjoint_from_train_targets),
        "is_disjoint_from_train_valid_noise_pool": bool(disjoint_from_train_noise),
        "substitutive_design": {
            "clean_chunks_kept": int(args.substitutive_clean_k),
            "noise_chunks_added": int(args.noise_k),
            "purpose": "length-control ablation by keeping final context chunk count close to clean",
        },
    }
    test_noise_meta_path = os.path.join(args.output_dir, "test_noise_pool_metadata.json")
    with open(test_noise_meta_path, "w", encoding="utf-8") as f:
        json.dump(test_noise_meta, f, indent=2, ensure_ascii=False)

    print("\n  TEST noise_pool_metadata:")
    print(json.dumps(test_noise_meta, indent=2, ensure_ascii=False))

    test_global_noise_pool, test_global_pool_embs = build_global_noise_pool(
        papers=test_noise_pool_papers,
        encoder=encoder,
        limit=None,
        min_chunks=1,
        num_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # ------------------------------------------------------------------
    # TRAIN split
    # Output requirement:
    #   train.jsonl = clean + easy noise + hard noise in ONE file.
    # ------------------------------------------------------------------
    train_clean, train_noisy_easy, train_noisy_hard = build_test_split_fast(
        papers=train_papers,
        encoder=encoder,
        summarizer=summarizer,
        limit=args.train_limit,
        final_k=args.final_k,
        noise_k=args.noise_k,
        min_chunks=args.min_chunks,
        use_multiple_queries=use_multiple_queries,
        shuffle_noisy=shuffle_noisy,
        global_noise_pool=global_noise_pool,
        global_pool_embs=global_pool_embs,
        rng=rng,
        test_easy_prob=args.train_easy_prob,
        test_hard_prob=args.train_hard_prob,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
        eval_split_name="train",
    )
    train_records = train_clean + train_noisy_easy + train_noisy_hard
    if not args.no_shuffle_train_records:
        rng.shuffle(train_records)

    # Clean-control train file with the same number of rows as the noise-aware file.
    # Default mode is unique: no repeated paper_id in train_clean_matched.jsonl.
    train_clean_extra_records: List[Dict[str, Any]] = []
    train_clean_control_meta: Dict[str, Any] = {"mode": args.clean_control_mode}

    if args.clean_control_mode == "repeat":
        train_clean_matched = make_clean_matched_records(
            clean_records=train_clean,
            copies=args.clean_matched_copies,
            split_name="train",
            rng=rng,
            shuffle=not args.no_shuffle_train_records,
        )
        train_clean_control_meta.update({
            "policy": "repeat_same_clean_rows",
            "clean_matched_copies": int(args.clean_matched_copies),
            "unique_paper_ids": False,
        })
    else:
        train_desired = len(train_records)
        train_extra_needed = max(0, train_desired - len(train_clean))
        # By default, keep extra clean-control papers disjoint from:
        # train targets, train/valid noise pool, and independent test noise pool.
        if args.clean_extra_offset is None:
            noise_range = noise_pool_meta.get("noise_pool_range", [0, 0])
            noise_end = int(noise_range[1] or 0)
            clean_extra_start = max(
                int(args.train_limit or 0),
                noise_end,
                int(test_end),
            )
        else:
            clean_extra_start = int(args.clean_extra_offset)

        if clean_extra_start >= len(train_papers):
            raise ValueError(
                f"clean_extra_offset={clean_extra_start} is outside filtered train size {len(train_papers)}"
            )

        train_extra_request = _compute_unique_clean_request(
            train_extra_needed, args.clean_extra_oversample_factor
        )
        train_extra_papers = train_papers[clean_extra_start:]
        train_extra_limit = min(len(train_extra_papers), train_extra_request)

        print("\n── TRAIN CLEAN-MATCHED UNIQUE EXTRA CLEAN POOL ─────────────────────")
        print(f"  clean_control_mode=unique")
        print(f"  desired train_clean_matched rows={train_desired:,}")
        print(f"  primary clean rows={len(train_clean):,}")
        print(f"  extra unique clean rows needed={train_extra_needed:,}")
        print(f"  extra clean train range starts at filtered train[{clean_extra_start}]")
        print(f"  requesting extra clean papers={train_extra_limit:,}")

        if train_extra_needed > 0:
            train_clean_extra_records, _extra_easy, _extra_hard = build_test_split_fast(
                papers=train_extra_papers,
                encoder=encoder,
                summarizer=summarizer,
                limit=train_extra_limit,
                final_k=args.final_k,
                noise_k=0,
                min_chunks=args.min_chunks,
                use_multiple_queries=use_multiple_queries,
                shuffle_noisy=shuffle_noisy,
                global_noise_pool=None,
                global_pool_embs=None,
                rng=rng,
                test_easy_prob=0.0,
                test_hard_prob=0.0,
                num_chunk_workers=args.num_workers,
                paper_batch=args.paper_batch,
                eval_split_name="train_clean_extra",
            )

        train_clean_matched = make_unique_clean_matched_records(
            primary_clean_records=train_clean,
            extra_clean_records=train_clean_extra_records,
            desired_count=train_desired,
            split_name="train",
            rng=rng,
            shuffle=not args.no_shuffle_train_records,
        )
        train_clean_control_meta.update({
            "policy": "unique_no_repeat",
            "desired_rows": int(train_desired),
            "primary_clean_rows": int(len(train_clean)),
            "extra_clean_rows_built": int(len(train_clean_extra_records)),
            "extra_clean_rows_used": int(max(0, train_desired - len(train_clean))),
            "clean_extra_offset_filtered_train": int(clean_extra_start),
            "clean_extra_requested_papers": int(train_extra_limit),
            "unique_paper_ids": True,
            "note": (
                "No exact clean row repetition is used. The clean-control set is row-count matched "
                "by adding extra clean-only papers from a disjoint tail of the filtered train split."
            ),
        })

    # ------------------------------------------------------------------
    # VALIDATION split
    # Output requirement:
    #   valid.jsonl = clean + easy noise + hard noise in ONE file.
    # This lets the training script evaluate the same validation file while
    # still seeing all three retrieval conditions.
    # ------------------------------------------------------------------
    valid_clean, valid_noisy_easy, valid_noisy_hard = build_test_split_fast(
        papers=valid_papers,
        encoder=encoder,
        summarizer=summarizer,
        limit=args.valid_limit,
        final_k=args.final_k,
        noise_k=args.noise_k,
        min_chunks=args.min_chunks,
        use_multiple_queries=use_multiple_queries,
        shuffle_noisy=shuffle_noisy,
        global_noise_pool=global_noise_pool,
        global_pool_embs=global_pool_embs,
        rng=rng,
        test_easy_prob=args.valid_easy_prob,
        test_hard_prob=args.valid_hard_prob,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
        eval_split_name="validation",
    )
    valid_records = valid_clean + valid_noisy_easy + valid_noisy_hard
    valid_clean_extra_records: List[Dict[str, Any]] = []
    valid_clean_control_meta: Dict[str, Any] = {"mode": args.clean_control_mode}

    if args.clean_control_mode == "repeat":
        valid_clean_matched = make_clean_matched_records(
            clean_records=valid_clean,
            copies=args.clean_matched_copies,
            split_name="validation",
            rng=rng,
            shuffle=False,
        )
        valid_clean_control_meta.update({
            "policy": "repeat_same_clean_rows",
            "clean_matched_copies": int(args.clean_matched_copies),
            "unique_paper_ids": False,
        })
    else:
        valid_desired = len(valid_records)
        valid_extra_needed = max(0, valid_desired - len(valid_clean))
        valid_extra_start = int(args.valid_clean_extra_offset) if args.valid_clean_extra_offset is not None else int(args.valid_limit or 0)
        if valid_extra_start >= len(valid_papers):
            raise ValueError(
                f"valid_clean_extra_offset={valid_extra_start} is outside filtered validation size {len(valid_papers)}"
            )
        valid_extra_request = _compute_unique_clean_request(
            valid_extra_needed, args.clean_extra_oversample_factor
        )
        valid_extra_papers = valid_papers[valid_extra_start:]
        valid_extra_limit = min(len(valid_extra_papers), valid_extra_request)

        print("\n── VALID CLEAN-MATCHED UNIQUE EXTRA CLEAN POOL ─────────────────────")
        print(f"  desired valid_clean_matched rows={valid_desired:,}")
        print(f"  primary valid clean rows={len(valid_clean):,}")
        print(f"  extra unique valid clean rows needed={valid_extra_needed:,}")
        print(f"  extra validation range starts at filtered validation[{valid_extra_start}]")
        print(f"  requesting extra validation clean papers={valid_extra_limit:,}")

        if valid_extra_needed > 0:
            valid_clean_extra_records, _valid_extra_easy, _valid_extra_hard = build_test_split_fast(
                papers=valid_extra_papers,
                encoder=encoder,
                summarizer=summarizer,
                limit=valid_extra_limit,
                final_k=args.final_k,
                noise_k=0,
                min_chunks=args.min_chunks,
                use_multiple_queries=use_multiple_queries,
                shuffle_noisy=shuffle_noisy,
                global_noise_pool=None,
                global_pool_embs=None,
                rng=rng,
                test_easy_prob=0.0,
                test_hard_prob=0.0,
                num_chunk_workers=args.num_workers,
                paper_batch=args.paper_batch,
                eval_split_name="validation_clean_extra",
            )

        valid_clean_matched = make_unique_clean_matched_records(
            primary_clean_records=valid_clean,
            extra_clean_records=valid_clean_extra_records,
            desired_count=valid_desired,
            split_name="validation",
            rng=rng,
            shuffle=False,
        )
        valid_clean_control_meta.update({
            "policy": "unique_no_repeat",
            "desired_rows": int(valid_desired),
            "primary_clean_rows": int(len(valid_clean)),
            "extra_clean_rows_built": int(len(valid_clean_extra_records)),
            "extra_clean_rows_used": int(max(0, valid_desired - len(valid_clean))),
            "valid_clean_extra_offset_filtered_validation": int(valid_extra_start),
            "valid_extra_requested_papers": int(valid_extra_limit),
            "unique_paper_ids": True,
        })

    # ------------------------------------------------------------------
    # TEST split: additive noise and substitutive noise.
    # Additive:      3 clean chunks + 2 noise chunks (old design)
    # Substitutive:  1 clean chunk  + 2 noise chunks = 3 chunks total
    # The substitutive set is the key length-control ablation: it keeps the
    # final context chunk count close to clean and helps separate noise effects
    # from the additive input-length confound.
    # IMPORTANT: both final noisy test variants use the independent TEST pool.
    # ------------------------------------------------------------------
    test_clean, test_noisy_easy, test_noisy_hard = build_test_split_fast(
        papers=test_papers,
        encoder=encoder,
        summarizer=summarizer,
        limit=args.test_limit,
        final_k=args.final_k,
        noise_k=args.noise_k,
        min_chunks=args.min_chunks,
        use_multiple_queries=use_multiple_queries,
        shuffle_noisy=shuffle_noisy,
        global_noise_pool=test_global_noise_pool,
        global_pool_embs=test_global_pool_embs,
        rng=rng,
        test_easy_prob=args.test_easy_prob,
        test_hard_prob=args.test_hard_prob,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
        eval_split_name="test",
    )

    _test_clean_subst, test_substitutive_easy, test_substitutive_hard = build_test_split_fast(
        papers=test_papers,
        encoder=encoder,
        summarizer=summarizer,
        limit=args.test_limit,
        final_k=args.substitutive_clean_k,
        noise_k=args.noise_k,
        min_chunks=args.min_chunks,
        use_multiple_queries=use_multiple_queries,
        shuffle_noisy=shuffle_noisy,
        global_noise_pool=test_global_noise_pool,
        global_pool_embs=test_global_pool_embs,
        rng=rng,
        test_easy_prob=args.test_easy_prob,
        test_hard_prob=args.test_hard_prob,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
        eval_split_name="test",
    )
    # Mark substitutive mode explicitly for downstream tables.
    for r in test_substitutive_easy:
        r["noise_design"] = "substitutive"
        r["noise_mode"] = "cross_doc_easy_substitutive"
    for r in test_substitutive_hard:
        r["noise_design"] = "substitutive"
        r["noise_mode"] = "cross_doc_hard_substitutive"
    for r in test_noisy_easy:
        r["noise_design"] = "additive"
    for r in test_noisy_hard:
        r["noise_design"] = "additive"
    for r in test_clean:
        r["noise_design"] = "clean"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    paths = {
        # Backward-compatible names. Use train.jsonl for noise-aware training.
        "train":               os.path.join(args.output_dir, "train.jsonl"),
        "valid":               os.path.join(args.output_dir, "valid.jsonl"),
        # Explicit names for cleaner experimental reporting.
        "train_noiseaware":     os.path.join(args.output_dir, "train_noiseaware.jsonl"),
        "train_clean_matched":  os.path.join(args.output_dir, "train_clean_matched.jsonl"),
        "valid_noiseaware":     os.path.join(args.output_dir, "valid_noiseaware.jsonl"),
        "valid_clean_matched":  os.path.join(args.output_dir, "valid_clean_matched.jsonl"),
        "test_clean":          os.path.join(args.output_dir, "test_clean.jsonl"),
        "test_noisy_easy":     os.path.join(args.output_dir, "test_noisy_easy.jsonl"),
        "test_noisy_hard":     os.path.join(args.output_dir, "test_noisy_hard.jsonl"),
        "test_noisy_easy_additive": os.path.join(args.output_dir, "test_noisy_easy_additive.jsonl"),
        "test_noisy_hard_additive": os.path.join(args.output_dir, "test_noisy_hard_additive.jsonl"),
        "test_noisy_easy_substitutive": os.path.join(args.output_dir, "test_noisy_easy_substitutive.jsonl"),
        "test_noisy_hard_substitutive": os.path.join(args.output_dir, "test_noisy_hard_substitutive.jsonl"),
    }
    write_jsonl(paths["train"],               train_records)
    write_jsonl(paths["valid"],               valid_records)
    write_jsonl(paths["train_noiseaware"],    train_records)
    write_jsonl(paths["train_clean_matched"], train_clean_matched)
    write_jsonl(paths["valid_noiseaware"],    valid_records)
    write_jsonl(paths["valid_clean_matched"], valid_clean_matched)
    write_jsonl(paths["test_clean"],          test_clean)
    # Backward-compatible additive filenames
    write_jsonl(paths["test_noisy_easy"],     test_noisy_easy)
    write_jsonl(paths["test_noisy_hard"],     test_noisy_hard)
    # Explicit additive/substitutive filenames for rank-B ablation reporting
    write_jsonl(paths["test_noisy_easy_additive"], test_noisy_easy)
    write_jsonl(paths["test_noisy_hard_additive"], test_noisy_hard)
    write_jsonl(paths["test_noisy_easy_substitutive"], test_substitutive_easy)
    write_jsonl(paths["test_noisy_hard_substitutive"], test_substitutive_hard)

    # Post-write validation so you can verify the data before BART/LED training.
    post_write_checks = {
        name: validate_jsonl_target_lengths(path)
        for name, path in paths.items()
    }
    with open(os.path.join(args.output_dir, "post_write_target_length_checks.json"), "w", encoding="utf-8") as f:
        json.dump(post_write_checks, f, indent=2, ensure_ascii=False)

    build_manifest = {
        "dataset_purpose": "BART-base and T5-base RAG summarization training/evaluation",
        "model_ready_for": ["facebook/bart-base", "google-t5/t5-base"],
        "source_dataset": "arXiv article/abstract loaded from disk",
        "target_filter": target_filter_meta,
        "jsonl_files": paths,
        "clean_control_metadata": {
            "train": train_clean_control_meta,
            "validation": valid_clean_control_meta,
        },
        "post_write_target_length_checks": post_write_checks,
        "retrieval_k_values": DEFAULT_RETRIEVAL_K_VALUES,
        "final_k": int(args.final_k),
        "noise_k": int(args.noise_k),
        "substitutive_clean_k": int(args.substitutive_clean_k),
        "seed": int(args.seed),
    }
    write_dataset_manifest(args.output_dir, build_manifest)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def pct(part: int, whole: int) -> str:
        return f"{100 * part / max(whole, 1):.1f}%"

    def pair_rate(noisy: List[Dict[str, Any]], clean: List[Dict[str, Any]]) -> str:
        return f"{len(noisy)}/{len(clean)} ({pct(len(noisy), len(clean))})"

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed/60:.1f} min")
    print("  Output design:")
    print("    train.jsonl / train_noiseaware.jsonl = clean + easy + hard")
    print("    train_clean_matched.jsonl = clean-only, row-count matched to noise-aware")
    print(f"    clean_control_mode = {args.clean_control_mode}")
    print("    valid.jsonl / valid_noiseaware.jsonl = clean + easy + hard")
    print("    valid_clean_matched.jsonl = clean-only, row-count matched")
    print("    test split  = clean + additive easy/hard + substitutive easy/hard")
    print("-" * 60)
    print(f"  target filter    : {args.min_target_words} <= target_words <= {args.max_target_words}")
    print(f"  post-write check : {os.path.join(args.output_dir, 'post_write_target_length_checks.json')}")
    print(f"  build manifest   : {os.path.join(args.output_dir, 'dataset_build_manifest.json')}")
    print("-" * 60)
    print(f"  train clean      : {len(train_clean):>7,}")
    print(f"  train noisy easy : {len(train_noisy_easy):>7,}  pair={pair_rate(train_noisy_easy, train_clean)}")
    print(f"  train noisy hard : {len(train_noisy_hard):>7,}  pair={pair_rate(train_noisy_hard, train_clean)}")
    print(f"  train total noise-aware : {len(train_records):>7,}  ->  {paths['train_noiseaware']}")
    print(f"  train clean matched     : {len(train_clean_matched):>7,}  ->  {paths['train_clean_matched']}")
    print("-" * 60)
    print(f"  valid clean      : {len(valid_clean):>7,}")
    print(f"  valid noisy easy : {len(valid_noisy_easy):>7,}  pair={pair_rate(valid_noisy_easy, valid_clean)}")
    print(f"  valid noisy hard : {len(valid_noisy_hard):>7,}  pair={pair_rate(valid_noisy_hard, valid_clean)}")
    print(f"  valid total noise-aware : {len(valid_records):>7,}  ->  {paths['valid_noiseaware']}")
    print(f"  valid clean matched     : {len(valid_clean_matched):>7,}  ->  {paths['valid_clean_matched']}")
    print("-" * 60)
    print(f"  test clean       : {len(test_clean):>7,}  ->  {paths['test_clean']}")
    print(f"  test noisy easy  : {len(test_noisy_easy):>7,}  ->  {paths['test_noisy_easy']}")
    print(f"    pair rate easy : {pair_rate(test_noisy_easy, test_clean)}")
    print(f"  test noisy hard  : {len(test_noisy_hard):>7,}  ->  {paths['test_noisy_hard']}")
    print(f"    pair rate hard : {pair_rate(test_noisy_hard, test_clean)}")
    print(f"  test subst easy  : {len(test_substitutive_easy):>7,}  ->  {paths['test_noisy_easy_substitutive']}")
    print(f"    pair rate subst easy : {pair_rate(test_substitutive_easy, test_clean)}")
    print(f"  test subst hard  : {len(test_substitutive_hard):>7,}  ->  {paths['test_noisy_hard_substitutive']}")
    print(f"    pair rate subst hard : {pair_rate(test_substitutive_hard, test_clean)}")
    print("  noise strategy   : dynamic percentile with controlled relaxation")
    print("                     easy: p70-p95 -> fallback p60-p98")
    print("                     hard: bottom p10 -> fallback bottom p20")
    if args.clean_control_mode == "repeat":
        print(f"  clean matched copies per paper: {args.clean_matched_copies}")
    else:
        print("  clean matched mode : unique_no_repeat (no repeated paper_id)")
        print(f"  train extra clean built: {train_clean_control_meta.get('extra_clean_rows_built', 0):,}")
        print(f"  valid extra clean built: {valid_clean_control_meta.get('extra_clean_rows_built', 0):,}")
    print("  eval fields      : retrieved/context chunk ids + precision@K/recall@K")
    print(f"  eval K values    : {DEFAULT_RETRIEVAL_K_VALUES}")
    print(f"  train/valid noise pool : {len(global_noise_pool):,} chunks from {len(noise_pool_papers):,} held-out papers")
    print(f"  train/valid noise meta : {noise_meta_path}")
    print(f"  TEST noise pool        : {len(test_global_noise_pool):,} chunks from {len(test_noise_pool_papers):,} held-out papers")
    print(f"  TEST noise meta        : {test_noise_meta_path}")
    print(f"  substitutive design    : {args.substitutive_clean_k} clean chunk(s) + {args.noise_k} noise chunk(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
