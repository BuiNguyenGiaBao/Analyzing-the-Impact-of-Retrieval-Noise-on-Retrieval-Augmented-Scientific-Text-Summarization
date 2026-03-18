
import os
import json
import random
import argparse
import time
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import faiss
from datasets import load_from_disk

from rulebase_chunkforpdf import process_document
from retrieval_tokenizer import DenseEncoder, MMRDenseRetriever, Document
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


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _clean_field(x: Any, join_with: str = "\n") -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return join_with.join(str(i).strip() for i in x if str(i).strip())
    return str(x).replace("/n", "\n").strip()


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

def make_documents_from_chunks(chunks: List[Dict[str, Any]]) -> List[Document]:
    return [
        Document(
            # BUG FIX: globally unique id = source_doc_id + chunk_id to avoid cross-paper collisions
            id=f"{c.get('source_doc_id', 'unk')}_{c['chunk_id']}",
            text=c["text"],
            metadata=c,
        )
        for c in chunks
        if c.get("text", "").strip()
    ]
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

def choose_query(paper_id: str, use_multiple: bool = True) -> str:
    if not use_multiple:
        return DEFAULT_QUERY_TEMPLATES[0]
    return random.Random(paper_id).choice(DEFAULT_QUERY_TEMPLATES)


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

    try:
        processed = process_document(article, source_doc_id=f"{split_name}_{idx}")
        chunks = processed.get("chunks", [])

        if not chunks:
            fallback_text = article.strip()
            if fallback_text:
                chunks = [{
                    "chunk_id": 0,
                    "text": fallback_text[:4000],
                    "source_doc_id": f"{split_name}_{idx}",
                }]
            else:
                print(f"[DEBUG] no chunks for {split_name}_{idx} | article_len={len(article)}")
                return None

        return {
            "paper_id": f"{split_name}_{idx}",
            "abstract": abstract,
            "chunks": chunks,
        }

    except Exception as e:
        print(f"[DEBUG] chunk error {split_name}_{idx}: {e}")
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

def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
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
    index:       faiss.IndexFlatIP,
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
            "noise_docs": [],
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
        "noise_docs": noise_docs,
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

    retrieved_docs, _, _ = retrieve_clean(
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

    clean_ex = summarizer.build_training_example(
        query=query,
        contexts=clean_contexts,
        target_text=abstract,
        max_contexts=final_k,
    )
    clean_ex.update({
        "paper_id": pid,
        "split": split_name,
        "sample_type": "clean",
        "num_contexts": len(clean_contexts),
        "num_chunks_total_in_paper": len(chunks),
        "num_documents_in_context": _count_unique_source_docs(retrieved_docs),
    })
    records.append(clean_ex)

    # ------------------------------------------------------------------
    # Optionally build noisy example
    # ------------------------------------------------------------------
    if add_noisy and rng.random() < noisy_prob:
        bundle = make_noisy_context(
            clean_docs=retrieved_docs,
            noise_k=noise_k,
            rng=rng,
            shuffle=shuffle_noisy,
            global_noise_pool=global_noise_pool,
            global_pool_embs=global_pool_embs,
            query_emb=query_emb,
            current_paper_id=pid,
            noise_mode=noise_mode,
        )

        if bundle["has_true_noise"]:
            noisy_ex = summarizer.build_training_example(
                query=query,
                contexts=bundle["noisy_contexts"],
                target_text=abstract,
                max_contexts=final_k + noise_k,
            )

            context_docs = list(retrieved_docs) + list(bundle.get("noise_docs", []))
            unique_docs = _count_unique_source_docs(context_docs)

            # Debug safeguard: a noisy example should contain at least
            # two distinct source documents.
            if unique_docs <= 1:
                print(f"[BUG] noisy but only 1 doc: {pid}")

            noisy_ex.update({
                "paper_id": pid,
                "split": split_name,
                "sample_type": "noisy",
                "num_contexts": len(bundle["noisy_contexts"]),
                "noise_mode": noise_mode,
                "noise_source": bundle["noise_source"],
                "num_noise_chunks": bundle.get("n_noise", 0),
                "num_chunks_total_in_paper": len(chunks),
                "num_documents_in_context": unique_docs,
            })
            records.append(noisy_ex)

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
            print(f"[WARN] {split_name}::{pid}: {e}")

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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Chunk and encode test papers once, then generate:
        - clean test examples
        - easy-noise test examples
        - hard-noise test examples
    """
    print(f"\n── TEST [easy + hard noise — single encode pass] ──────────────────────")

    # 1) Chunk test papers
    chunked = batch_chunk_papers(
        papers,
        "test",
        num_workers=num_chunk_workers,
        limit=limit,
    )
    print(f"  [test] chunked papers before filter: {len(chunked)}")

    # 2) Filter papers by minimum number of chunks
    chunked = [c for c in chunked if len(c.get("chunks", [])) >= min_chunks]
    print(f"  [test] papers after filter(min_chunks={min_chunks}): {len(chunked)}")

    if not chunked:
        print("  [test] no valid papers after chunk filtering")
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
        tqdm(paper_map.items(), desc="  assemble [test]", unit="paper")
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

            retrieved_docs, _, _ = retrieve_clean(
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
            clean_ex = summarizer.build_training_example(
                query=query,
                contexts=clean_contexts,
                target_text=abstract,
                max_contexts=final_k,
            )
            clean_ex.update({
    "paper_id": pid,
    "split": "test",
    "sample_type": "clean",
    "num_contexts": len(clean_contexts),
    "num_chunks_total_in_paper": len(chunks),
    "num_documents_in_context": _count_unique_source_docs(retrieved_docs),
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
                    easy_ex = summarizer.build_training_example(
                        query=query,
                        contexts=bundle_easy["noisy_contexts"],
                        target_text=abstract,
                        max_contexts=final_k + noise_k,
                    )
                    easy_context_docs = list(retrieved_docs) + list(bundle_easy.get("noise_docs", []))

                    easy_ex.update({
                        "paper_id": pid,
                        "split": "test",
                        "sample_type": "noisy",
                        "noise_mode": "cross_doc_easy",
                        "noise_source": bundle_easy["noise_source"],
                        "num_contexts": len(bundle_easy["noisy_contexts"]),
                        "num_noise_chunks": bundle_easy.get("n_noise", 0),
                        "num_chunks_total_in_paper": len(chunks),
                        "num_documents_in_context": _count_unique_source_docs(easy_context_docs),
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
                    hard_ex = summarizer.build_training_example(
                        query=query,
                        contexts=bundle_hard["noisy_contexts"],
                        target_text=abstract,
                        max_contexts=final_k + noise_k,
                    )
                    hard_context_docs = list(retrieved_docs) + list(bundle_hard.get("noise_docs", []))
                    hard_ex.update({
                        "paper_id": pid,
                        "split": "test",
                        "sample_type": "noisy",
                        "noise_mode": "cross_doc_hard",
                        "noise_source": bundle_hard["noise_source"],
                        "num_contexts": len(bundle_hard["noisy_contexts"]),
                        "num_noise_chunks": bundle_hard.get("n_noise", 0),
                        "num_chunks_total_in_paper": len(chunks),
                        "num_documents_in_context": _count_unique_source_docs(hard_context_docs),
                    })
                    hard_records.append(hard_ex)

        except Exception as e:
            print(f"[WARN] test::{pid}: {e}")

        if (i + 1) % 500 == 0:
            print(
                f"  [test] {i+1}/{len(paper_map)} "
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

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG training data — easy/hard cross-doc noise."
    )
    parser.add_argument("--source",      type=str, default="arxiv", choices=["arxiv"])
    parser.add_argument("--arxiv_dir",   type=str, default=ARXIV_DIR)
    parser.add_argument("--output_dir",  type=str, default="./prepared_data")

    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--valid_limit", type=int, default=None)
    parser.add_argument("--test_limit",  type=int, default=None)

    parser.add_argument("--final_k",  type=int, default=3)
    parser.add_argument("--noise_k",  type=int, default=2)
    parser.add_argument("--min_chunks", type=int, default=3)
    parser.add_argument("--single_query", action="store_true")
    parser.add_argument("--no_shuffle_noisy", action="store_true")

    # Pool large enough to supply both easy and hard noise candidates
    parser.add_argument("--noise_pool_limit", type=int, default=500)

    # Train noise mode: easy, hard, or random (cross_document)
    parser.add_argument(
        "--train_noise_mode", type=str, default="cross_doc_easy",
        choices=["cross_doc_easy", "cross_doc_hard", "cross_document"],
        help="Noise mode used when building the train split."
    )
    parser.add_argument(
        "--train_noisy_prob", type=float, default=0.3,
        help="Probability of creating a noisy sample for each training paper." )
    parser.add_argument(
    "--test_easy_prob",
    type=float,
    default=0.3,
    help="Probability of creating an easy-noise test sample for each test paper."
)

    parser.add_argument(
        "--test_hard_prob",
        type=float,
        default=0.3,
        help="Probability of creating a hard-noise test sample for each test paper."
    )

    # Performance
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--paper_batch",       type=int, default=500)
    parser.add_argument("--seed",              type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------
    # Load raw papers
    # If the split limits are not provided, the pipeline runs on the full dataset.
    # ------------------------------------------------------------------
    print("Loading dataset...")
    train_papers = load_arxiv_arrow(os.path.join(args.arxiv_dir, "train"))
    valid_papers = load_arxiv_arrow(os.path.join(args.arxiv_dir, "validation"))
    test_papers  = load_arxiv_arrow(os.path.join(args.arxiv_dir, "test"))
    print(f"  train={len(train_papers):,}  valid={len(valid_papers):,}  "
          f"test={len(test_papers):,}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    print("\nLoading encoder & summarizer...")
    encoder              = DenseEncoder(batch_size=args.encode_batch_size)
    summarizer           = T5Summarizer()
    use_multiple_queries = not args.single_query
    shuffle_noisy        = not args.no_shuffle_noisy

    # ------------------------------------------------------------------
    # Build noise pool ONCE — shared for both easy and hard noise
    # ------------------------------------------------------------------
    print(f"\n  noise_pool_limit={args.noise_pool_limit}")
    global_noise_pool, global_pool_embs = build_global_noise_pool(
                papers=train_papers,
                encoder=encoder,
                limit=args.noise_pool_limit,
                min_chunks=1,
                num_workers=args.num_workers,
                paper_batch=args.paper_batch,
            )

    # Shared kwargs cho build_split_fast / build_test_split_fast
    _shared = dict(
        encoder=encoder, summarizer=summarizer,
        min_chunks=args.min_chunks,
        use_multiple_queries=use_multiple_queries,
        final_k=args.final_k, noise_k=args.noise_k,
        global_noise_pool=global_noise_pool,
        global_pool_embs=global_pool_embs,
        rng=rng,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # ------------------------------------------------------------------
    # Train split — uses train_noise_mode (default: easy)
    # ------------------------------------------------------------------
    train_records = build_split_fast(
        papers=train_papers, split_name="train",
        limit=args.train_limit,
        add_noisy=True, noisy_prob=args.train_noisy_prob,
        shuffle_noisy=shuffle_noisy,
        noise_mode=args.train_noise_mode,
        **_shared,
    )

    # ------------------------------------------------------------------
    # Validation split — no noise needed
    # ------------------------------------------------------------------
    valid_records = build_split_fast(
        papers=valid_papers, split_name="validation",
        limit=args.valid_limit,
        encoder=encoder, summarizer=summarizer,
        min_chunks=args.min_chunks,
        use_multiple_queries=use_multiple_queries,
        final_k=args.final_k, noise_k=0,
        add_noisy=False, noisy_prob=0.0, shuffle_noisy=False,
        noise_mode="cross_document",
        global_noise_pool=None,
        global_pool_embs=None,
        rng=rng,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # ------------------------------------------------------------------
    # Test split — chunk + encode once, generate easy + hard noise in a single pass
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
        global_noise_pool=global_noise_pool,
        global_pool_embs=global_pool_embs,
        rng=rng,
        test_easy_prob=args.test_easy_prob,
        test_hard_prob=args.test_hard_prob,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    paths = {
        "train":           os.path.join(args.output_dir, "train.jsonl"),
        "valid":           os.path.join(args.output_dir, "valid.jsonl"),
        "test_clean":      os.path.join(args.output_dir, "test_clean.jsonl"),
        "test_noisy_easy": os.path.join(args.output_dir, "test_noisy_easy.jsonl"),
        "test_noisy_hard": os.path.join(args.output_dir, "test_noisy_hard.jsonl"),
    }
    write_jsonl(paths["train"],           train_records)
    write_jsonl(paths["valid"],           valid_records)
    write_jsonl(paths["test_clean"],      test_clean)
    write_jsonl(paths["test_noisy_easy"], test_noisy_easy)
    write_jsonl(paths["test_noisy_hard"], test_noisy_hard)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    train_c = sum(1 for r in train_records if r["sample_type"] == "clean")
    train_n = sum(1 for r in train_records if r["sample_type"] == "noisy")

    def pair_rate(noisy, clean):
        pct = 100 * len(noisy) / max(len(clean), 1)
        return f"{len(noisy)}/{len(clean)} ({pct:.1f}%)"

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  train clean      : {train_c:>7,}  ->  {paths['train']}")
    print(f"  train noisy      : {train_n:>7,}      mode={args.train_noise_mode}")
    print(f"  train noisy prob : {args.train_noisy_prob}")
    print(f"  valid            : {len(valid_records):>7,}  ->  {paths['valid']}")
    print(f"  test clean       : {len(test_clean):>7,}  ->  {paths['test_clean']}")
    print(f"  test noisy easy  : {len(test_noisy_easy):>7,}  ->  {paths['test_noisy_easy']}")
    print(f"    pair rate easy : {pair_rate(test_noisy_easy, test_clean)}")
    print(f"  test noisy hard  : {len(test_noisy_hard):>7,}  ->  {paths['test_noisy_hard']}")
    print(f"    pair rate hard : {pair_rate(test_noisy_hard, test_clean)}")
    print("  noise strategy   : dynamic percentile with controlled relaxation")
    print("                     easy: p70-p95 -> fallback p60-p98")
    print("                     hard: bottom p10 -> fallback bottom p20")
    print(f"  noise pool       : {len(global_noise_pool):,} chunks from {args.noise_pool_limit} papers")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()