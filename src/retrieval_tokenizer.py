# -*- coding: utf-8 -*-
"""Dense retrieval utilities for RAG summarization dataset building.

Fixed version highlights
------------------------
- FAISS is optional. If faiss-cpu is not installed, the code falls back to a
  NumPy inner-product index with the same small API used by this project.
- Python 3.9 compatible typing; no ``list[str] | str`` syntax.
- Empty documents are filtered before indexing.
- Encoder device supports CUDA, Apple MPS, or CPU.
- Embeddings are always returned as float32 and L2-normalized.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:  # FAISS is optional; Windows/Mac installs often fail.
    import faiss  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    faiss = None  # type: ignore


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


class NumpyIndexFlatIP:
    """Small FAISS-compatible IndexFlatIP fallback.

    This is slower than FAISS but safe for the per-paper candidate sets used by
    the dataset builder. It implements only ``add`` and ``search`` because that
    is all this project needs.
    """

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.embeddings: Optional[np.ndarray] = None

    def add(self, embeddings: np.ndarray) -> None:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings with shape (N, {self.dim}), got {arr.shape}."
            )
        self.embeddings = arr

    def search(self, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)

        q = np.asarray(query_emb, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"Expected query shape (1, {self.dim}), got {q.shape}.")

        scores = (self.embeddings @ q.T).squeeze()
        scores = np.asarray(np.atleast_1d(scores), dtype=np.float32)
        k = min(max(int(k), 0), len(scores))
        if k == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
        idxs = np.argsort(-scores)[:k].astype(np.int64)
        return scores[idxs].reshape(1, -1), idxs.reshape(1, -1)


class DenseEncoder:
    """Encode text into L2-normalized dense embeddings using a frozen model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = max(int(batch_size), 1)
        self.max_length = max(int(max_length), 8)
        self.device = torch.device(device or self._choose_device())

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _choose_device() -> str:
        env_device = os.environ.get("DENSE_ENCODER_DEVICE", "").strip()
        if env_device:
            return env_device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @staticmethod
    def _clean_texts(texts: Union[str, Sequence[Any]]) -> List[str]:
        if isinstance(texts, str):
            return [texts.strip()]
        cleaned = []
        for x in texts:
            s = "" if x is None else str(x).strip()
            cleaned.append(s)
        return cleaned

    def encode(self, texts: Union[str, Sequence[Any]]) -> np.ndarray:
        """Encode one string or a sequence of strings.

        Returns an array with shape ``(N, D)``. Empty strings are allowed because
        some callers need index alignment, but an entirely empty list is not.
        """
        clean_texts = self._clean_texts(texts)
        if len(clean_texts) == 0:
            raise ValueError("`texts` must not be empty.")

        all_embeddings: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(clean_texts), self.batch_size):
                batch = clean_texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                emb = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings).astype(np.float32)


class MMRDenseRetriever:
    """Dense retriever with Maximal Marginal Relevance re-ranking."""

    def __init__(self, encoder: DenseEncoder, mmr_lambda: float = 0.5) -> None:
        if not 0.0 <= float(mmr_lambda) <= 1.0:
            raise ValueError("`mmr_lambda` must be in [0, 1].")
        self.encoder = encoder
        self.mmr_lambda = float(mmr_lambda)
        self.documents: List[Document] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.index: Optional[Any] = None

    def build_index(self, documents: List[Document]) -> None:
        valid_docs = [d for d in documents if d and str(d.text).strip()]
        if not valid_docs:
            raise ValueError("`documents` must contain at least one non-empty document.")

        self.documents = valid_docs
        texts = [d.text for d in valid_docs]
        self.doc_embeddings = self.encoder.encode(texts)

        dim = int(self.doc_embeddings.shape[1])
        if faiss is not None:
            index = faiss.IndexFlatIP(dim)
        else:
            index = NumpyIndexFlatIP(dim)
        index.add(self.doc_embeddings)
        self.index = index

    def _ensure_ready(self) -> None:
        if self.index is None or self.doc_embeddings is None or not self.documents:
            raise RuntimeError("Call build_index() before retrieval.")

    def retrieve_candidates(
        self,
        query: str,
        candidate_k: int = 50,
    ) -> Tuple[List[int], List[float], np.ndarray]:
        self._ensure_ready()
        query = "" if query is None else str(query).strip()
        if not query:
            raise ValueError("`query` must not be empty.")
        if candidate_k <= 0:
            raise ValueError("`candidate_k` must be > 0.")

        candidate_k = min(int(candidate_k), len(self.documents))
        q_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(q_emb, candidate_k)

        cand_indices = [int(i) for i in idxs[0].tolist()]
        cand_scores = [float(s) for s in scores[0].tolist()]
        cand_embs = self.doc_embeddings[cand_indices]
        return cand_indices, cand_scores, cand_embs

    def search(
        self,
        query: str,
        k: int = 10,
        candidate_k: Optional[int] = None,
    ) -> List[SearchResult]:
        self._ensure_ready()
        if k <= 0:
            raise ValueError("`k` must be > 0.")
        if candidate_k is None:
            candidate_k = max(k * 5, 50)
        candidate_k = min(int(candidate_k), len(self.documents))

        cand_indices, cand_rel, cand_embs = self.retrieve_candidates(query, candidate_k)
        if not cand_indices:
            return []

        sim_matrix = np.asarray(cand_embs @ cand_embs.T, dtype=np.float32)
        selected: List[int] = []
        remaining: Set[int] = set(range(len(cand_indices)))

        for _ in range(min(int(k), len(cand_indices))):
            best_pos = -1
            best_score = float("-inf")
            for pos in list(remaining):
                rel = float(cand_rel[pos])
                red = float(np.max(sim_matrix[pos, selected])) if selected else 0.0
                mmr_score = self.mmr_lambda * rel - (1.0 - self.mmr_lambda) * red
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_pos = pos
            if best_pos < 0:
                break
            selected.append(best_pos)
            remaining.remove(best_pos)

        return [
            SearchResult(
                document=self.documents[cand_indices[pos]],
                score=float(cand_rel[pos]),
                rank=rank + 1,
            )
            for rank, pos in enumerate(selected)
        ]

    def sample_negative_documents(
        self,
        n: int = 2,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Document]:
        self._ensure_ready()
        if n <= 0:
            return []
        rng = rng or random.Random(seed)
        exclude_ids = exclude_ids or set()
        pool = [doc for doc in self.documents if doc.id not in exclude_ids]
        return rng.sample(pool, min(int(n), len(pool))) if pool else []

    def sample_hard_negative_documents(
        self,
        query: str,
        n: int = 2,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        exclude_ids: Optional[Set[str]] = None,
        candidate_k: int = 50,
        skip_top_m: int = 5,
    ) -> List[Document]:
        self._ensure_ready()
        if n <= 0:
            return []
        rng = rng or random.Random(seed)
        exclude_ids = exclude_ids or set()

        cand_indices, _, _ = self.retrieve_candidates(query=query, candidate_k=candidate_k)
        hard_pool: List[Document] = []
        for rank_pos, doc_idx in enumerate(cand_indices):
            if rank_pos < int(skip_top_m):
                continue
            doc = self.documents[int(doc_idx)]
            if doc.id not in exclude_ids:
                hard_pool.append(doc)
        return rng.sample(hard_pool, min(int(n), len(hard_pool))) if hard_pool else []

    def build_training_contexts(
        self,
        query: str,
        final_k: int = 3,
        noise_k: int = 2,
        shuffle: bool = False,
        seed: Optional[int] = None,
        candidate_k: Optional[int] = None,
        hard_negative: bool = False,
    ) -> dict:
        rng = random.Random(seed)
        retrieved = self.search(query=query, k=final_k, candidate_k=candidate_k)
        clean_contexts = [r.document.text for r in retrieved]
        exclude_ids = {r.document.id for r in retrieved}

        if hard_negative:
            negatives = self.sample_hard_negative_documents(
                query=query,
                n=noise_k,
                rng=rng,
                exclude_ids=exclude_ids,
                candidate_k=max(candidate_k or 50, 50),
                skip_top_m=final_k,
            )
        else:
            negatives = self.sample_negative_documents(
                n=noise_k,
                rng=rng,
                exclude_ids=exclude_ids,
            )

        noisy_contexts = clean_contexts + [doc.text for doc in negatives]
        if shuffle:
            rng.shuffle(noisy_contexts)

        return {
            "clean_contexts": clean_contexts,
            "noisy_contexts": noisy_contexts,
            "retrieved_items": retrieved,
        }
