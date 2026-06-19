import random
from dataclasses import dataclass, field
from typing import List, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# Data classes

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

# Dense encoder
class DenseEncoder:
    """
    Encode text into L2-normalized dense embeddings using a frozen transformer.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts: List[str] | str) -> np.ndarray:
        """
        Encode a string or list of strings.
        Returns:
            np.ndarray of shape (N, D), dtype float32, L2-normalized.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise ValueError("`texts` must not be empty.")

        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]

                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                ).to(self.device)

                outputs = self.model(**inputs)
                emb = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)


# MMR dense retriever
class MMRDenseRetriever:
    """
    Dense retriever with Maximal Marginal Relevance (MMR) re-ranking.

    MMR:
        argmax_d [ lambda * sim(d, q) - (1 - lambda) * max_{d' in S} sim(d, d') ]
    """

    def __init__(
        self,
        encoder: DenseEncoder,
        mmr_lambda: float = 0.5,
    ) -> None:
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("`mmr_lambda` must be in [0, 1].")

        self.encoder = encoder
        self.mmr_lambda = mmr_lambda

        self.documents: List[Document] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None

    # Index management

    def build_index(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("`documents` must not be empty.")

        self.documents = documents
        texts = [d.text for d in documents]
        self.doc_embeddings = self.encoder.encode(texts)

        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)


    # Retrieval helpers


    def _ensure_ready(self) -> None:
        if self.index is None or self.doc_embeddings is None or not self.documents:
            raise RuntimeError("Call build_index() before retrieval.")

    def retrieve_candidates(
        self,
        query: str,
        candidate_k: int = 50,
    ) -> tuple[list[int], list[float], np.ndarray]:
        """
        Retrieve top candidate_k by cosine similarity.
        Returns:
            cand_indices, cand_scores, cand_embs
        """
        self._ensure_ready()

        if candidate_k <= 0:
            raise ValueError("`candidate_k` must be > 0.")

        candidate_k = min(candidate_k, len(self.documents))

        q_emb = self.encoder.encode(query)  # (1, D)
        scores, idxs = self.index.search(q_emb, candidate_k)

        cand_indices = idxs[0].tolist()
        cand_scores = scores[0].tolist()
        cand_embs = self.doc_embeddings[cand_indices]

        return cand_indices, cand_scores, cand_embs

    # ---------------------------------------------------------------------
    # Core MMR search
    # ---------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        candidate_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Return up to k documents via MMR re-ranking.

        Args:
            query: retrieval query
            k: final number of selected documents
            candidate_k: size of candidate pool before MMR
                         default = max(k*5, 50)
        """
        self._ensure_ready()

        if k <= 0:
            raise ValueError("`k` must be > 0.")

        if candidate_k is None:
            candidate_k = max(k * 5, 50)

        candidate_k = min(candidate_k, len(self.documents))

        cand_indices, cand_rel, cand_embs = self.retrieve_candidates(
            query=query,
            candidate_k=candidate_k,
        )

        if not cand_indices:
            return []

        sim_matrix = cand_embs @ cand_embs.T  # cosine because embeddings normalized

        selected: List[int] = []
        remaining = set(range(len(cand_indices)))

        for _ in range(min(k, len(cand_indices))):
            best_pos = -1
            best_score = float("-inf")

            for pos in remaining:
                rel = cand_rel[pos]

                if selected:
                    sim_to_selected = float(np.max(sim_matrix[pos, selected]))
                else:
                    sim_to_selected = 0.0

                mmr_score = (
                    self.mmr_lambda * rel
                    - (1.0 - self.mmr_lambda) * sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_pos = pos

            if best_pos == -1:
                break

            selected.append(best_pos)
            remaining.remove(best_pos)

        return [
            SearchResult(
                document=self.documents[cand_indices[pos]],
                score=cand_rel[pos],
                rank=rank + 1,
            )
            for rank, pos in enumerate(selected)
        ]

    # ---------------------------------------------------------------------
    # Negative sampling
    # ---------------------------------------------------------------------

    def sample_negative_documents(
        self,
        n: int = 2,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> List[Document]:
        """
        Sample random negatives from the indexed documents.
        """
        self._ensure_ready()

        if n <= 0:
            return []

        if rng is None:
            rng = random.Random(seed)

        exclude_ids = exclude_ids or set()
        pool = [doc for doc in self.documents if doc.id not in exclude_ids]

        if not pool:
            return []

        return rng.sample(pool, min(n, len(pool)))

    def sample_hard_negative_documents(
        self,
        query: str,
        n: int = 2,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        exclude_ids: Optional[set[str]] = None,
        candidate_k: int = 50,
        skip_top_m: int = 5,
    ) -> List[Document]:
        """
        Sample hard negatives:
        documents that are similar to the query but not in exclude_ids.

        Typical use:
            - retrieve top 50 by similarity
            - skip top 5 clean docs
            - sample 2 hard negatives from the rest
        """
        self._ensure_ready()

        if n <= 0:
            return []

        if rng is None:
            rng = random.Random(seed)

        exclude_ids = exclude_ids or set()

        cand_indices, _, _ = self.retrieve_candidates(
            query=query,
            candidate_k=candidate_k,
        )

        hard_pool: List[Document] = []
        for rank_pos, doc_idx in enumerate(cand_indices):
            if rank_pos < skip_top_m:
                continue
            doc = self.documents[doc_idx]
            if doc.id in exclude_ids:
                continue
            hard_pool.append(doc)

        if not hard_pool:
            return []

        return rng.sample(hard_pool, min(n, len(hard_pool)))

    # ---------------------------------------------------------------------
    # Convenience wrapper
    # ---------------------------------------------------------------------

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
        """
        Convenience wrapper for building clean and noisy contexts.

        Returns:
            {
                "clean_contexts": ...,
                "noisy_contexts": ...,
                "retrieved_items": ...
            }
        """
        rng = random.Random(seed)

        retrieved = self.search(
            query=query,
            k=final_k,
            candidate_k=candidate_k,
        )
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