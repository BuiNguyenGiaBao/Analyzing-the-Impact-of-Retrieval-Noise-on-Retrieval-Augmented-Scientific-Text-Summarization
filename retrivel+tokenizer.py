import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Union, Optional
import faiss
from dataclasses import dataclass
import json
from collections import OrderedDict

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

class DenseEncoder(nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """Perform mean pooling on token embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, 
               normalize: bool = True) -> np.ndarray:
        """Encode texts into dense vectors"""
        if isinstance(texts, str):
            texts = [texts]
            
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                outputs = self.transformer(**inputs)
                
                # Mean pooling
                embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Normalize if requested
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

class MMRDenseRetriever:
    
    def __init__(self, encoder: DenseEncoder, lambda_param: float = 0.5):
        """
            encoder: DenseEncoder instance
            lambda_param: MMR diversity parameter (0 = max similarity, 1 = max diversity)
        """
        self.encoder = encoder
        self.lambda_param = lambda_param
        self.documents = []
        self.index = None
        self.doc_ids = []
        self.doc_embeddings = None  # Store document embeddings for MMR
        
    def build_index(self, documents: List[Document], batch_size: int = 32):
        """Build FAISS index from documents"""
        print(f"Building index with {len(documents)} documents...")
        
        self.documents = documents
        self.doc_ids = [doc.id for doc in documents]
        
        # Extract texts from documents
        texts = [doc.text for doc in documents]
        
        # Encode all documents and store embeddings
        self.doc_embeddings = self.encoder.encode(texts, batch_size=batch_size)
        
        # Get embedding dimension
        dim = self.doc_embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        
        # Add vectors to index
        self.index.add(self.doc_embeddings.astype(np.float32))
        
        print(f"Index built successfully! Shape: {self.doc_embeddings.shape}")
        
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        # Encode query
        query_vector = self.encoder.encode(query)
        
        # Search in index
        scores, indices = self.index.search(query_vector.astype(np.float32), k)
        
        # Return documents with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def mmr_search(self, query: str, initial_k: int = 50, final_k: int = 10,
                   lambda_param: Optional[float] = None) -> List[Tuple[Document, float]]:
        """
        Perform MMR diversified search
        
        Args:
            query: Search query
            initial_k: Number of initial candidates to consider
            final_k: Number of final diversified results
            lambda_param: MMR parameter (if None, uses self.lambda_param)
        
        Returns:
            List of (document, mmr_score) tuples
        """
        if lambda_param is None:
            lambda_param = self.lambda_param
            
        # Encode query
        query_vector = self.encoder.encode(query)
        
        # Get initial candidates
        scores, indices = self.index.search(query_vector.astype(np.float32), initial_k)
        
        # Prepare candidate lists
        candidates = []
        candidate_embeddings = []
        candidate_indices = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                candidates.append(self.documents[idx])
                candidate_embeddings.append(self.doc_embeddings[idx])
                candidate_indices.append(idx)
        
        if not candidates:
            return []
        
        # Convert to numpy arrays
        candidate_embeddings = np.array(candidate_embeddings)
        query_vector = query_vector.flatten()
        
        # Calculate similarity scores with query
        query_similarities = np.dot(candidate_embeddings, query_vector)
        
        # MMR selection
        selected_indices = []
        selected_embeddings = []
        
        # Create mask for available candidates
        available_mask = np.ones(len(candidates), dtype=bool)
        
        for _ in range(min(final_k, len(candidates))):
            if not np.any(available_mask):
                break
                
            mmr_scores = []
            
            for i in range(len(candidates)):
                if not available_mask[i]:
                    mmr_scores.append(-float('inf'))
                    continue
                
                # Similarity to query
                sim_query = query_similarities[i]
                
                # Diversity term: max similarity to already selected documents
                if selected_embeddings:
                    sim_to_selected = np.max([
                        np.dot(candidate_embeddings[i], selected_emb)
                        for selected_emb in selected_embeddings
                    ])
                else:
                    sim_to_selected = 0
                
                # MMR score
                mmr_score = lambda_param * sim_query - (1 - lambda_param) * sim_to_selected
                mmr_scores.append(mmr_score)
            
            # Select best candidate
            best_idx = np.argmax(mmr_scores)
            selected_indices.append(best_idx)
            selected_embeddings.append(candidate_embeddings[best_idx])
            available_mask[best_idx] = False
        
        # Return selected documents with their MMR scores
        results = []
        for idx in selected_indices:
            doc = candidates[idx]
            mmr_score = lambda_param * query_similarities[idx]
            results.append((doc, float(mmr_score)))
        
        return results
    
    def adaptive_mmr_search(self, query: str, initial_k: int = 50, final_k: int = 10,
                           diversity_threshold: float = 0.3) -> List[Tuple[Document, float]]:
        """
        Adaptive MMR that adjusts lambda based on result diversity
        
        Args:
            query: Search query
            initial_k: Number of initial candidates
            final_k: Number of final results
            diversity_threshold: Threshold for detecting low diversity
        """
        # Get initial results with default lambda
        results = self.mmr_search(query, initial_k, initial_k, lambda_param=0.7)
        
        if not results:
            return []
        
        # Calculate diversity of top results
        top_embeddings = []
        for doc, _ in results[:min(10, len(results))]:
            doc_idx = self.doc_ids.index(doc.id)
            top_embeddings.append(self.doc_embeddings[doc_idx])
        
        if len(top_embeddings) > 1:
            # Calculate average similarity between top results
            similarities = []
            for i in range(len(top_embeddings)):
                for j in range(i+1, len(top_embeddings)):
                    sim = np.dot(top_embeddings[i], top_embeddings[j])
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            
            # Adjust lambda based on diversity
            if avg_similarity > diversity_threshold:
                # Results are too similar, increase diversity
                lambda_param = 0.3
            else:
                # Results are diverse enough, focus on relevance
                lambda_param = 0.7
        else:
            lambda_param = 0.5
        
        # Perform final search with adaptive lambda
        return self.mmr_search(query, initial_k, final_k, lambda_param)
    
    def batch_mmr_search(self, queries: List[str], initial_k: int = 50, 
                        final_k: int = 10) -> List[List[Tuple[Document, float]]]:
        """Perform MMR search for multiple queries"""
        results = []
        for query in queries:
            query_results = self.mmr_search(query, initial_k, final_k)
            results.append(query_results)
        return results

class TopKDiverseRetriever(MMRDenseRetriever):
    """Enhanced retriever with multiple diversification strategies"""
    
    def __init__(self, encoder: DenseEncoder, lambda_param: float = 0.5):
        super().__init__(encoder, lambda_param)
        self.cache = OrderedDict()  # Simple cache for recent queries
        
    def search_with_clustering(self, query: str, k: int = 10, n_clusters: int = 5) -> List[Tuple[Document, float]]:
        """
        Diversify results using clustering
        """
        from sklearn.cluster import KMeans
        
        # Get initial candidates
        scores, indices = self.index.search(
            self.encoder.encode(query).astype(np.float32), 
            k * 3
        )
        
        # Collect candidate embeddings and documents
        candidate_embeddings = []
        candidate_docs = []
        candidate_scores = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                candidate_embeddings.append(self.doc_embeddings[idx])
                candidate_docs.append(self.documents[idx])
                candidate_scores.append(score)
        
        if len(candidate_docs) < n_clusters:
            return list(zip(candidate_docs, candidate_scores))
        
        # Cluster candidates
        kmeans = KMeans(n_clusters=min(n_clusters, len(candidate_docs)), random_state=42)
        cluster_labels = kmeans.fit_predict(np.array(candidate_embeddings))
        
        # Select best document from each cluster
        selected_docs = []
        selected_scores = []
        
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if cluster_indices:
                # Get best scoring document in cluster
                best_idx = max(cluster_indices, key=lambda i: candidate_scores[i])
                selected_docs.append(candidate_docs[best_idx])
                selected_scores.append(candidate_scores[best_idx])
        
        return list(zip(selected_docs, selected_scores))
    
    def search_with_threshold_diversity(self, query: str, k: int = 10, 
                                       similarity_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """
        Diversify by removing results too similar to already selected ones
        """
        # Get initial results
        results = self.search(query, k=k*2)
        
        if not results:
            return []
        
        diversified_results = []
        seen_embeddings = []
        
        for doc, score in results:
            # Get document embedding
            doc_idx = self.doc_ids.index(doc.id)
            doc_embedding = self.doc_embeddings[doc_idx]
            
            # Check if too similar to already selected documents
            is_diverse = True
            for seen_emb in seen_embeddings:
                similarity = np.dot(doc_embedding, seen_emb)
                if similarity > similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diversified_results.append((doc, score))
                seen_embeddings.append(doc_embedding)
                
                if len(diversified_results) >= k:
                    break
        
        return diversified_results
    
    def cached_search(self, query: str, k: int = 10, use_mmr: bool = True, 
                     cache_size: int = 100) -> List[Tuple[Document, float]]:
        """
        Search with caching for frequently used queries
        """
        # Manage cache size
        if len(self.cache) > cache_size:
            self.cache.popitem(last=False)
        
        # Check cache
        cache_key = f"{query}_{k}_{use_mmr}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Perform search
        if use_mmr:
            results = self.mmr_search(query, initial_k=k*3, final_k=k)
        else:
            results = self.search(query, k)
        
        # Cache results
        self.cache[cache_key] = results
        
        return results

# Evaluation class for diversified retrieval
class DiversifiedRetrievalEvaluator:
    """Evaluate diversified retrieval performance"""
    
    @staticmethod
    def diversity_at_k(results: List[Tuple[Document, float]], 
                      embeddings: np.ndarray, k: int) -> float:
        """Calculate diversity@k as average pairwise distance"""
        if len(results) < 2:
            return 0.0
        
        # Get embeddings for top-k results
        result_embeddings = embeddings[:k]
        
        # Calculate average pairwise distance
        total_distance = 0
        count = 0
        
        for i in range(min(k, len(result_embeddings))):
            for j in range(i+1, min(k, len(result_embeddings))):
                # Convert similarity to distance (1 - cosine similarity)
                distance = 1 - np.dot(result_embeddings[i], result_embeddings[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    @staticmethod
    def relevance_diversity_tradeoff(results: List[Tuple[Document, float]], 
                                    relevance_scores: dict,
                                    embeddings: np.ndarray, k: int) -> dict:
        """Calculate both relevance and diversity metrics"""
        # Average relevance
        avg_relevance = np.mean([relevance_scores.get(doc.id, 0) for doc, _ in results[:k]])
        
        # Diversity
        diversity = DiversifiedRetrievalEvaluator.diversity_at_k(results, embeddings, k)
        
        # Harmonic mean of relevance and diversity
        if avg_relevance > 0 and diversity > 0:
            harmonic = 2 * (avg_relevance * diversity) / (avg_relevance + diversity)
        else:
            harmonic = 0
        
        return {
            'avg_relevance': avg_relevance,
            'diversity': diversity,
            'tradeoff_score': harmonic
        }

