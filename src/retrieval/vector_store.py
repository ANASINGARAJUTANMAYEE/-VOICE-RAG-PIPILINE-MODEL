"""
In-process FAISS Vector Store
Maintains a FlatIP (Inner Product = Cosine Similarity for normalized vectors) index
and a parallel metadata dictionary mapping vector_id -> chunk metadata dict.
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissVectorStore:
    def __init__(self, dimension: int = 384):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is not installed. Please install it with pip.")
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store: Dict[int, Dict[str, Any]] = {}

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Adds vectors and corresponding metadata records to the store.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError(f"Vectors count ({len(vectors)}) != metadata count ({len(metadata_list)})")
            
        start_id = self.total_vectors
        # Ensure float32 and contiguous
        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        
        self.index.add(vectors_f32)
        
        for i, meta in enumerate(metadata_list):
            vec_id = start_id + i
            self.metadata_store[vec_id] = meta

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        lang_filter: str = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches top_k nearest neighbors by cosine similarity.
        Optionally filters by language.
        Returns list of (metadata_dict, score).
        """
        if self.total_vectors == 0:
            return []

        query_f32 = np.ascontiguousarray(query_vector, dtype=np.float32)
        
        # Search a wider pool if lang_filter is active
        fetch_k = top_k * 3 if lang_filter else top_k
        fetch_k = min(fetch_k, self.total_vectors)

        scores, indices = self.index.search(query_f32, fetch_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata_store.get(int(idx))
            if not meta:
                continue
                
            if lang_filter and meta.get("lang") != lang_filter:
                continue
                
            results.append((meta, float(score)))
            if len(results) >= top_k:
                break
                
        return results

    def save(self, index_path: Path = Path("data/faiss_index.bin"), meta_path: Path = Path("data/faiss_metadata.json")):
        """Persists the FAISS index and metadata store to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False)
        print(f"[+] Saved FAISS index ({self.total_vectors} vectors) to {index_path} and metadata to {meta_path}", flush=True)

    @classmethod
    def load(cls, index_path: Path = Path("data/faiss_index.bin"), meta_path: Path = Path("data/faiss_metadata.json")) -> "FaissVectorStore":
        """Loads a persisted FAISS index and metadata store from disk."""
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"FAISS index '{index_path}' or metadata '{meta_path}' not found.")
            
        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata_raw = json.load(f)
            
        # Convert keys back to int
        metadata_store = {int(k): v for k, v in metadata_raw.items()}
        
        store = cls(dimension=index.d)
        store.index = index
        store.metadata_store = metadata_store
        print(f"[+] Loaded FAISS index ({store.total_vectors} vectors, dim={store.dimension}) from {index_path}", flush=True)
        return store
