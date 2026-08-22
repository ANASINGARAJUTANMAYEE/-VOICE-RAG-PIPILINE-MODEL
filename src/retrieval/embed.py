"""
Multilingual Embedding Module
Supports sentence-transformers models (e.g. paraphrase-multilingual-MiniLM-L12-v2 or BAAI/bge-m3).
Provides normalized embeddings for fast FAISS cosine similarity.
"""

import sys
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers is not installed. Please install it with pip.")
            print(f"[+] Loading embedding model '{self.model_name}' on {self.device}...", flush=True)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            dim = self._model.get_embedding_dimension() if hasattr(self._model, "get_embedding_dimension") else self._model.get_sentence_embedding_dimension()
            print(f"[+] Embedding model loaded. Vector dimension: {dim}", flush=True)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Computes L2-normalized embeddings for a list of text strings.
        Returns a float32 numpy array of shape (N, dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Computes L2-normalized embedding for a single query string.
        Returns a float32 numpy array of shape (1, dimension).
        """
        emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return emb.astype(np.float32)


# Global singleton instance for fast reuse in backend & benchmark
_global_embedder = None


def get_embedder(model_name: str = DEFAULT_MODEL_NAME) -> MultilingualEmbedder:
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = MultilingualEmbedder(model_name=model_name)
    return _global_embedder
