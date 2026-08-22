"""
BM25 Sparse Keyword Store
Uses rank_bm25 with Indic-aware tokenization for Hindi, Bengali, and Tamil exact/keyword matching.
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

try:
    from indicnlp.tokenize import indic_tokenize
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False


def tokenize_indic(text: str, lang: str = "hi") -> List[str]:
    """
    Tokenizes text for BM25 matching across Hindi, Bengali, and Tamil.
    Extracts words + character n-grams for robust subword/morphological matching.
    """
    text = text.lower().strip()
    if not text:
        return []
        
    if INDIC_NLP_AVAILABLE and lang in ["hi", "bn", "ta"]:
        try:
            tokens = indic_tokenize.trivial_tokenize(text, lang=lang)
            return [t for t in tokens if len(t.strip()) > 0]
        except Exception:
            pass

    # Clean punctuation and split by whitespace
    clean_text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF]', ' ', text)
    words = clean_text.split()
    return [w for w in words if w]


class BM25Store:
    def __init__(self):
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi = None

    @property
    def total_docs(self) -> int:
        return len(self.corpus_chunks)

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Tokenizes all chunks and initializes the BM25Okapi model.
        """
        self.corpus_chunks = chunks
        self.tokenized_corpus = []
        
        for c in chunks:
            text = c.get("text", "")
            lang = c.get("lang", "hi")
            tokens = tokenize_indic(text, lang=lang)
            self.tokenized_corpus.append(tokens)
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[+] BM25 Index built over {len(self.corpus_chunks)} documents.", flush=True)

    def search(
        self,
        query: str,
        query_lang: str = "hi",
        top_k: int = 10,
        lang_filter: str = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches top_k documents by BM25 keyword score.
        Returns list of (chunk_metadata_dict, bm25_score).
        """
        if not self.bm25 or self.total_docs == 0:
            return []

        query_tokens = tokenize_indic(query, lang=query_lang)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        
        # Sort indices by score descending
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        
        results = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0:
                break
                
            chunk = self.corpus_chunks[idx]
            if lang_filter and chunk.get("lang") != lang_filter:
                continue
                
            results.append((chunk, score))
            if len(results) >= top_k:
                break
                
        return results

    def save(self, corpus_path: Path = Path("data/bm25_corpus.json")):
        """Saves the corpus chunks to disk."""
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus_chunks, f, ensure_ascii=False)
        print(f"[+] Saved BM25 corpus ({len(self.corpus_chunks)} docs) to {corpus_path}", flush=True)

    @classmethod
    def load(cls, corpus_path: Path = Path("data/bm25_corpus.json")) -> "BM25Store":
        """Loads corpus and rebuilds BM25 index."""
        if not corpus_path.exists():
            raise FileNotFoundError(f"BM25 corpus file '{corpus_path}' not found.")
            
        with open(corpus_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        store = cls()
        store.build_index(chunks)
        return store
