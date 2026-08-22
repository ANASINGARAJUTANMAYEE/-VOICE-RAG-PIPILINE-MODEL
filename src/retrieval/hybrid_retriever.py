"""
Hybrid Retriever Module
Merges Dense (FAISS) + Sparse (BM25) and combines across the 3 chunking strategies using RRF (Reciprocal Rank Fusion).
Applies language-aware boosting and fallback, and conditional reranking.
Instruments and logs the retrieval stage latency (<200ms target).
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.retrieval.embed import MultilingualEmbedder, get_embedder
from src.retrieval.vector_store import FaissVectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.rerank import ConditionalReranker

# Per-language BM25 sparse weight overrides.
# paraphrase-multilingual-MiniLM-L12-v2 produces weaker dense representations
# for domain-specific Bengali and Tamil vocabulary, causing FAISS to misfire and
# leaving BM25 as the sole reliable signal. Boosting sparse_weight for bn/ta
# gives keyword matches more RRF credit without requiring a model swap.
# Diagnosed from a 45-query in-corpus probe (hi=100%, ta=73%, bn=60%).
_LANG_SPARSE_WEIGHT: Dict[str, float] = {
    "hi": 0.8,   # Default — dense + sparse well-balanced for Hindi
    "bn": 1.4,   # Boosted — dense embeddings underperform on Bengali domain vocab
    "ta": 1.1,   # Lightly boosted — full 1.4 moves Tamil OOC boundary (2099 query)
                 # above the calibrated 0.11 threshold; 1.1 improves recall conservatively
}


class HybridRetriever:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        bm25_store: BM25Store,
        embedder: Optional[MultilingualEmbedder] = None,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder or get_embedder()
        self.rrf_k = rrf_k
        self.reranker = ConditionalReranker(ambiguity_threshold=0.004)

    def retrieve(
        self,
        query: str,
        query_lang: str = "hi",
        top_k: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = None
    ) -> Dict[str, Any]:
        """
        Executes hybrid retrieval:
        1. Embeds query with multilingual embedder.
        2. Queries FAISS vector store (dense).
        3. Queries BM25 keyword store (sparse).
        4. Fuses ranks via Reciprocal Rank Fusion (RRF, k=60).
           sparse_weight is applied per-language from _LANG_SPARSE_WEIGHT:
           Hindi=0.8 (balanced), Bengali=1.4, Tamil=1.4 (BM25-boosted to
           compensate for weaker dense embeddings on domain-specific vocab).
        5. Applies same-language boosting (+25%) and cross-language penalty (-15%).
        6. Conditionally executes reranker if top candidates are ambiguous.
        7. Deduplicates cross-strategy duplicates.
        8. Measures exact latency.
        """
        # Resolve per-language sparse weight if not explicitly overridden
        if sparse_weight is None:
            sparse_weight = _LANG_SPARSE_WEIGHT.get(query_lang, 0.8)
        t_start = time.perf_counter()

        # Step 1: Embed query
        t_embed_0 = time.perf_counter()
        query_vec = self.embedder.embed_query(query)
        embed_ms = (time.perf_counter() - t_embed_0) * 1000.0

        # Step 2: Dense FAISS Search
        t_dense_0 = time.perf_counter()
        dense_results = self.vector_store.search(query_vec, top_k=top_k * 4)
        dense_ms = (time.perf_counter() - t_dense_0) * 1000.0

        # Step 3: Sparse BM25 Search
        t_sparse_0 = time.perf_counter()
        bm25_results = self.bm25_store.search(query, query_lang=query_lang, top_k=top_k * 4)
        sparse_ms = (time.perf_counter() - t_sparse_0) * 1000.0

        # Step 4: Reciprocal Rank Fusion (RRF)
        # RRF formula: RRF_score(d) = sum_m [ weight_m / (k + rank_m(d)) ]
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # Fuse Dense Ranks
        for rank, (meta, score) in enumerate(dense_results, start=1):
            cid = meta.get("chunk_id", str(rank))
            chunk_map[cid] = meta
            rrf_contrib = dense_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contrib

        # Fuse BM25 Ranks
        for rank, (meta, score) in enumerate(bm25_results, start=1):
            cid = meta.get("chunk_id", str(rank))
            chunk_map[cid] = meta
            rrf_contrib = sparse_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contrib

        # Step 5: Same-Language Boost (prioritize detected query language)
        fused_list: List[Tuple[Dict[str, Any], float]] = []
        for cid, score in rrf_scores.items():
            meta = chunk_map[cid]
            chunk_lang = meta.get("lang", "")
            
            # Boost if same language
            boosted_score = score
            if chunk_lang == query_lang:
                boosted_score *= 1.25
            else:
                boosted_score *= 0.85

            # Multi-strategy diversity weighting: slight bonus for semantic & metadata chunkers
            strategy = meta.get("chunk_strategy", "")
            if strategy in ("semantic", "metadata"):
                boosted_score *= 1.05

            fused_list.append((meta, boosted_score))

        # Sort by boosted RRF score descending
        fused_list.sort(key=lambda x: x[1], reverse=True)

        # Step 6: Conditional Reranking
        final_candidates, was_reranked, rerank_ms = self.reranker.rerank(
            query=query,
            scored_chunks=fused_list,
            top_k=top_k
        )

        total_retrieval_ms = (time.perf_counter() - t_start) * 1000.0

        # Step 7: Deduplicate cross-strategy duplicates.
        # The same underlying passage may appear multiple times from different chunking
        # strategies (e.g. hi_1102432_5_meta_0 and hi_1102432_5_semantic_0 both refer
        # to passage 5 of query hi_1102432). We derive a passage key by stripping the
        # known strategy suffixes and keep only the top-scoring chunk per unique passage.
        _STRATEGY_TOKENS = ("_meta_", "_semantic_", "_fixed_")
        seen_passage_keys: set = set()
        deduplicated: list = []
        for chunk, score in final_candidates:
            cid = chunk.get("chunk_id", "")
            # Derive passage-level key by cutting at first strategy token
            passage_key = cid
            for tok in _STRATEGY_TOKENS:
                if tok in cid:
                    passage_key = cid[:cid.index(tok)]
                    break
            if passage_key not in seen_passage_keys:
                seen_passage_keys.add(passage_key)
                deduplicated.append((chunk, score))
            if len(deduplicated) >= top_k:
                break
        final_candidates = deduplicated

        # Extract top score for confidence guardrail
        top_confidence_score = final_candidates[0][1] if final_candidates else 0.0

        # Step 8: Source coherence score.
        # Pairwise avg cosine similarity between top-3 passage embeddings.
        # Low coherence (passages disagree topically) indicates retrieval noise:
        # the retriever latched onto keyword fragments rather than a real topic cluster.
        # Uses already-computed query embedder — no extra model needed.
        source_coherence_score = 1.0  # default: passes if fewer than 2 passages
        top_texts = [c.get("text", "") for c, _ in final_candidates[:3] if c.get("text")]
        if len(top_texts) >= 2:
            vecs = self.embedder.embed_texts(top_texts, batch_size=4, show_progress=False)
            # embed_texts returns L2-normalised vectors: cosine = dot product
            pairs = [
                float(np.dot(vecs[i], vecs[j]))
                for i in range(len(vecs))
                for j in range(i + 1, len(vecs))
            ]
            source_coherence_score = round(sum(pairs) / len(pairs), 5)

        return {
            "results": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text"),
                    "score": round(score, 5),
                    "lang": chunk.get("lang"),
                    "chunk_strategy": chunk.get("chunk_strategy"),
                    "query_id": chunk.get("query_id"),
                    "is_selected": chunk.get("is_selected", 0),
                    "tokens_count": chunk.get("tokens_count", 0),
                    "char_span": chunk.get("char_span", [0, 0])
                }
                for chunk, score in final_candidates
            ],
            "top_confidence_score": round(top_confidence_score, 5),
            "source_coherence_score": source_coherence_score,
            "was_reranked": was_reranked,
            "latency": {
                "embed_ms": round(embed_ms, 2),
                "dense_search_ms": round(dense_ms, 2),
                "sparse_search_ms": round(sparse_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
                "total_retrieval_ms": round(total_retrieval_ms, 2)
            }
        }
