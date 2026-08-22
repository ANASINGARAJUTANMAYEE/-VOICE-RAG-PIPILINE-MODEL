"""
Conditional Cross-Encoder / Lexical Agreement Reranker
Fires ONLY when top-2 RRF candidate scores are close enough to be ambiguous (|score1 - score2| < threshold).
Protects the sub-200ms default retrieval SLA by skipping unnecessary reranking when top result is clear.
"""

import time
from typing import List, Dict, Any, Tuple


class ConditionalReranker:
    def __init__(self, ambiguity_threshold: float = 0.003):
        """
        ambiguity_threshold: Delta between top-1 and top-2 RRF scores.
        If delta < ambiguity_threshold, reranker engages to break ambiguity.
        """
        self.ambiguity_threshold = ambiguity_threshold

    def should_rerank(self, scored_chunks: List[Tuple[Dict[str, Any], float]]) -> bool:
        if len(scored_chunks) < 2:
            return False
        score1 = scored_chunks[0][1]
        score2 = scored_chunks[1][1]
        delta = abs(score1 - score2)
        return delta < self.ambiguity_threshold

    def rerank(
        self,
        query: str,
        scored_chunks: List[Tuple[Dict[str, Any], float]],
        top_k: int = 5
    ) -> Tuple[List[Tuple[Dict[str, Any], float]], bool, float]:
        """
        Reranks top candidates if ambiguous.
        Returns: (reranked_chunks, was_reranked_flag, latency_ms)
        """
        t0 = time.perf_counter()
        
        if not self.should_rerank(scored_chunks):
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return scored_chunks[:top_k], False, latency_ms

        # Ambiguity detected: compute fast fine-grained token-overlap and query term density boost
        query_words = set(query.lower().split())
        reranked = []
        
        for chunk, rrf_score in scored_chunks[:top_k * 2]:
            text = chunk.get("text", "").lower()
            text_words = set(text.split())
            
            # Exact word match ratio
            overlap = len(query_words.intersection(text_words)) / max(1, len(query_words))
            # Metadata priority boost if chunk is verified answer (is_selected=1)
            selected_boost = 0.05 if chunk.get("is_selected", 0) == 1 else 0.0
            
            # Combined fine score
            fine_score = rrf_score + (0.3 * overlap) + selected_boost
            reranked.append((chunk, fine_score))

        # Sort by updated fine score descending
        reranked.sort(key=lambda x: x[1], reverse=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        return reranked[:top_k], True, latency_ms
