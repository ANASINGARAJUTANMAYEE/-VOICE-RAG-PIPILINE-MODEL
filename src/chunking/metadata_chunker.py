"""
Metadata-Aware Chunker
Treats MS MARCO's native passages as atomic units.
Sub-chunks only if over the token budget (e.g., >350 tokens).
Attaches full dataset metadata: lang, query_id, is_selected, chunk_strategy, char_span.
"""

from typing import List, Dict, Any


def chunk_metadata_aware(
    text: str,
    metadata: Dict[str, Any],
    max_token_budget: int = 350
) -> List[Dict[str, Any]]:
    """
    Preserves MS MARCO passage boundaries as atomic chunks.
    Only splits if the passage exceeds the specified max token budget.
    """
    words = text.strip().split()
    if not words:
        return []
        
    # If within token budget, preserve as atomic native passage
    if len(words) <= max_token_budget:
        return [{
            "chunk_id": f"{metadata.get('chunk_id', 'chk')}_meta_0",
            "text": text,
            "tokens_count": len(words),
            "chunk_strategy": "metadata",
            "char_span": [0, len(text)],
            "lang": metadata.get("lang"),
            "query_id": metadata.get("query_id"),
            "query": metadata.get("query"),
            "answer": metadata.get("answer"),
            "is_selected": metadata.get("is_selected", 0),
            "source_passage_id": metadata.get("chunk_id")
        }]
        
    # Otherwise, sub-chunk into max_token_budget chunks with small overlap
    chunks = []
    step = int(max_token_budget * 0.85)
    for i in range(0, len(words), step):
        window_words = words[i:i + max_token_budget]
        chunk_text = " ".join(window_words)
        start_char = text.find(window_words[0]) if window_words else 0
        end_char = start_char + len(chunk_text)
        
        chunks.append({
            "chunk_id": f"{metadata.get('chunk_id', 'chk')}_meta_{len(chunks)}",
            "text": chunk_text,
            "tokens_count": len(window_words),
            "chunk_strategy": "metadata",
            "char_span": [max(0, start_char), min(len(text), end_char)],
            "lang": metadata.get("lang"),
            "query_id": metadata.get("query_id"),
            "query": metadata.get("query"),
            "answer": metadata.get("answer"),
            "is_selected": metadata.get("is_selected", 0),
            "source_passage_id": metadata.get("chunk_id")
        })
        
        if i + max_token_budget >= len(words):
            break
            
    return chunks
