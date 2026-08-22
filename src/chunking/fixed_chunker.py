"""
Fixed-size Chunker
Splits text into chunks of ~256 tokens with ~15% overlap.
Works across Hindi, Bengali, and Tamil.
"""

from typing import List, Dict, Any


def tokenize_words(text: str) -> List[str]:
    """Simple whitespace + punctuation aware word tokenizer for Indic/multilingual text."""
    return text.strip().split()


def chunk_fixed(
    text: str,
    metadata: Dict[str, Any],
    chunk_size_tokens: int = 256,
    overlap_pct: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Splits text into fixed token windows with sliding overlap.
    """
    words = tokenize_words(text)
    if not words:
        return []
    
    if len(words) <= chunk_size_tokens:
        char_span = (0, len(text))
        return [{
            "chunk_id": f"{metadata.get('chunk_id', 'chk')}_fixed_0",
            "text": text,
            "tokens_count": len(words),
            "chunk_strategy": "fixed",
            "char_span": list(char_span),
            "lang": metadata.get("lang"),
            "query_id": metadata.get("query_id"),
            "query": metadata.get("query"),
            "answer": metadata.get("answer"),
            "is_selected": metadata.get("is_selected", 0),
            "source_passage_id": metadata.get("chunk_id")
        }]
    
    step = max(1, int(chunk_size_tokens * (1.0 - overlap_pct)))
    chunks = []
    
    for i in range(0, len(words), step):
        window_words = words[i:i + chunk_size_tokens]
        chunk_text = " ".join(window_words)
        
        # Approximate char span
        start_char = text.find(window_words[0]) if window_words else 0
        end_char = start_char + len(chunk_text)
        
        chunks.append({
            "chunk_id": f"{metadata.get('chunk_id', 'chk')}_fixed_{len(chunks)}",
            "text": chunk_text,
            "tokens_count": len(window_words),
            "chunk_strategy": "fixed",
            "char_span": [max(0, start_char), min(len(text), end_char)],
            "lang": metadata.get("lang"),
            "query_id": metadata.get("query_id"),
            "query": metadata.get("query"),
            "answer": metadata.get("answer"),
            "is_selected": metadata.get("is_selected", 0),
            "source_passage_id": metadata.get("chunk_id")
        })
        
        if i + chunk_size_tokens >= len(words):
            break
            
    return chunks
