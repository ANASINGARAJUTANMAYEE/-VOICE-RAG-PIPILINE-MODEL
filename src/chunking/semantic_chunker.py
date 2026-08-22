"""
Semantic Chunker
Sentence-boundary aware chunking using indic-nlp-library for Hindi, Bengali, and Tamil.
Packs sentences up to ~200-300 tokens and uses sentence-level boundaries and similarity cut points.
"""

import re
from typing import List, Dict, Any

try:
    from indicnlp.tokenize import sentence_tokenize
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False


def split_sentences_indic(text: str, lang: str) -> List[str]:
    """
    Split Indic text into sentences using indic-nlp-library or regex fallback with poornaviram / danda (|).
    """
    if INDIC_NLP_AVAILABLE and lang in ["hi", "bn", "ta"]:
        try:
            return sentence_tokenize.sentence_split(text, lang=lang)
        except Exception:
            pass
            
    # Fallback regex for Hindi/Bengali danda (।), Tamil period/question, and general punctuation
    sentences = re.split(r'(?<=[।!?.\n])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def calculate_jaccard_similarity(sent1: str, sent2: str) -> float:
    """Computes lexical token overlap similarity between two adjacent sentences."""
    words1 = set(sent1.lower().split())
    words2 = set(sent2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def chunk_semantic(
    text: str,
    metadata: Dict[str, Any],
    min_tokens: int = 150,
    max_tokens: int = 300,
    similarity_threshold: float = 0.08
) -> List[Dict[str, Any]]:
    """
    Splits text along sentence boundaries and groups coherent sentences up to ~200-300 tokens.
    Uses similarity drops between adjacent sentences to choose natural breakpoint boundaries.
    """
    lang = metadata.get("lang", "hi")
    sentences = split_sentences_indic(text, lang)
    
    if not sentences:
        return []
        
    chunks = []
    current_sentences = []
    current_token_count = 0
    
    for i, sent in enumerate(sentences):
        sent_tokens = len(sent.split())
        
        # Check if adding this sentence exceeds maximum token capacity
        if current_token_count + sent_tokens > max_tokens and current_sentences:
            chunk_text = " ".join(current_sentences)
            start_char = text.find(current_sentences[0])
            end_char = start_char + len(chunk_text)
            
            chunks.append({
                "chunk_id": f"{metadata.get('chunk_id', 'chk')}_semantic_{len(chunks)}",
                "text": chunk_text,
                "tokens_count": current_token_count,
                "chunk_strategy": "semantic",
                "char_span": [max(0, start_char), min(len(text), end_char)],
                "lang": lang,
                "query_id": metadata.get("query_id"),
                "query": metadata.get("query"),
                "answer": metadata.get("answer"),
                "is_selected": metadata.get("is_selected", 0),
                "source_passage_id": metadata.get("chunk_id")
            })
            current_sentences = [sent]
            current_token_count = sent_tokens
            continue
            
        # If we have reached minimum token threshold, evaluate semantic similarity cut point
        if current_token_count >= min_tokens and i + 1 < len(sentences):
            sim = calculate_jaccard_similarity(sent, sentences[i + 1])
            # If similarity drops below threshold, this is a natural topic transition point
            if sim < similarity_threshold:
                current_sentences.append(sent)
                current_token_count += sent_tokens
                chunk_text = " ".join(current_sentences)
                start_char = text.find(current_sentences[0])
                end_char = start_char + len(chunk_text)
                
                chunks.append({
                    "chunk_id": f"{metadata.get('chunk_id', 'chk')}_semantic_{len(chunks)}",
                    "text": chunk_text,
                    "tokens_count": current_token_count,
                    "chunk_strategy": "semantic",
                    "char_span": [max(0, start_char), min(len(text), end_char)],
                    "lang": lang,
                    "query_id": metadata.get("query_id"),
                    "query": metadata.get("query"),
                    "answer": metadata.get("answer"),
                    "is_selected": metadata.get("is_selected", 0),
                    "source_passage_id": metadata.get("chunk_id")
                })
                current_sentences = []
                current_token_count = 0
                continue

        current_sentences.append(sent)
        current_token_count += sent_tokens
        
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        start_char = text.find(current_sentences[0])
        end_char = start_char + len(chunk_text)
        
        chunks.append({
            "chunk_id": f"{metadata.get('chunk_id', 'chk')}_semantic_{len(chunks)}",
            "text": chunk_text,
            "tokens_count": current_token_count,
            "chunk_strategy": "semantic",
            "char_span": [max(0, start_char), min(len(text), end_char)],
            "lang": lang,
            "query_id": metadata.get("query_id"),
            "query": metadata.get("query"),
            "answer": metadata.get("answer"),
            "is_selected": metadata.get("is_selected", 0),
            "source_passage_id": metadata.get("chunk_id")
        })
        
    return chunks
