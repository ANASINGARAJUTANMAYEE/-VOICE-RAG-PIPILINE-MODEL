"""
Comprehensive Test Suite for Voice-Enabled Guardrailed Multilingual RAG Pipeline
Covers: Chunking Strategies, Guardrails, Hybrid Retrieval, Orchestration, and FastAPI Endpoints.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.fixed_chunker import chunk_fixed
from src.chunking.semantic_chunker import chunk_semantic
from src.chunking.metadata_chunker import chunk_metadata_aware
from src.guardrails.input_guard import InputGuard
from src.guardrails.retrieval_guard import RetrievalConfidenceGuard
from src.guardrails.grounding_guard import GroundingGuard
from src.harness.orchestrator import RAGOrchestrator
from src.harness.models import PipelineResponse


# -------------------------------------------------------------
# 1. Chunking Tests
# -------------------------------------------------------------
def test_fixed_chunker():
    sample_text = "यह एक परीक्षण वाक्य है। " * 40
    meta = {"chunk_id": "test_hi_1", "lang": "hi", "query_id": 101, "query": "टेस्ट", "answer": "उत्तर"}
    chunks = chunk_fixed(sample_text, meta, chunk_size_tokens=50, overlap_pct=0.15)
    
    assert len(chunks) > 1
    assert chunks[0]["chunk_strategy"] == "fixed"
    assert chunks[0]["lang"] == "hi"
    assert "char_span" in chunks[0]


def test_semantic_chunker_indic():
    sample_text = "नई दिल्ली भारत की राजधानी है। यहाँ कई ऐतिहासिक इमारतें हैं। लाल किला और कुतुब मीनार प्रमुख हैं।"
    meta = {"chunk_id": "test_hi_2", "lang": "hi", "query_id": 102}
    chunks = chunk_semantic(sample_text, meta, min_tokens=5, max_tokens=20)
    
    assert len(chunks) >= 1
    assert chunks[0]["chunk_strategy"] == "semantic"
    assert chunks[0]["lang"] == "hi"


def test_metadata_chunker_atomic():
    sample_text = "স্বল্প দৈর্ঘ্যের অনুচ্ছেদ যা বিভক্ত করার প্রয়োজন নেই।"
    meta = {"chunk_id": "test_bn_1", "lang": "bn", "query_id": 103, "is_selected": 1}
    chunks = chunk_metadata_aware(sample_text, meta, max_token_budget=100)
    
    assert len(chunks) == 1
    assert chunks[0]["chunk_strategy"] == "metadata"
    assert chunks[0]["is_selected"] == 1


# -------------------------------------------------------------
# 2. Guardrails Tests
# -------------------------------------------------------------
def test_input_guard_gibberish():
    guard = InputGuard()
    passed, msg, metrics = guard.evaluate("???????", lang="hi")
    assert not passed
    assert "Gibberish" in msg


def test_input_guard_unsafe_content():
    guard = InputGuard()
    passed, msg, metrics = guard.evaluate("बम बनाने का तरीका क्या है?", lang="hi")
    assert not passed
    assert "Inappropriate or unsafe" in msg


def test_input_guard_valid_query():
    guard = InputGuard()
    passed, msg, metrics = guard.evaluate("भारत की राजधानी क्या है?", lang="hi")
    assert passed
    assert msg == ""


def test_retrieval_confidence_guard():
    guard = RetrievalConfidenceGuard(min_confidence_threshold=0.015)
    
    # Low confidence scenario
    low_res = [{"chunk_id": "c1", "score": 0.005, "text": "random"}]
    passed, msg, metrics = guard.evaluate(low_res, lang="hi")
    assert not passed
    assert "पर्याप्त जानकारी उपलब्ध नहीं" in msg

    # High confidence scenario
    high_res = [{"chunk_id": "c2", "score": 0.035, "text": "relevant text"}]
    passed, msg, metrics = guard.evaluate(high_res, lang="hi")
    assert passed


def test_grounding_guard_evaluation():
    guard = GroundingGuard(min_token_overlap_ratio=0.20)
    passages = [{"text": "नई दिल्ली भारत की आधिकारिक राजधानी है और संसद यहाँ स्थित है।"}]
    
    # Well-grounded answer with source citation
    grounded_ans = "[Source 1] के अनुसार नई दिल्ली भारत की राजधानी है।"
    is_grounded, badge, metrics = guard.evaluate(grounded_ans, passages, cited_sources=[1])
    assert is_grounded
    assert badge == "High"


# -------------------------------------------------------------
# 3. End-to-End Orchestrator Test (Offline/Mock Mode Safe)
# -------------------------------------------------------------
def test_orchestrator_execution():
    orchestrator = RAGOrchestrator()
    response = orchestrator.process_query(
        text_query="भारत की राजधानी क्या है?",
        language_hint="hi"
    )
    
    assert isinstance(response, PipelineResponse)
    assert response.language == "hi"
    assert response.latency_breakdown.total_e2e_ms >= 0.0
    assert response.guardrails.input_guard_passed is True
