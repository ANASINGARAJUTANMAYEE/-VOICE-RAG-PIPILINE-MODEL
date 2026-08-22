"""
Pydantic Data Models
Defines typed contracts across all pipeline stages: STT, Input Guard, Retrieval, Generation, Grounding Guard, and Orchestrator.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class StageLatency(BaseModel):
    stt_ms: float = Field(default=0.0, description="Speech-to-text transcription latency")
    input_guard_ms: float = Field(default=0.0, description="Input guardrail check latency")
    retrieval_ms: float = Field(default=0.0, description="Hybrid retrieval (embed + FAISS + BM25 + RRF + rerank) latency (<200ms target)")
    retrieval_guard_ms: float = Field(default=0.0, description="Retrieval confidence guardrail check latency")
    generation_ms: float = Field(default=0.0, description="LLM generation latency")
    grounding_guard_ms: float = Field(default=0.0, description="Grounding & citation check latency")
    total_e2e_ms: float = Field(default=0.0, description="Total end-to-end latency")


class PassageSource(BaseModel):
    chunk_id: str
    text: str
    score: float
    lang: str
    chunk_strategy: str
    query_id: Optional[Any] = None
    is_selected: int = 0
    tokens_count: int = 0
    char_span: List[int] = [0, 0]


class GuardrailStatus(BaseModel):
    input_guard_passed: bool = True
    retrieval_guard_passed: bool = True
    grounding_guard_passed: bool = True
    confidence_level: str = "High"
    refusal_reason: Optional[str] = None


class PipelineResponse(BaseModel):
    query_text: str
    answer: str
    language: str
    confidence: str
    is_refusal: bool
    sources: List[PassageSource] = []
    latency_breakdown: StageLatency
    guardrails: GuardrailStatus
