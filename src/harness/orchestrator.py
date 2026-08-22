"""
Pipeline Orchestrator Module
Coordinates the end-to-end execution sequence:
1. Speech-to-Text (Sarvam STT)
2. Input Guardrail
3. Hybrid Retrieval (FAISS + BM25 + RRF + Conditional Rerank)
4. Retrieval Confidence Guardrail
5. LLM Generation (Google Gemini gemini-3.1-flash-lite with Tenacity retries)
6. Grounding & Anti-Hallucination Guardrail
"""

import sys
import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure UTF-8 for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.harness.models import PipelineResponse, StageLatency, GuardrailStatus, PassageSource
from src.speech.stt import SarvamSTTClient
from src.guardrails.input_guard import InputGuard
from src.retrieval.embed import get_embedder
from src.retrieval.vector_store import FaissVectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.guardrails.retrieval_guard import RetrievalConfidenceGuard
from src.generation.llm_client import LLMClient
from src.guardrails.grounding_guard import GroundingGuard


class RAGOrchestrator:
    def __init__(
        self,
        vector_store_path: Path = Path("data/faiss_index.bin"),
        vector_meta_path: Path = Path("data/faiss_metadata.json"),
        bm25_corpus_path: Path = Path("data/bm25_corpus.json")
    ):
        print("[+] Initializing RAG Orchestrator...", flush=True)
        self.stt_client = SarvamSTTClient()
        self.input_guard = InputGuard()
        self.retrieval_guard = RetrievalConfidenceGuard()
        self.llm_client = LLMClient()
        self.grounding_guard = GroundingGuard()

        # Load retrieval stores if available
        self.hybrid_retriever: Optional[HybridRetriever] = None
        if vector_store_path.exists() and vector_meta_path.exists() and bm25_corpus_path.exists():
            try:
                vstore = FaissVectorStore.load(vector_store_path, vector_meta_path)
                bstore = BM25Store.load(bm25_corpus_path)
                embedder = get_embedder()
                self.hybrid_retriever = HybridRetriever(vector_store=vstore, bm25_store=bstore, embedder=embedder)
                print(f"[+] Hybrid Retriever ready with {vstore.total_vectors} vectors.", flush=True)
            except Exception as e:
                print(f"[!] Warning: Could not initialize hybrid retriever: {e}", flush=True)
        else:
            print("[!] Note: Index files not found yet. Indices will be loaded once created.", flush=True)

    def process_query(
        self,
        audio_bytes: Optional[bytes] = None,
        text_query: Optional[str] = None,
        language_hint: str = "hi",
        audio_filename: str = "audio.webm",
        top_k: int = 5
    ) -> PipelineResponse:
        """
        Executes the full guardrailed RAG pipeline.
        """
        t_e2e_start = time.perf_counter()
        latencies = StageLatency()
        guard_status = GuardrailStatus()

        # -------------------------------------------------------------
        # STAGE 1: Speech-to-Text (if audio provided)
        # -------------------------------------------------------------
        query_text = (text_query or "").strip()
        detected_lang = language_hint[:2].lower()

        if audio_bytes:
            t0_stt = time.perf_counter()
            stt_res = self.stt_client.transcribe(
                audio_bytes=audio_bytes,
                filename=audio_filename,
                language_hint=language_hint
            )
            latencies.stt_ms = stt_res["latency_ms"]
            detected_lang = stt_res["detected_language"]
            
            if stt_res["status"] != "success":
                # Handle STT failure mode gracefully
                refusal_msg = stt_res.get("error_message") or "Speech recognition could not detect clear audio."
                latencies.total_e2e_ms = round((time.perf_counter() - t_e2e_start) * 1000.0, 2)
                guard_status.input_guard_passed = False
                guard_status.refusal_reason = refusal_msg
                guard_status.confidence_level = "Refusal (Audio Error)"
                
                return PipelineResponse(
                    query_text="[Audio Transcription Failed]",
                    answer=refusal_msg,
                    language=detected_lang,
                    confidence=guard_status.confidence_level,
                    is_refusal=True,
                    sources=[],
                    latency_breakdown=latencies,
                    guardrails=guard_status
                )
                
            query_text = stt_res["transcript"]

        # -------------------------------------------------------------
        # STAGE 2: Input Guardrail
        # -------------------------------------------------------------
        t0_in_guard = time.perf_counter()
        in_passed, in_refusal, in_metrics = self.input_guard.evaluate(query_text, lang=detected_lang)
        latencies.input_guard_ms = round((time.perf_counter() - t0_in_guard) * 1000.0, 2)

        if not in_passed:
            latencies.total_e2e_ms = round((time.perf_counter() - t_e2e_start) * 1000.0, 2)
            guard_status.input_guard_passed = False
            guard_status.refusal_reason = in_refusal
            guard_status.confidence_level = "Refusal (Input Guard)"

            return PipelineResponse(
                query_text=query_text,
                answer=in_refusal,
                language=detected_lang,
                confidence=guard_status.confidence_level,
                is_refusal=True,
                sources=[],
                latency_breakdown=latencies,
                guardrails=guard_status
            )

        # -------------------------------------------------------------
        # STAGE 3: Sub-200ms Hybrid Retrieval
        # -------------------------------------------------------------
        retrieved_passages: List[Dict[str, Any]] = []
        if self.hybrid_retriever:
            retrieval_res = self.hybrid_retriever.retrieve(
                query=query_text,
                query_lang=detected_lang,
                top_k=top_k
            )
            retrieved_passages = retrieval_res["results"]
            latencies.retrieval_ms = retrieval_res["latency"]["total_retrieval_ms"]
        else:
            latencies.retrieval_ms = 0.0

        # -------------------------------------------------------------
        # STAGE 4: Retrieval Confidence Guardrail
        # -------------------------------------------------------------
        t0_ret_guard = time.perf_counter()
        source_coherence = retrieval_res.get("source_coherence_score", 1.0) if self.hybrid_retriever else 1.0
        ret_passed, ret_refusal, ret_metrics = self.retrieval_guard.evaluate(
            retrieval_results=retrieved_passages,
            lang=detected_lang,
            source_coherence_score=source_coherence
        )
        latencies.retrieval_guard_ms = round((time.perf_counter() - t0_ret_guard) * 1000.0, 2)

        if not ret_passed:
            latencies.total_e2e_ms = round((time.perf_counter() - t_e2e_start) * 1000.0, 2)
            guard_status.retrieval_guard_passed = False
            guard_status.refusal_reason = "Out of corpus / insufficient retrieval confidence"
            guard_status.confidence_level = "Refusal (Low Confidence)"

            sources_models = [PassageSource(**p) for p in retrieved_passages]
            return PipelineResponse(
                query_text=query_text,
                answer=ret_refusal,
                language=detected_lang,
                confidence=guard_status.confidence_level,
                is_refusal=True,
                sources=sources_models,
                latency_breakdown=latencies,
                guardrails=guard_status
            )

        # -------------------------------------------------------------
        # STAGE 5: LLM Generation
        # -------------------------------------------------------------
        t0_gen = time.perf_counter()
        gen_res = self.llm_client.generate_answer(
            query=query_text,
            retrieved_passages=retrieved_passages,
            lang=detected_lang
        )
        latencies.generation_ms = gen_res["latency_ms"]
        answer_text = gen_res["answer"]
        cited_sources = gen_res.get("cited_sources", [])

        # -------------------------------------------------------------
        # STAGE 6: Grounding Guardrail
        # -------------------------------------------------------------
        t0_ground = time.perf_counter()
        grounded, conf_badge, ground_metrics = self.grounding_guard.evaluate(
            answer=answer_text,
            retrieved_passages=retrieved_passages,
            cited_sources=cited_sources
        )
        latencies.grounding_guard_ms = round((time.perf_counter() - t0_ground) * 1000.0, 2)
        guard_status.grounding_guard_passed = grounded
        guard_status.confidence_level = conf_badge

        # Final end-to-end latency
        latencies.total_e2e_ms = round((time.perf_counter() - t_e2e_start) * 1000.0, 2)

        sources_models = [PassageSource(**p) for p in retrieved_passages]

        return PipelineResponse(
            query_text=query_text,
            answer=answer_text,
            language=detected_lang,
            confidence=conf_badge,
            is_refusal=gen_res.get("is_refusal", False),
            sources=sources_models,
            latency_breakdown=latencies,
            guardrails=guard_status
        )
