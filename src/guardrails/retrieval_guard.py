"""
Retrieval Confidence Guardrail Module
Evaluates top retrieval candidates against a calibrated confidence threshold
and source semantic coherence threshold.
Short-circuits LLM generation and directly returns "insufficient context" if:
1. Top candidate score is below min_confidence_threshold (0.11), OR
2. Pairwise source coherence is below min_source_coherence_threshold (0.25).
"""

from typing import List, Dict, Any, Tuple, Optional
from src.generation.prompt_templates import REFUSAL_MESSAGES

# Calibrated minimum threshold for top candidate RRF / hybrid score
# Out-of-corpus queries score <= 0.107; genuine in-corpus queries score >= 0.113
DEFAULT_CONFIDENCE_THRESHOLD = 0.11

# Calibrated minimum threshold for source passage semantic coherence (pairwise avg cosine)
# Fragmented/hallucinatory retrievals score < 0.10 (e.g. -0.12); genuine in-corpus topical clusters score >= 0.55
DEFAULT_SOURCE_COHERENCE_THRESHOLD = 0.25


class RetrievalConfidenceGuard:
    def __init__(
        self,
        min_confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_source_coherence_threshold: float = DEFAULT_SOURCE_COHERENCE_THRESHOLD
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_source_coherence_threshold = min_source_coherence_threshold

    def evaluate(
        self,
        retrieval_results: List[Dict[str, Any]],
        lang: str = "hi",
        source_coherence_score: Optional[float] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluates retrieval output before invoking generation.
        Returns: (passed: bool, refusal_message: str, metrics: dict)
        """
        if not retrieval_results:
            refusal = REFUSAL_MESSAGES.get(lang, REFUSAL_MESSAGES["hi"])
            return False, refusal, {"reason": "empty_retrieval", "top_score": 0.0, "source_coherence": 0.0, "passed": False}

        top_score = retrieval_results[0].get("score", 0.0)

        # Check if top score meets calibrated confidence
        if top_score < self.min_confidence_threshold:
            refusal = REFUSAL_MESSAGES.get(lang, REFUSAL_MESSAGES["hi"])
            return False, refusal, {
                "reason": "low_retrieval_confidence",
                "top_score": top_score,
                "threshold": self.min_confidence_threshold,
                "passed": False
            }

        # Check if source coherence meets semantic agreement threshold
        coherence = source_coherence_score if source_coherence_score is not None else 1.0
        if coherence < self.min_source_coherence_threshold:
            refusal = REFUSAL_MESSAGES.get(lang, REFUSAL_MESSAGES["hi"])
            return False, refusal, {
                "reason": "low_source_coherence",
                "top_score": top_score,
                "source_coherence": coherence,
                "coherence_threshold": self.min_source_coherence_threshold,
                "passed": False
            }

        return True, "", {
            "top_score": top_score,
            "threshold": self.min_confidence_threshold,
            "source_coherence": coherence,
            "passed": True
        }

