"""
Grounding & Anti-Hallucination Guardrail Module
Evaluates generated answers against retrieved context passages for source attribution and lexical/claim overlap.
Tags responses with High, Medium, or Low (Not Fully Grounded) confidence badges.
"""

from typing import List, Dict, Any, Tuple


class GroundingGuard:
    def __init__(self, min_token_overlap_ratio: float = 0.20):
        self.min_token_overlap_ratio = min_token_overlap_ratio

    def evaluate(
        self,
        answer: str,
        retrieved_passages: List[Dict[str, Any]],
        cited_sources: List[int]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluates answer grounding against cited passages.
        Returns: (is_well_grounded: bool, confidence_level: str, metrics: dict)
        confidence_level: 'High' | 'Medium' | 'Low (Not Fully Grounded)' | 'Refusal'
        """
        clean_answer = answer.strip()

        # Check 0: Context Fallback responses are literal raw quotes without generative synthesis
        if clean_answer.startswith("[Context Fallback]"):
            return False, "Low (Context Fallback)", {
                "overlap_ratio": 1.0,
                "has_citations": True,
                "is_fallback": True,
                "reason": "Direct passage fallback without generative synthesis"
            }

        # Check 1: Refusal answers are considered safely grounded refusals
        if not cited_sources and ("पर्याप्त जानकारी उपलब्ध नहीं" in clean_answer or "পর্যাপ্ত তথ্য নেই" in clean_answer or "போதுமான தகவல் இல்லை" in clean_answer or "not enough information" in clean_answer.lower()):
            return True, "Refusal (Safe)", {"overlap_ratio": 1.0, "has_citations": False, "is_refusal": True}

        if not retrieved_passages:
            return False, "Low (Not Fully Grounded)", {"overlap_ratio": 0.0, "has_citations": False}

        # Check 2: Check if sources were explicitly cited
        has_citations = len(cited_sources) > 0 or ("[Source" in clean_answer) or ("[source" in clean_answer)

        # Collect text of cited passages (or all top passages if citations were omitted)
        cited_texts = []
        if cited_sources:
            for src_idx in cited_sources:
                if 1 <= src_idx <= len(retrieved_passages):
                    cited_texts.append(retrieved_passages[src_idx - 1].get("text", ""))
        else:
            cited_texts = [p.get("text", "") for p in retrieved_passages[:2]]

        combined_context = " ".join(cited_texts).lower()
        
        # Token overlap ratio calculation
        answer_words = set(clean_answer.lower().split())
        # Filter common punctuation and short stopwords
        meaningful_answer_words = {w for w in answer_words if len(w) > 2 and not w.startswith("[source")}
        
        if not meaningful_answer_words:
            overlap_ratio = 1.0
        else:
            context_words = set(combined_context.split())
            matching_words = meaningful_answer_words.intersection(context_words)
            overlap_ratio = len(matching_words) / len(meaningful_answer_words)

        metrics = {
            "overlap_ratio": round(overlap_ratio, 3),
            "has_citations": has_citations,
            "cited_sources_count": len(cited_sources)
        }

        # Determine grounding badge level
        if has_citations and overlap_ratio >= self.min_token_overlap_ratio:
            return True, "High", metrics
        elif has_citations or overlap_ratio >= (self.min_token_overlap_ratio * 0.75):
            return True, "Medium", metrics
        else:
            return False, "Low (Not Fully Grounded)", metrics
