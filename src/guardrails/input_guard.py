"""
Input Guardrail Module
Performs pre-retrieval safety, gibberish, and off-topic checks on user query transcripts.
Short-circuits invalid or malicious inputs before reaching the vector DB or LLM.
"""

import re
from typing import Dict, Any, Tuple
from src.generation.prompt_templates import REFUSAL_MESSAGES

# Multilingual toxicity / unsafe keywords filter (hi, bn, ta, en)
UNSAFE_KEYWORDS = {
    "bomb", "kill", "attack", "hack", "weapon", "terror",
    "हत्या", "मारना", "बम", "हमला", "विस्फोट",
    "হত্যা", "বোমা", "আক্রমণ",
    "கொலை", "தாக்குதல்", "வெடிகுண்டு"
}

# Generic off-topic blacklist (e.g. system prompt extraction, jailbreaks)
JAILBREAK_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"reveal secret",
    r"drop database",
    r"delete from",
    r"<script>"
]


class InputGuard:
    def __init__(self, min_query_length: int = 3):
        self.min_query_length = min_query_length

    def evaluate(self, query: str, lang: str = "hi") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluates input query.
        Returns: (passed: bool, reason_or_refusal: str, metrics: dict)
        """
        clean_query = query.strip()
        
        # Check 1: Empty or too short
        if len(clean_query) < self.min_query_length:
            refusal = "Query is too short or empty."
            return False, refusal, {"check": "length", "passed": False}

        # Check 2: Gibberish / non-speech repetition (e.g. "aaaaa" or "?????")
        if len(set(clean_query)) <= 2 and len(clean_query) > 5:
            refusal = "Gibberish or repetitive input detected."
            return False, refusal, {"check": "gibberish", "passed": False}

        # Check 3: Basic safety & toxicity check
        lower_query = clean_query.lower()
        for kw in UNSAFE_KEYWORDS:
            if kw in lower_query:
                refusal = "Inappropriate or unsafe content detected. Request refused."
                return False, refusal, {"check": "safety", "passed": False, "keyword": kw}

        # Check 4: Jailbreak / Prompt Injection patterns
        for pattern in JAILBREAK_PATTERNS:
            if re.search(pattern, lower_query):
                refusal = "Security violation detected. Request refused."
                return False, refusal, {"check": "jailbreak", "passed": False}

        return True, "", {"check": "all", "passed": True}
