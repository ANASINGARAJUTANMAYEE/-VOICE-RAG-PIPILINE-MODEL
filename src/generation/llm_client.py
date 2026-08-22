"""
LLM Generation Client
Integrates with Google Gemini (gemini-1.5-flash) for fast, grounded, multilingual answer synthesis.
Wrapped in tenacity retries with exponential backoff and structured latency tracking.
"""

import os
import time
from typing import List, Dict, Any, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.generation.prompt_templates import build_system_prompt, build_user_prompt, REFUSAL_MESSAGES

GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.1-flash-lite"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
        reraise=True
    )
    def _call_gemini_api(self, messages: List[Dict[str, str]], max_tokens: int = 350, temperature: float = 0.1) -> str:
        # Gemini has no separate "system" role in the basic generateContent API.
        # Fold the system prompt into the first user turn instead.
        system_text = ""
        user_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                user_text = m["content"]

        combined_prompt = f"{system_text}\n\n{user_text}" if system_text else user_text

        url = GEMINI_URL_TEMPLATE.format(model=self.model)
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": combined_prompt}]}
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }

        response = requests.post(url, headers=headers, params=params, json=payload, timeout=15.0)

        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                # Gemini can return no candidates if blocked by safety filters
                raise ValueError(f"Gemini returned no candidates. Full response: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError(f"Gemini candidate had no content parts: {candidates[0]}")
            return parts[0]["text"].strip()
        elif response.status_code in (401, 403):
            raise PermissionError("Gemini API authentication failed. Invalid API key.")
        else:
            response.raise_for_status()

    def generate_answer(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        lang: str = "hi",
        is_strict_reprompt: bool = False
    ) -> Dict[str, Any]:
        """
        Generates a grounded response citing retrieved passages.
        Returns:
        {
            "answer": str,
            "latency_ms": float,
            "cited_sources": List[int],
            "is_refusal": bool,
            "status": "success" | "mock" | "error"
        }
        """
        t0 = time.perf_counter()

        system_prompt = build_system_prompt(lang=lang)
        if is_strict_reprompt:
            system_prompt += "\n\nCRITICAL WARNING: Previous response lacked source grounding. You MUST explicitly quote and cite facts directly from [Source X]."

        user_prompt = build_user_prompt(query=query, passages=retrieved_passages, lang=lang)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Check API key
        if not self.api_key:
            # Mock generator fallback for offline local testing
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if retrieved_passages:
                top_text = retrieved_passages[0].get("text", "")[:120]
                answer = f"[Source 1] के अनुसार: {top_text}..." if lang == "hi" else f"[Source 1]: {top_text}..."
            else:
                answer = REFUSAL_MESSAGES.get(lang, REFUSAL_MESSAGES["hi"])

            return {
                "answer": answer,
                "latency_ms": latency_ms,
                "cited_sources": [1] if retrieved_passages else [],
                "is_refusal": not bool(retrieved_passages),
                "status": "mock"
            }

        try:
            answer = self._call_gemini_api(messages)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            # Check if answer is a refusal
            refusal_text = REFUSAL_MESSAGES.get(lang, REFUSAL_MESSAGES["hi"])
            is_refusal = (refusal_text in answer) or ("पर्याप्त जानकारी उपलब्ध नहीं" in answer) or ("পর্যাপ্ত তথ্য নেই" in answer) or ("போதுமான தகவல் இல்லை" in answer)

            # Extract cited source IDs (e.g. [Source 1], [Source 2])
            cited_sources = []
            for i in range(1, len(retrieved_passages) + 1):
                if f"Source {i}" in answer or f"source {i}" in answer:
                    cited_sources.append(i)

            return {
                "answer": answer,
                "latency_ms": latency_ms,
                "cited_sources": cited_sources,
                "is_refusal": is_refusal,
                "status": "success"
            }

        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            # Graceful fallback: return top retrieved passage directly instead of empty error
            fallback_text = retrieved_passages[0].get("text", "") if retrieved_passages else REFUSAL_MESSAGES.get(lang, "")
            return {
                "answer": f"[Context Fallback] [Source 1]: {fallback_text}",
                "latency_ms": latency_ms,
                "cited_sources": [1] if retrieved_passages else [],
                "is_refusal": False,
                "status": "error",
                "error_message": str(e)
            }