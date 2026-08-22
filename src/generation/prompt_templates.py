"""
Strict Grounding Prompt Templates
Enforces citation of source passages [Source X] and strict language alignment (Hindi, Bengali, Tamil).
Instructs graceful fallback when context is insufficient.
"""

from typing import List, Dict, Any

REFUSAL_MESSAGES = {
    "hi": "प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।",
    "bn": "প্রদত্ত প্রসঙ্গে এই প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য নেই।",
    "ta": "வழங்கப்பட்ட சூழலில் இந்த கேள்விக்கு பதிலளிக்க போதுமான தகவல் இல்லை।",
    "en": "I do not have enough information in the provided context to answer this question."
}

LANGUAGE_NAMES = {
    "hi": "Hindi (हिन्दी)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)"
}


def build_system_prompt(lang: str = "hi") -> str:
    lang_name = LANGUAGE_NAMES.get(lang, "Hindi")
    refusal = REFUSAL_MESSAGES.get(lang, REFUSAL_MESSAGES["hi"])

    return f"""You are a helpful, strictly grounded AI knowledge assistant specialized in {lang_name}.

CRITICAL GROUNDING RULES:
1. You MUST answer the user's question ONLY using the facts directly stated in the provided [Source X] context passages below.
2. DO NOT assume, extrapolate, or use external knowledge not present in the provided sources.
3. Every claim in your answer MUST include an explicit bracket citation, e.g. [Source 1], [Source 2].
4. Your response language MUST be strictly in {lang_name}.
5. If the provided context passages DO NOT contain sufficient information to answer the question accurately, you MUST reply with exactly:
   "{refusal}"
6. Keep your response concise, factual, and clear (max 3-5 sentences)."""


def format_context_passages(passages: List[Dict[str, Any]]) -> str:
    if not passages:
        return "No relevant passages found."
        
    formatted = []
    for idx, p in enumerate(passages, start=1):
        text = p.get("text", "").strip()
        lang = p.get("lang", "").upper()
        strategy = p.get("chunk_strategy", "default")
        score = p.get("score", 0.0)
        formatted.append(f"[Source {idx}] (Lang: {lang}, Strategy: {strategy}, Score: {score:.3f}):\n{text}")
        
    return "\n\n".join(formatted)


def build_user_prompt(query: str, passages: List[Dict[str, Any]], lang: str = "hi") -> str:
    context_str = format_context_passages(passages)
    return f"""Context Passages:
{context_str}

User Question ({lang.upper()}):
{query}

Answer (Strictly in {LANGUAGE_NAMES.get(lang, 'Hindi')}, cited with [Source X]):"""
