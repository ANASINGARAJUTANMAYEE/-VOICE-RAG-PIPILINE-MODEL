import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()
from src.harness.orchestrator import RAGOrchestrator

o = RAGOrchestrator()
hr = o.hybrid_retriever

queries = [
    ('মঙ্গল গ্রহে কতগুলো এলিয়েন শহর আছে?', 'bn', 'Mars / Sci-Fi OOC (Bengali)'),
    ('2099 ஆம் ஆண்டில் யார் உலகை ஆட்சி செய்வார்கள்?', 'ta', 'Year 2099 Speculation OOC (Tamil)'),
    ('What is quantum foam spacetime fluctuation?', 'hi', 'English Sci-fi in Hindi context'),
    ('कॉर्पोरेशन क्या है?', 'hi', 'Corporation In-Corpus (Hindi)'),
    ('সৌরজগতের বৃহত্তম গ্রহ কোনটি?', 'bn', 'Largest planet In-Corpus (Bengali)'),
]

print(f'DEFAULT_CONFIDENCE_THRESHOLD = {o.retrieval_guard.min_confidence_threshold}\n')

for q, lang, desc in queries:
    res = hr.retrieve(query=q, query_lang=lang, top_k=5)
    print(f'=== Query: "{q}" ({desc}) ===')
    print(f'Top confidence score: {res["top_confidence_score"]}')
    print(f'Was reranked: {res["was_reranked"]}')
    for i, r in enumerate(res['results'][:3], 1):
        print(f'  #{i} chunk_id: {r["chunk_id"]} | score: {r["score"]} | text: {r["text"][:80]}...')
    eval_pass, refusal, metrics = o.retrieval_guard.evaluate(res['results'], lang=lang)
    print(f'Guardrail Decision: passed={eval_pass} | reason={metrics.get("reason", "OK")}\n')
