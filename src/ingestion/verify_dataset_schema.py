"""
Dataset Verification & Schema Inspection Tool
Performs live sanity checks against the AI4Bharat MS MARCO-XI Parquet corpus
on Hugging Face for Hindi (hi), Bengali (bn), and Tamil (ta).
"""

import sys
import json
import duckdb

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

LANG_FILES = {
    "hi": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet",
    "bn": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/benval.parquet",
    "ta": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/tamval.parquet",
}

def verify_dataset_schema():
    print("================================================================")
    print("AI4Bharat MS MARCO-XI Dataset Schema & Remote Connectivity Check")
    print("================================================================")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    for lang, url in LANG_FILES.items():
        print(f"\n[+] Verifying language: '{lang}' from {url.split('/')[-1]}...")
        try:
            df = con.execute(f"""
                SELECT query_id, query, Answer, Eng_Query, Eng_Answer,
                       passages.Translated_passages[1] as first_passage
                FROM '{url}'
                LIMIT 1
            """).df()

            rec = df.to_dict(orient="records")[0]
            print(f"    [✓] Query ID:       {rec.get('query_id')}")
            print(f"    [✓] English Query:  {rec.get('Eng_Query')}")
            print(f"    [✓] Native Query:   {rec.get('query')}")
            print(f"    [✓] English Answer: {rec.get('Eng_Answer')}")
            print(f"    [✓] Native Answer:  {rec.get('Answer')}")
            passage = str(rec.get("first_passage", ""))[:120]
            print(f"    [✓] First Passage:  {passage}...")
        except Exception as e:
            print(f"    [✗] Verification failed for '{lang}': {e}")
            return False

    print("\n================================================================")
    print("[✓] All dataset endpoints and schemas verified successfully!")
    print("================================================================")
    return True

if __name__ == "__main__":
    verify_dataset_schema()
