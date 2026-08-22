"""
Dataset Ingestion Module
Fetches real multilingual MS MARCO translated records from Hugging Face for
Hindi (hi), Bengali (bn), and Tamil (ta) using DuckDB HTTPFS parquet streaming.
Outputs ~4200-4500 balanced records to data/data_subset.jsonl.
"""

import os
import sys
import json
from pathlib import Path
import duckdb

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUT_PATH = Path("data/data_subset.jsonl")

# Verified parquet URLs on Hugging Face (ai4bharat/MSMARCO-XI validation split)
LANG_FILES = {
    "hi": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet",
    "bn": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/benval.parquet",
    "ta": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/tamval.parquet",
}

def load_data(limit_per_lang: int = 1400):
    print("================================================================")
    print("STEP 1: Ingesting Real Multilingual Data from Hugging Face")
    print("Dataset: ai4bharat/MSMARCO-XI (Validation Split)")
    print(f"Languages: {list(LANG_FILES.keys())} (~{limit_per_lang} records each)")
    print("================================================================")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    all_records = []
    for lang, url in LANG_FILES.items():
        print(f"\n[+] Fetching {lang.upper()} records from: {url} ...", flush=True)
        query = f"SELECT * FROM read_parquet('{url}') LIMIT {limit_per_lang}"
        rows = con.execute(query).fetchall()
        cols = [desc[0] for desc in con.description]
        
        for row in rows:
            rec = dict(zip(cols, row))
            rec["lang"] = lang  # Normalize language tag (hi, bn, ta)
            all_records.append(rec)
        
        print(f"    [✓] Fetched {len(rows)} records for '{lang}'.", flush=True)

    assert len(all_records) >= 4000, f"Expected ~4200 records, got {len(all_records)} — check parquet fetch"

    print(f"\n[+] Writing {len(all_records)} records to {OUTPUT_PATH} ...", flush=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"\n[✓] Successfully saved {len(all_records)} records to {OUTPUT_PATH}!")
    print(f"Total records: {len(all_records)}")
    if all_records:
        print("Columns:", list(all_records[0].keys()))
        print("\n--- First Record Sample ---")
        sample = {k: (str(v)[:120] + "..." if len(str(v)) > 120 else v) for k, v in all_records[0].items()}
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        print("---------------------------")

    return all_records

if __name__ == "__main__":
    records = load_data(limit_per_lang=1400)
