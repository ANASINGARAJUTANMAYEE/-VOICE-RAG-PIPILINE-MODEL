"""
Latency Benchmark Harness
Runs automated evaluation over 50-100 real queries across Hindi, Bengali, and Tamil.
Records per-stage latency (STT, Retrieval, Rerank, Generation, Total) into eval/latency_results.csv.
Computes and prints P50, P70, P100 percentiles.
"""

import sys
import os
import csv
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so OPENAI_API_KEY and SARVAM_API_KEY are available
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.harness.orchestrator import RAGOrchestrator


def run_latency_benchmark(
    samples_count: int = 60,
    data_path: Path = Path("data/data_subset.jsonl"),
    output_csv: Path = Path("eval/latency_results.csv")
):
    print("=" * 70)
    print("STEP 9: Latency Benchmark Harness (P50 / P70 / P100 Evaluation)")
    print(f"Target: {samples_count} queries across Hindi, Bengali, and Tamil")
    print("=" * 70, flush=True)

    if not data_path.exists():
        raise FileNotFoundError(f"'{data_path}' not found. Please run data ingestion first.")

    # Load dataset queries
    queries_by_lang = {"hi": [], "bn": [], "ta": []}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q = item.get("query", "").strip()
                lang = item.get("lang", "hi")
                if q and lang in queries_by_lang and q not in queries_by_lang[lang]:
                    queries_by_lang[lang].append(q)

    # Sample queries equally per language
    samples_per_lang = samples_count // 3
    test_suite = []
    for l in ["hi", "bn", "ta"]:
        available = queries_by_lang[l]
        sampled = random.sample(available, min(len(available), samples_per_lang))
        for q in sampled:
            test_suite.append({"query": q, "lang": l})

    random.shuffle(test_suite)
    print(f"[+] Assembled test suite of {len(test_suite)} queries across HI, BN, TA.", flush=True)

    # Initialize Orchestrator
    orchestrator = RAGOrchestrator()

    # Benchmark results list
    records = []
    
    print("\n[+] Executing benchmark runs...", flush=True)
    for idx, test_case in enumerate(test_suite, start=1):
        q = test_case["query"]
        lang = test_case["lang"]

        resp = orchestrator.process_query(text_query=q, language_hint=lang)
        lat = resp.latency_breakdown

        record = {
            "query_id": idx,
            "lang": lang,
            "query": q[:40] + ("..." if len(q) > 40 else ""),
            "retrieval_ms": lat.retrieval_ms,
            "stt_ms": lat.stt_ms,
            "generation_ms": lat.generation_ms,
            "total_e2e_ms": lat.total_e2e_ms,
            "confidence": resp.confidence,
            "is_refusal": resp.is_refusal
        }
        records.append(record)
        if idx % 10 == 0 or idx == len(test_suite):
            print(f" -> Completed {idx}/{len(test_suite)} queries (Last Retrieval: {lat.retrieval_ms:.1f}ms)", flush=True)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[✓] Results successfully exported to '{output_csv}'", flush=True)

    # Compute Percentiles
    retrieval_times = [r["retrieval_ms"] for r in records]
    gen_times = [r["generation_ms"] for r in records]
    total_times = [r["total_e2e_ms"] for r in records]

    def get_percentiles(arr):
        return {
            "p50": np.percentile(arr, 50),
            "p70": np.percentile(arr, 70),
            "p100": np.percentile(arr, 100)
        }

    ret_pct = get_percentiles(retrieval_times)
    gen_pct = get_percentiles(gen_times)
    tot_pct = get_percentiles(total_times)

    RETRIEVAL_SLA_MS = 200.0
    ret_sla = f"< {RETRIEVAL_SLA_MS:.0f}ms [PASS]" if ret_pct['p50'] < RETRIEVAL_SLA_MS else f"> {RETRIEVAL_SLA_MS:.0f}ms [FAIL]"

    print("\n" + "=" * 70)
    print("               FINAL LATENCY BENCHMARK REPORT (ms)")
    print("=" * 70)
    print(f"{'STAGE':<25} | {'P50 (Median)':<12} | {'P70':<10} | {'P100 (Max)':<10} | {'SLA STATUS':<15}")
    print("-" * 70)
    print(f"{'1. Hybrid Retrieval':<25} | {ret_pct['p50']:>10.2f} ms | {ret_pct['p70']:>8.2f} ms | {ret_pct['p100']:>8.2f} ms | {ret_sla:<15}")
    print(f"{'2. LLM Generation':<25} | {gen_pct['p50']:>10.2f} ms | {gen_pct['p70']:>8.2f} ms | {gen_pct['p100']:>8.2f} ms | {'Network-bound':<15}")
    print(f"{'3. Total End-to-End':<25} | {tot_pct['p50']:>10.2f} ms | {tot_pct['p70']:>8.2f} ms | {tot_pct['p100']:>8.2f} ms | {'Honest Measured':<15}")
    print("=" * 70)
    print(f"\n[Note] Retrieval SLA target: <{RETRIEVAL_SLA_MS:.0f}ms. P50={ret_pct['p50']:.1f}ms.")
    if ret_pct['p50'] >= RETRIEVAL_SLA_MS:
        print(f"[!] SLA MISS: Retrieval P50 ({ret_pct['p50']:.1f}ms) exceeds {RETRIEVAL_SLA_MS:.0f}ms target.")
        print(f"    Root cause: CPU-bound transformer embedding (~{ret_pct['p50']:.0f}ms/query).")
        print(f"    Fix: GPU inference OR smaller/distilled embedding model.")


    # Simple ASCII Bar Chart
    print("\nSTAGE LATENCY PROPORTION (P50 Breakdown):")
    max_w = 40
    ret_w = max(1, int((ret_pct['p50'] / tot_pct['p50']) * max_w))
    gen_w = max(1, int((gen_pct['p50'] / tot_pct['p50']) * max_w))

    print(f"Retrieval ({ret_pct['p50']:.1f}ms) : " + "█" * ret_w)
    print(f"Generation ({gen_pct['p50']:.1f}ms): " + "█" * gen_w)
    print("=" * 70 + "\n", flush=True)

    return records


if __name__ == "__main__":
    run_latency_benchmark()
