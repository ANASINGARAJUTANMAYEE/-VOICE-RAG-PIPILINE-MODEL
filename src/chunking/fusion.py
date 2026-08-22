"""
Multi-Strategy Chunk Fusion Module
Takes data/data_subset.jsonl and applies Fixed, Semantic, and Metadata-aware chunking.
Produces a unified dataset data/chunks_multistrategy.jsonl tagged by strategy.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from src.chunking.fixed_chunker import chunk_fixed
from src.chunking.semantic_chunker import chunk_semantic
from src.chunking.metadata_chunker import chunk_metadata_aware

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_chunking_fusion(
    input_file: Path = Path("data/data_subset.jsonl"),
    output_file: Path = Path("data/chunks_multistrategy.jsonl")
) -> List[Dict[str, Any]]:
    print("=" * 65)
    print("STEP 2: Running 3-Strategy Chunking & Fusion")
    print(f"Reading from: {input_file}")
    print("=" * 65, flush=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file '{input_file}' not found. Please run STEP 1 ingestion first.")

    raw_passages = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_passages.append(json.loads(line))

    print(f"[+] Loaded {len(raw_passages)} raw source passages from {input_file}", flush=True)

    all_fused_chunks = []
    strategy_counts = {"fixed": 0, "semantic": 0, "metadata": 0}
    lang_counts = {"hi": 0, "bn": 0, "ta": 0}

    for item in raw_passages:
        # Check if record contains nested MS MARCO passages dict
        if "passages" in item and isinstance(item["passages"], dict):
            passages = item["passages"]
            trans_passages = passages.get("Translated_passages", [])
            eng_passages = passages.get("English_passages", [])
            is_selected_list = passages.get("is_selected", [])

            for idx, (text, en_text, sel) in enumerate(zip(trans_passages, eng_passages, is_selected_list)):
                if not text or not str(text).strip():
                    continue
                
                passage_meta = {
                    "chunk_id": f"{item.get('lang', 'unk')}_{item.get('query_id', 0)}_{idx}",
                    "query_id": item.get("query_id"),
                    "lang": item.get("lang"),
                    "query": item.get("query"),
                    "answer": item.get("Answer"),
                    "eng_query": item.get("Eng_Query"),
                    "eng_answer": item.get("Eng_Answer"),
                    "eng_text": en_text,
                    "is_selected": int(sel),
                }

                # 1. Fixed Chunking
                fixed_chunks = chunk_fixed(text, passage_meta)
                for fc in fixed_chunks:
                    all_fused_chunks.append(fc)
                    strategy_counts["fixed"] += 1
                    lang_counts[fc["lang"]] = lang_counts.get(fc["lang"], 0) + 1

                # 2. Semantic Chunking
                semantic_chunks = chunk_semantic(text, passage_meta)
                for sc in semantic_chunks:
                    all_fused_chunks.append(sc)
                    strategy_counts["semantic"] += 1
                    lang_counts[sc["lang"]] = lang_counts.get(sc["lang"], 0) + 1

                # 3. Metadata Chunking
                meta_chunks = chunk_metadata_aware(text, passage_meta)
                for mc in meta_chunks:
                    all_fused_chunks.append(mc)
                    strategy_counts["metadata"] += 1
                    lang_counts[mc["lang"]] = lang_counts.get(mc["lang"], 0) + 1
        else:
            text = item.get("text", "")
            if not text.strip():
                continue

            # 1. Fixed Chunking
            fixed_chunks = chunk_fixed(text, item)
            for fc in fixed_chunks:
                all_fused_chunks.append(fc)
                strategy_counts["fixed"] += 1
                lang_counts[fc["lang"]] = lang_counts.get(fc["lang"], 0) + 1

            # 2. Semantic Chunking
            semantic_chunks = chunk_semantic(text, item)
            for sc in semantic_chunks:
                all_fused_chunks.append(sc)
                strategy_counts["semantic"] += 1
                lang_counts[sc["lang"]] = lang_counts.get(sc["lang"], 0) + 1

            # 3. Metadata Chunking
            meta_chunks = chunk_metadata_aware(text, item)
            for mc in meta_chunks:
                all_fused_chunks.append(mc)
                strategy_counts["metadata"] += 1
                lang_counts[mc["lang"]] = lang_counts.get(mc["lang"], 0) + 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chk in all_fused_chunks:
            f.write(json.dumps(chk, ensure_ascii=False) + "\n")

    print(f"\n[✓] Chunk Fusion Complete! Generated {len(all_fused_chunks)} multi-strategy chunks.")
    print(f" -> Output saved to '{output_file}'", flush=True)

    print("\nStrategy Breakdown:")
    for strat, count in strategy_counts.items():
        print(f" - {strat.capitalize()} Chunker: {count} chunks")

    print("\nLanguage Breakdown:")
    for lang, count in lang_counts.items():
        print(f" - {lang.upper()}: {count} chunks")

    # Sample preview of one chunk from each strategy
    print("\nSample Preview (1 from each strategy):")
    for strat in ["fixed", "semantic", "metadata"]:
        sample = next((c for c in all_fused_chunks if c["chunk_strategy"] == strat), None)
        if sample:
            preview = {
                "chunk_id": sample["chunk_id"],
                "strategy": sample["chunk_strategy"],
                "lang": sample["lang"],
                "tokens": sample["tokens_count"],
                "text_snippet": sample["text"][:80] + "..."
            }
            print(f"[{strat.upper()}]:", json.dumps(preview, ensure_ascii=False))

    return all_fused_chunks


if __name__ == "__main__":
    run_chunking_fusion()
