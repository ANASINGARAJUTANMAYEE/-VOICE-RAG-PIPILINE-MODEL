#!/usr/bin/env bash
# ==============================================================================
# Single-Command Indexing Pipeline for Voice-Enabled Multilingual RAG
# Sequence: Pull Data -> 3-Strategy Chunking -> Embed & Build FAISS/BM25 Indices
# ==============================================================================

set -e

echo "======================================================================"
echo "Starting End-to-End Dataset Indexing Pipeline (Hindi, Bengali, Tamil)"
echo "======================================================================"

# Step 1: Pull and normalize dataset
echo ""
echo "[1/4] Running Dataset Ingestion (src/ingestion/load_data.py)..."
python src/ingestion/load_data.py

# Step 2: Multi-strategy chunking & fusion
echo ""
echo "[2/4] Running 3-Strategy Chunking Fusion (src/chunking/fusion.py)..."
python src/chunking/fusion.py

# Step 3: Embed and build vector/keyword indices
echo ""
echo "[3/4] Building FAISS Vector Index and BM25 Store (scripts/build_indices.py)..."
python scripts/build_indices.py

# Step 4: Run latency validation benchmark
echo ""
echo "[4/4] Executing Latency Verification Benchmark (src/evaluation/latency_bench.py)..."
python src/evaluation/latency_bench.py --samples 30

echo ""
echo "======================================================================"
echo "[SUCCESS] Indexing Pipeline Completed! Server is ready to launch."
echo "Launch dev server with: uvicorn app.main:app --port 8000 --reload"
echo "======================================================================"
