"""
Build Vector & Keyword Indices
Reads data/chunks_multistrategy.jsonl, computes embeddings with MultilingualEmbedder,
and builds FAISS IndexFlatIP + BM25Okapi indices.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.embed import get_embedder
from src.retrieval.vector_store import FaissVectorStore
from src.retrieval.bm25_store import BM25Store

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_indices(
    chunks_file: Path = Path("data/chunks_multistrategy.jsonl"),
    faiss_index_path: Path = Path("data/faiss_index.bin"),
    faiss_meta_path: Path = Path("data/faiss_metadata.json"),
    bm25_corpus_path: Path = Path("data/bm25_corpus.json"),
    batch_size: int = 128
):
    print("=" * 65)
    print("STEP 3: Building Dense (FAISS) and Sparse (BM25) Indices")
    print(f"Source chunks: {chunks_file}")
    print("=" * 65, flush=True)

    if not chunks_file.exists():
        raise FileNotFoundError(f"'{chunks_file}' not found. Run chunking fusion first.")

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"[+] Loaded {len(chunks)} chunks for indexing.", flush=True)

    # 1. Build BM25 Index
    print("\n[+] Building BM25 keyword index...", flush=True)
    t0_bm25 = time.perf_counter()
    bm25_store = BM25Store()
    bm25_store.build_index(chunks)
    bm25_store.save(bm25_corpus_path)
    bm25_duration = round(time.perf_counter() - t0_bm25, 2)
    print(f"[✓] BM25 Index successfully saved in {bm25_duration}s", flush=True)

    # 2. Build FAISS Dense Vector Index
    print("\n[+] Initializing multilingual embedding model...", flush=True)
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]

    print(f"[+] Computing embeddings for {len(texts)} chunks (batch_size={batch_size})...", flush=True)
    t0_embed = time.perf_counter()
    embeddings = embedder.embed_texts(texts, batch_size=batch_size, show_progress=True)
    embed_duration = round(time.perf_counter() - t0_embed, 2)
    print(f"[✓] Embeddings generated in {embed_duration}s. Shape: {embeddings.shape}", flush=True)

    print("\n[+] Adding vectors to FAISS IndexFlatIP...", flush=True)
    vector_store = FaissVectorStore(dimension=embedder.dimension)
    vector_store.add_vectors(embeddings, chunks)
    vector_store.save(faiss_index_path, faiss_meta_path)

    print("\n" + "=" * 65)
    print(f"[✓] All indices successfully created and persisted!")
    print(f" -> FAISS: {vector_store.total_vectors} vectors ({faiss_index_path})")
    print(f" -> BM25:  {bm25_store.total_docs} documents ({bm25_corpus_path})")
    print("=" * 65, flush=True)


if __name__ == "__main__":
    build_indices()
