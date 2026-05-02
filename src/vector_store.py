# src/vector_store.py

import json
import numpy as np
import faiss
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = PROJECT_ROOT / "vector_store"

# all-MiniLM-L6-v2 output dimension — must match the Embedder model.
EMBEDDING_DIM = 384


class VectorStore:
    """
    Persistent centralized FAISS index across all documents.

    FAISS stores only vectors; metadata (chunk text, doc name, chunk id)
    is kept in a parallel JSON list where meta[i] describes vector i.
    Both files are written together on every save() call.
    """

    def __init__(self, store_dir: Path = STORE_DIR):
        self.index_path = store_dir / "index.faiss"
        self.meta_path = store_dir / "meta.json"
        store_dir.mkdir(exist_ok=True)

        if self.index_path.exists() and self.meta_path.exists():
            # Load existing index and metadata from disk.
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            # First run — create a fresh flat L2 index.
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.meta = []

    def add(self, embeddings: np.ndarray, metadata: list):
        """
        Add a batch of embeddings with their metadata.

        metadata must be a list of dicts with at least:
          { "doc_name", "source_file", "chunk_id", "text" }
        len(metadata) must equal embeddings.shape[0].
        """
        self.index.add(embeddings.astype("float32"))
        self.meta.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        doc_name: str = None
    ) -> list:
        """
        Search for the top_k nearest chunks.

        If doc_name is given, results are filtered to that document only.
        In that case we retrieve a larger candidate set first, then filter
        down, so we always return exactly top_k results when available.
        """
        if self.index.ntotal == 0:
            return []

        # Fetch more candidates when filtering so we still get top_k after.
        fetch_k = min(top_k * 5 if doc_name else top_k, self.index.ntotal)

        distances, indices = self.index.search(
            query_embedding.astype("float32"), fetch_k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = self.meta[idx]
            if doc_name and entry["doc_name"] != doc_name:
                continue
            results.append({**entry, "score": float(-dist)})
            if len(results) == top_k:
                break

        return results

    def has_document(self, doc_name: str) -> bool:
        """Return True if this document's chunks are already in the store."""
        return any(m["doc_name"] == doc_name for m in self.meta)

    def save(self):
        """Persist the FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
