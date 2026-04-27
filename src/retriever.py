# src/retriever.py

import numpy as np
import faiss


class Retriever:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype("float32")
        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        query_embedding = query_embedding.astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices


if __name__ == "__main__":
    from loader import load_document
    from chunker import split_text
    from embedder import Embedder

    file_path = "data/Microsoft/PressReleaseFY26Q1.docx"

    text = load_document(file_path)
    chunks = split_text(text, chunk_size=200, overlap=50)

    embedder = Embedder()
    chunk_embeddings = embedder.encode(chunks)

    retriever = Retriever(chunk_embeddings)

    query = "What was Microsoft's revenue?"
    query_embedding = embedder.encode([query])

    distances, indices = retriever.search(query_embedding, top_k=3)

    print("\nQuery:", query)
    print("\nTop retrieved chunks:\n")

    for rank, idx in enumerate(indices[0], start=1):
        print(f"--- Result {rank} ---")
        print(f"Chunk index: {idx}")
        print(f"Distance: {distances[0][rank-1]:.4f}")
        print(chunks[idx][:700])
        print()
        