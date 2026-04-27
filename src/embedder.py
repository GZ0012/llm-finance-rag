# src/embedder.py

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings.astype("float32")


if __name__ == "__main__":
    from loader import load_document
    from chunker import split_text

    file_path = "data/Microsoft/PressReleaseFY26Q1.docx"

    text = load_document(file_path)
    chunks = split_text(text, chunk_size=200, overlap=50)

    embedder = Embedder()
    embeddings = embedder.encode(chunks)

    print("Embedding completed.")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First vector preview: {embeddings[0][:10]}")