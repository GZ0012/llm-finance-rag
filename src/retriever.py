# src/retriever.py

import numpy as np
import faiss
from sklearn.cluster import KMeans


class Retriever:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype("float32")

        dim = self.embeddings.shape[1]
        
        #Still using faiss to localize the embedding
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        self.cluster_weights = self.compute_cluster_weights(self.embeddings)

    def compute_cluster_weights(
        self,
        embeddings,
        max_k=8,
        min_weight=0.85,
        max_weight=1
    ): #should be autoselect. However, since the resource that we used has less noice, we prefer lower range of weight change
        n = len(embeddings)
        
        #check the min requirement
        if n <= 2:
            return np.ones(n)

        max_k = min(max_k, n - 1)
        
        #elbow method to find k
        inertias = []
        k_values = list(range(2, max_k + 1))

        for k in k_values:
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            km.fit(embeddings)
            inertias.append(km.inertia_)

        drops = np.diff(inertias)
        second_drops = np.diff(drops)

        if len(second_drops) > 0:
            best_k = k_values[int(np.argmax(second_drops)) + 1]
        else:
            best_k = 2

        kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_

        distances = np.linalg.norm(embeddings - centers[labels], axis=1)

        ranks = distances.argsort().argsort()
        percentiles = ranks / max(1, n - 1)
        
        #important calculation of wieghts
        weights = max_weight - percentiles * (max_weight - min_weight)

        return weights.astype("float32")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, candidate_k: int = 15):
        query_embedding = query_embedding.astype("float32")

        candidate_k = max(candidate_k, top_k)

        distances, indices = self.index.search(query_embedding, candidate_k)

        reranked = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            sim = -dist
            weight = self.cluster_weights[idx]
            final_score = sim * weight

            reranked.append((idx, final_score))

        reranked.sort(key=lambda x: x[1], reverse=True)

        top = reranked[:top_k]

        final_indices = np.array([[i for i, _ in top]])
        final_scores = np.array([[s for _, s in top]])

        return final_scores, final_indices


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
        