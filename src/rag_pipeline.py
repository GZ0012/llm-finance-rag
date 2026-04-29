# src/rag_pipeline.py

from loader import load_document
from chunker import split_text
from embedder import Embedder
from retriever import Retriever
from llm_client import ask_llm


def ask(question: str):
    file_path = "data/Microsoft/PressReleaseFY26Q1.docx"

    # 1. load
    text = load_document(file_path)

    # 2. chunk
    chunks = split_text(text, chunk_size=200, max_multiplier=3)

    # 3. embed
    embedder = Embedder()
    chunk_embeddings = embedder.encode(chunks)

    # 4. retrieve
    retriever = Retriever(chunk_embeddings)
    query_embedding = embedder.encode([question])
    _, indices = retriever.search(query_embedding, top_k=10)

    # 5. build context
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    # 6. LLM
    answer = ask_llm(context, question)

    return answer


if __name__ == "__main__":
    question = "what was the revenue of microsoft this quater"
    print(ask(question))