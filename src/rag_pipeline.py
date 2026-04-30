# src/rag_pipeline.py

from embedder import Embedder
from retriever import Retriever
from llm_client import ask_llm
from document_manager import load_chunked_document


def ask(
    question: str,
    chunk_path: str,
    top_k: int = 5,
    prompt_type: str = "role"
):
    """
    Ask a question under one selected chunked document.

    This function does NOT re-chunk or re-embed the document.
    It only:
    1. loads precomputed chunks + embeddings
    2. embeds the user question once
    3. retrieves top_k chunks
    4. sends context to LLM
    """

    # 1. load precomputed chunk file
    data = load_chunked_document(chunk_path)

    chunks = data["chunks"]
    chunk_embeddings = data["embeddings"]
    source_file = data["source_file"]

    # 2. embed question only once
    embedder = Embedder()
    query_embedding = embedder.encode([question])

    # 3. retrieve from saved chunk embeddings
    retriever = Retriever(chunk_embeddings)
    _, indices = retriever.search(query_embedding, top_k)

    # 4. prepare retrieved chunks
    retrieved_chunks = []

    for rank, i in enumerate(indices[0], start=1):
        i = int(i)
        text = chunks[i].strip().replace("\n", " ")
        preview = " ".join(text.split()[:8])

        retrieved_chunks.append({
            "rank": rank,
            "source_file": source_file,
            "chunk_id": i,
            "preview": preview,
            "text": text
        })

    # 5. build context for LLM
    context = "\n\n".join(
        f"[Source {item['rank']} | Chunk {item['chunk_id']}]\n{item['text']}"
        for item in retrieved_chunks
    )

    # 6. ask LLM with selected prompt type
    answer = ask_llm(
        context=context,
        question=question,
        prompt_type=prompt_type
    )

    return answer, retrieved_chunks