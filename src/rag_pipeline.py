# src/rag_pipeline.py

from embedder import Embedder
from llm_client import ask_llm
from document_manager import load_chunked_document
from vector_store import VectorStore


# Keywords that signal the user wants a broad/overview answer rather than
# a specific fact. When matched, the pipeline skips retrieval and uses the
# pre-generated document summary as context instead.
_GLOBAL_KEYWORDS = frozenset({
    # Explicit summary requests
    "summarize", "summary", "overview", "overall",

    # "what (is/does/did/was) this/the document/report/file/release ..."
    "what is this document", "what does this document",
    "what did this document", "what was this document",
    "what is the document", "what does the document",
    "what did the document", "what was the document",
    "what is this report", "what does this report",
    "what did this report", "what was this report",
    "what is this release", "what did this release",
    "about this document", "about the document",
    "about this report", "about the report",
    "this document about", "the document about",

    # "talk/discuss/cover/describe" + document reference
    "talk about", "talking about", "talks about",
    "discuss", "covered in this", "covered in the",

    # State / overview questions
    "what is going on", "what's going on",
    "what happened", "what's in this", "what is in this",
    "what is this about", "what's this about",

    # Topic / theme requests
    "key topics", "main topics", "key themes", "main themes",
    "highlights", "key points", "main points",
    "what is covered", "what are the main", "what are the key",

    # Broad / general framing
    "in general", "broadly", "tell me about this",
    "tell me about the", "describe this", "describe the",
    "explain this document", "explain the document",
    "explain this report", "explain the report",
})


def is_global_query(question: str) -> bool:
    """Return True if the question is asking for a broad document overview."""
    q = question.lower()
    return any(kw in q for kw in _GLOBAL_KEYWORDS)


def ask(
    question: str,
    chunk_path: str,
    top_k: int = 5,
    prompt_type: str = "role"
):
    """
    Ask a question against one selected document.

    Uses the shared VectorStore (persistent FAISS index) for retrieval —
    no per-query index rebuild. The pkl is only loaded for the summary
    (global queries) and the doc_name filter.
    """

    # 1. load the pkl for summary and doc name (no longer needed for embeddings)
    data = load_chunked_document(chunk_path)
    doc_name = data["file_name"]

    # 2. Global path: broad/overview questions use the stored summary directly,
    #    skipping retrieval entirely.
    if is_global_query(question):
        summary = data.get("summary")
        if summary:
            answer = ask_llm(context=summary, question=question, prompt_type="global")
            return answer, []

    # 3. Embed the query once, then search the shared vector store.
    #    Filtering by doc_name scopes results to the selected document.
    embedder = Embedder()
    query_embedding = embedder.encode([question])

    vector_store = VectorStore()
    results = vector_store.search(query_embedding, top_k=top_k, doc_name=doc_name)

    # 4. Build retrieved_chunks in the format the rest of the pipeline expects.
    retrieved_chunks = []
    for rank, r in enumerate(results, start=1):
        text = r["text"].strip().replace("\n", " ")
        retrieved_chunks.append({
            "rank": rank,
            "source_file": r["source_file"],
            "chunk_id": r["chunk_id"],
            "preview": " ".join(text.split()[:8]),
            "text": text,
        })

    # 5. Build context string for the LLM
    context = "\n\n".join(
        f"[Source {item['rank']} | Chunk {item['chunk_id']}]\n{item['text']}"
        for item in retrieved_chunks
    )

    # 6. Ask LLM with the selected prompt type
    answer = ask_llm(context=context, question=question, prompt_type=prompt_type)

    return answer, retrieved_chunks