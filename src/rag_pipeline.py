# src/rag_pipeline.py

from embedder import Embedder
from retriever import Retriever
from llm_client import ask_llm
from document_manager import load_chunked_document


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

    # 2. Global path: if the question is broad/overview, skip retrieval and
    #    answer directly from the pre-generated document summary.
    #    Returns empty sources list since no specific chunks were retrieved.
    if is_global_query(question):
        summary = data.get("summary")
        if summary:
            answer = ask_llm(context=summary, question=question, prompt_type="global")
            return answer, []

    # 3. embed question only once
    embedder = Embedder()
    query_embedding = embedder.encode([question])

    # 4. retrieve from saved chunk embeddings
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