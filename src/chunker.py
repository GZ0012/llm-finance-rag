# src/chunker.py

import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embedder import Embedder


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation.
    1 token is approximately 4 English characters.
    """
    return max(1, len(text) // 4)


def auto_chunk_size(
    total_tokens: int,
    top_k: int = 5,
    max_context_tokens: int = 3000
) -> int:
    """
    Auto-detect chunk size based on document length.

    Uses the square-root law from information retrieval:
      chunk_size ≈ sqrt(total_tokens)

    This produces roughly sqrt(N) chunks from N tokens, which
    balances two competing pressures:
      - Too small → chunks lose coherence (a metric split from its label)
      - Too large → retrieval is imprecise (too much noise per chunk)

    A second ceiling ensures that top_k retrieved chunks
    always fit within the LLM's usable context window.
    """

    # Square-root heuristic: targets ~sqrt(total_tokens) total chunks.
    # Derivation: if chunk_size = sqrt(N), then N / chunk_size = sqrt(N) chunks.
    natural_size = int(math.sqrt(total_tokens))

    # LLM context ceiling: prevent top_k * chunk_size from overflowing the prompt.
    # e.g. top_k=5, max_context_tokens=3000 → each chunk capped at 600 tokens.
    context_ceiling = max_context_tokens // max(1, top_k)

    # Hard lower bound: financial sentences need ~80 tokens to stay coherent
    # (a single earnings metric with its label, units, and YoY comparison).
    MIN_CHUNK = 80

    # Hard upper bound: above ~512 tokens, retrieved chunks carry too much
    # off-topic content, which degrades LLM answer quality.
    MAX_CHUNK = 512

    size = min(natural_size, context_ceiling, MAX_CHUNK)
    size = max(size, MIN_CHUNK)

    return size


def normalize_input(text):
    """
    Accept either:
    - list[str] paragraphs from loader.py
    - raw string text

    Return:
    - list[str] paragraphs
    """
    if isinstance(text, str):
        return [p.strip() for p in text.split("\n") if p.strip()]

    return [p.strip() for p in text if p and p.strip()]


def get_adjacent_similarities(embeddings):
    """
    Compute cosine similarity between adjacent paragraph embeddings.

    similarities[i] = similarity between paragraph i and paragraph i+1
    """
    sims = []

    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        sims.append(sim)

    return np.array(sims)


def get_split_indices_by_target_chunks(similarities, target_chunks):
    """
    Select the lowest similarity positions as split points.

    If target_chunks = 20, we need 19 split points.
    """
    target_splits = target_chunks - 1

    if target_splits <= 0:
        return set()

    target_splits = min(target_splits, len(similarities))

    split_indices = np.argsort(similarities)[:target_splits]

    return set(split_indices)


def build_chunks_from_split_indices(paragraphs, split_indices, source="unknown"):
    """
    Build chunk dictionaries based on split indices.

    If i is in split_indices, split after paragraph i.
    """
    chunks = []
    current = []
    chunk_id = 0

    for i, para in enumerate(paragraphs):
        current.append(para)

        if i in split_indices:
            chunk_text = "\n\n".join(current)

            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "text": chunk_text,
                "paragraph_start": i - len(current) + 1,
                "paragraph_end": i,
                "token_count": estimate_tokens(chunk_text)
            })

            chunk_id += 1
            current = []

    if current:
        end_i = len(paragraphs) - 1
        chunk_text = "\n\n".join(current)

        chunks.append({
            "chunk_id": chunk_id,
            "source": source,
            "text": chunk_text,
            "paragraph_start": end_i - len(current) + 1,
            "paragraph_end": end_i,
            "token_count": estimate_tokens(chunk_text)
        })

    return chunks


def split_oversized_chunks(
    paragraphs,
    embeddings,
    split_indices,
    target_chunks,
    max_multiplier=2.5
):
    """
    If any chunk is too large, split it again at the weakest
    local semantic boundary.

    max_chunk_tokens = target average chunk size * max_multiplier
    """
    total_tokens = sum(estimate_tokens(p) for p in paragraphs)

    avg_chunk_tokens = max(1, total_tokens / target_chunks)
    max_chunk_tokens = int(avg_chunk_tokens * max_multiplier)

    changed = True

    while changed:
        changed = False

        chunks = build_chunks_from_split_indices(
            paragraphs=paragraphs,
            split_indices=split_indices
        )

        for chunk in chunks:
            if chunk["token_count"] <= max_chunk_tokens:
                continue

            start = chunk["paragraph_start"]
            end = chunk["paragraph_end"]

            if start >= end:
                continue

            local_sims = []

            for i in range(start, end):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                local_sims.append((i, sim))

            # Split at the lowest local similarity point
            best_split_index, _ = min(local_sims, key=lambda x: x[1])

            if best_split_index not in split_indices:
                split_indices.add(best_split_index)
                changed = True
                break

    return split_indices, max_chunk_tokens


def semantic_chunk_by_size(
    paragraphs,
    embeddings,
    chunk_size=400,
    source="unknown",
    max_multiplier=2.5
):
    """
    Semantic chunking controlled by desired average chunk size.

    Steps:
    1. Estimate total document tokens.
    2. Convert chunk_size into target_chunks.
    3. Compute adjacent paragraph similarities.
    4. Split at the weakest semantic boundaries.
    5. Further split oversized chunks.
    """
    if not paragraphs:
        return [], 0

    if len(paragraphs) == 1:
        chunk = {
            "chunk_id": 0,
            "source": source,
            "text": paragraphs[0],
            "paragraph_start": 0,
            "paragraph_end": 0,
            "token_count": estimate_tokens(paragraphs[0])
        }
        return [chunk], estimate_tokens(paragraphs[0])

    total_tokens = sum(estimate_tokens(p) for p in paragraphs)

    target_chunks = max(1, int(round(total_tokens / chunk_size)))

    similarities = get_adjacent_similarities(embeddings)

    split_indices = get_split_indices_by_target_chunks(
        similarities=similarities,
        target_chunks=target_chunks
    )

    split_indices, max_chunk_tokens = split_oversized_chunks(
        paragraphs=paragraphs,
        embeddings=embeddings,
        split_indices=split_indices,
        target_chunks=target_chunks,
        max_multiplier=max_multiplier
    )

    chunks = build_chunks_from_split_indices(
        paragraphs=paragraphs,
        split_indices=split_indices,
        source=source
    )

    return chunks, max_chunk_tokens


def split_text(
    text,
    chunk_size=400,
    max_multiplier=2.5,
    source="unknown"
):
    """
    Main public function used by rag_pipeline.py.

    Args:
        text: list[str] paragraphs from loader.py, or raw string text
        chunk_size: target average chunk size in estimated tokens
        max_multiplier: maximum chunk size = average chunk size * max_multiplier
        source: source document name/path

    Returns:
        list[str]: plain chunk texts
    """
    paragraphs = normalize_input(text)

    if not paragraphs:
        return []

    embedder = Embedder()
    paragraph_embeddings = embedder.encode(paragraphs)

    chunks, _ = semantic_chunk_by_size(
        paragraphs=paragraphs,
        embeddings=paragraph_embeddings,
        chunk_size=chunk_size,
        source=source,
        max_multiplier=max_multiplier
    )

    return [chunk["text"] for chunk in chunks]


if __name__ == "__main__":
    from loader import load_document

    file_path = "data/Microsoft/PressReleaseFY26Q1.docx"

    paragraphs = load_document(file_path)

    chunks = split_text(
        text=paragraphs,
        chunk_size=400,
        max_multiplier=2.5,
        source=file_path
    )

    print("Paragraphs:", len(paragraphs))
    print("Chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print("Chunk ID:", i)
        print("Length:", len(chunk))
        print(chunk[:1000])