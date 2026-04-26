# src/chunker.py

from typing import List


def split_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


if __name__ == "__main__":
    from loader import load_document

    file_path = "data/Microsoft/PressReleaseFY26Q1.docx"

    text = load_document(file_path)
    chunks = split_text(text, chunk_size=200, overlap=50)

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:700])