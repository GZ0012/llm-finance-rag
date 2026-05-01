# src/document_manager.py

from pathlib import Path
import pickle

from loader import load_document
from chunker import split_text, auto_chunk_size, estimate_tokens
from embedder import Embedder
from llm_client import generate_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHUNK_DIR = PROJECT_ROOT / "chunk"
CHUNK_DIR.mkdir(exist_ok=True)


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

# Human-readable label for each supported extension, shown at processing time.
EXTENSION_LABELS = {
    ".pdf":  "PDF",
    ".docx": "Word Document",
    ".txt":  "Plain Text",
    ".md":   "Markdown",
}


def get_chunk_path(file_path: Path) -> Path:
    return CHUNK_DIR / f"{file_path.stem}_chunked.pkl"


def update_documents():
    updated = []

    embedder = Embedder()

    for file_path in DATA_DIR.rglob("*"):
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        chunk_path = get_chunk_path(file_path)

        if chunk_path.exists():
            continue

        # Auto-detect file type and log it before any heavy work begins,
        # so you can confirm the right loader will be used.
        doc_type = EXTENSION_LABELS.get(suffix, suffix)
        print(f"Detected [{doc_type}]: {file_path.name}")

        text = load_document(str(file_path))

        # Estimate total document tokens before chunking so auto_chunk_size
        # can pick an appropriate chunk size for this specific document.
        paragraphs = text if isinstance(text, list) else text.split("\n")
        total_tokens = sum(estimate_tokens(p) for p in paragraphs if p.strip())

        # Auto-select chunk size based on document length (sqrt heuristic).
        # Stored in chunk_data so you can inspect it later with /docs.
        chunk_size = auto_chunk_size(total_tokens)
        print(f"  total_tokens={total_tokens}  →  auto chunk_size={chunk_size}")

        chunks = split_text(text, chunk_size=chunk_size, max_multiplier=2.5)
        embeddings = embedder.encode(chunks)

        # Generate a full-document summary once at index time so global
        # queries ("summarize this", "what is this document about") can be
        # answered without retrieval — just the stored summary as context.
        print(f"  Generating document summary...")
        summary = generate_summary(chunks)

        chunk_data = {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "doc_type": doc_type,
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_size": chunk_size,   # saved so you can audit the decision later
            "total_tokens": total_tokens,
            "summary": summary,
        }

        with open(chunk_path, "wb") as f:
            pickle.dump(chunk_data, f)

        updated.append(file_path.name)

    return updated


def list_chunked_documents():
    docs = []

    for chunk_file in CHUNK_DIR.glob("*_chunked.pkl"):
        with open(chunk_file, "rb") as f:
            data = pickle.load(f)

        docs.append({
            "name": data["file_name"],
            "chunk_path": str(chunk_file),
            "source_file": data["source_file"],
        })

    return docs


def load_chunked_document(chunk_path: str):
    with open(chunk_path, "rb") as f:
        data = pickle.load(f)

    return data