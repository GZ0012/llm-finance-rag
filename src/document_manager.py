# src/document_manager.py

from pathlib import Path
import pickle

from loader import load_document
from chunker import split_text
from embedder import Embedder


DATA_DIR = Path("/Users/garyzhou/github/llm-finance-rag/data")
CHUNK_DIR = Path("/Users/garyzhou/github/llm-finance-rag/chunk")
CHUNK_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}


def get_chunk_path(file_path: Path) -> Path:
    return CHUNK_DIR / f"{file_path.stem}_chunked.pkl"


def update_documents():
    updated = []

    embedder = Embedder()

    for file_path in DATA_DIR.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        chunk_path = get_chunk_path(file_path)

        if chunk_path.exists():
            continue

        print(f"Processing: {file_path.name}")

        text = load_document(str(file_path))
        chunks = split_text(text, chunk_size=100, max_multiplier=2.5) #This is the size setting
        embeddings = embedder.encode(chunks)

        chunk_data = {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "chunks": chunks,
            "embeddings": embeddings,
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