from loader import load_document
from chunker import semantic_chunk_by_target
from sentence_transformers import SentenceTransformer

file_path = "data/Microsoft/PressReleaseFY26Q1.docx"

paragraphs = load_document(file_path)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(paragraphs)

chunks = semantic_chunk_by_target(
    paragraphs,
    embeddings,
    target_chunks=20,
    source=file_path
)

print("Paragraphs:", len(paragraphs))
print("Chunks:", len(chunks))

for chunk in chunks[:3]:
    print("=" * 80)
    print("Chunk ID:", chunk["chunk_id"])
    print("Length:", len(chunk["text"]))
    print(chunk["text"][:1000])