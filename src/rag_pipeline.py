from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np

file_path = "data/Microsoft/PressReleaseFY26Q1.docx"

doc = Document(file_path)

paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
msft_text = "\n".join(paragraphs)

def split_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


chunks = split_text(msft_text)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedding_model.encode(chunks)

print(f"Embedding shape: {chunk_embeddings.shape}")
print(chunk_embeddings[0][:10])
    
    