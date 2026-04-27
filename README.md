# LLM Finance RAG

A Retrieval-Augmented Generation (RAG) pipeline for context-aware question answering over financial documents (e.g., earnings reports, press releases).

---

## Overview

This project implements an end-to-end RAG system that enables users to ask natural language questions about financial documents and receive grounded, context-aware answers.

Instead of sending entire documents to an LLM, the system retrieves only the most relevant sections, significantly improving efficiency and reducing hallucinations.

---

## Architecture

Document (DOCX/PDF) to
Loader (extract text) to
Chunker (split into segments) to
Embedding Model (Sentence Transformers) to
FAISS Vector Index to
Retriever (Top-K similar chunks) to
LLM (GPT) to
Final Answer

---

##  Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-finance-rag.git
cd llm-finance-rag
```

### 2. Install dependencies

pip install -r requirements.txt

### 3. Configure environment variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini

Usage

Run the pipeline:

python src/rag_pipeline.py

Example query:

question = "What was Microsoft's total revenue for the quarter?"

Example output:

Microsoft reported revenue of $77.7 billion for the quarter.


## Future Improvements

- Precompute and persist FAISS index
- Add PDF support
- Add source citations (which chunk was used)
- Streaming responses
- Simple UI (Streamlit or web app)

## Tech Stack

- Python
- SentenceTransformers
- FAISS
- OpenAI API
- NumPy

## Chunker Update
Implemented adaptive semantic chunking that derives target chunk counts from desired token size and enforces upper bounds via recursive splitting at weakest semantic boundaries.