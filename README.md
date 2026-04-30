# LLM Finance RAG

A Retrieval-Augmented Generation (RAG) pipeline for context-aware question answering over financial documents (e.g., earnings reports, press releases).

---

## Overview

This project implements an end-to-end RAG system that enables users to ask natural language questions about financial documents and receive grounded, context-aware answers.

Instead of sending entire documents to an LLM, the system retrieves only the most relevant sections, significantly improving efficiency and reducing hallucinations.

---

## Architecture

→ Loader: extracts paragraphs and tables
→ Chunker: semantic chunking with adaptive size control
→ Embedding Model: converts chunks into vector representations
→ FAISS Vector Index: stores searchable chunk embeddings
→ Retriever: retrieves top-k chunks and applies density-based reranking
→ LLM: generates grounded answers from retrieved context
→ Final Answer

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

## chunker update

Problem:
Splitting text purely by character count is pretty unreliable. It can break important context across chunks, which leads to weaker retrieval and sometimes inaccurate answers—even if FAISS finds the “right” place.

Solution:
Instead of splitting by size, I first embed each paragraph and look at the cosine similarity between adjacent paragraphs.

To split a document into k chunks, I find the k-1 points where similarity drops the most and split there.

This way, chunks are separated based on actual semantic boundaries, not arbitrary length, which helps keep each chunk more self-contained and contextually consistent.


## outlier update
Problem:
When using semantic chunking, noise (e.g. something like “I am noise” inside a financial report) often gets isolated into its own chunk. While that’s actually correct behavior, it can still mess with retrieval if treated equally.

Solution:
I keep the similarity-based chunking (since it already isolates noise well), and instead handle noise during retrieval.

To avoid removing useful but rare information, I don’t filter anything out. Instead, I:

- Use KMeans clustering to group semantically similar chunks
- Use an elbow method to estimate a reasonable number of clusters
- Measure how far each chunk is from its cluster center
- Convert that into a soft weight using percentile ranking
- weight = max_weight - percentile * (max_weight - min_weight)

Chunks that are far from their cluster center (potential outliers) get slightly downweighted, while more “typical” chunks are preserved.

Since financial reports are usually pretty clean, I keep the weight range tight to avoid overcorrecting.


#### 1. Document-Level Chunk Management
- Documents placed in `data/` are automatically processed using the `/update document` command.
- Each document is independently chunked and stored as: chunk/<document_name>_chunked.pkl
- This design ensures that each document can be queried independently, avoiding cross-document interference.

---

#### 2. Embedding Caching (Performance Optimization)
- Chunk embeddings are computed **only once** during the update phase.
- Stored together with chunk metadata in `.pkl` files.
- During query time: Only the user question is embedded
- This significantly reduces latency and makes the system scalable.

---

#### 3. Interactive TUI (Text-Based Interface)

The system includes a lightweight terminal-based UI:

```bash
python src/chatui.py
```
Supported commands:
- /update document   # Process new files in data/
- /docs              # List available documents
- /use 1             # Select a document
- /k 5               # Set retrieval depth (top-k)
- /record            # Show session record file

#### 4. Source Attribution (Explainability)

Each answer includes retrieved chunk references:

[1] Revenue was $77.7 billion...
[2] Microsoft Cloud revenue...
Improves transparency
Allows users to trace model outputs back to source text

#### 5. Session Recording (Traceability)

Each chat session is saved as a structured Markdown file:

records/rag_session_YYYYMMDD_HHMMSS.md

Each entry contains:

- Question
- Selected Document
- Answer
- Retrieved Chunks (Top-k)

This enables:

Reproducibility
Debugging
Experiment tracking

##  Recent Update: Prompt System + Persistent RAG

This update introduces a more modular and production-style RAG architecture with prompt flexibility and improved system design.

### 🧠 Prompt System (CoT / Role / Few-shot)

- Prompts are no longer hardcoded in Python
- Moved to `src/prompts/*.md` for better modularity
- Supported prompt types:

```text
role     → standard role-based QA
cot      → chain-of-thought style (hidden reasoning)
fewshot  → example-driven prompting
```

Switch prompts dynamically in TUI:
/prompt role
/prompt cot
/prompt fewshot

---
