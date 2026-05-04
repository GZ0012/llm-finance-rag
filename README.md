# LLM Finance RAG

A RAG pipeline for natural language Q&A over financial documents (earnings reports, press releases, 10-Ks).

---

## Architecture

```
Loader → Chunker → Embedder → VectorStore (persistent FAISS) → LLM → Answer
```

- **Loader** — extracts paragraphs and tables from `.pdf`, `.docx`, `.txt`, `.md`
- **Chunker** — semantic chunking at cosine-similarity boundaries with auto chunk sizing
- **VectorStore** — shared persistent FAISS index across all documents; searched directly at query time, no per-query rebuild
- **LLM** — OpenAI model with modular prompt templates (role / CoT / few-shot / global)

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-finance-rag.git
cd llm-finance-rag
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
```

Run the TUI:

```bash
python src/chatui.py
```

TUI commands:

```
/update document   Scan data/ for new files and chunk them
/docs              List available documents
/use 1             Select a document
/k 5               Set retrieval depth (top-k)
/prompt role|cot|fewshot   Switch prompt type
/record            Show session record file path
exit               Quit
```

---

## Tech Stack

- Python, NumPy
- SentenceTransformers (`all-MiniLM-L6-v2`)
- FAISS, scikit-learn (KMeans)
- OpenAI API
- pdfplumber, python-docx

---

## Design Notes

**Semantic chunking** — splits at the lowest cosine-similarity boundaries between adjacent paragraphs rather than by character count, keeping each chunk semantically self-contained.

**Outlier-aware retrieval** — uses KMeans clustering on chunk embeddings to softly downweight chunks that are far from their cluster center (`weight = max - percentile × range`), reducing noise without discarding rare but valid content.

**Persistent vector store** — embeddings are computed once at index time and written to `vector_store/index.faiss` + `vector_store/meta.json`. At query time only the query is embedded; the FAISS index is loaded from disk and searched directly — no per-query rebuild.

**Session recording** — each chat session is saved to `records/rag_session_YYYYMMDD_HHMMSS.md` with question, answer, document, and retrieved chunks for reproducibility and debugging.

---

## Prompt System

Prompts live in `src/prompts/*.md` and can be switched at runtime:

```
/prompt role      standard financial analyst role
/prompt cot       chain-of-thought reasoning
/prompt fewshot   few-shot example-driven
```

Global/overview questions are routed automatically to a separate `global.md` prompt — no manual switching needed.

---

## Recent Update: Auto Chunk Sizing, PDF Support & Global Query Routing

### 1. Auto Chunk Size Detection

Previously, chunk size was hardcoded at 70 tokens regardless of document length. Short documents got over-split; large documents were under-split.

**Solution:** `auto_chunk_size()` in `chunker.py` uses the square-root law from information retrieval:

```
chunk_size ≈ sqrt(total_tokens)
```

This produces roughly `sqrt(N)` chunks from `N` tokens, balancing two competing pressures:
- Too small → financial sentences break mid-context (a metric split from its value)
- Too large → retrieval becomes imprecise (too much off-topic content per chunk)

A second ceiling ensures `top_k × chunk_size` never overflows the LLM's context window.

| Document size | Auto chunk size | Approx. chunks |
|---|---|---|
| 4K tokens (press release) | 80 | ~50 |
| 20K tokens (transcript) | 141 | ~141 |
| 100K tokens (annual report) | 316 | ~316 |
| 200K tokens (10-K) | 447 | ~447 |

Hard bounds: `min=80`, `max=512`. The computed `chunk_size` and `total_tokens` are stored in the `.pkl` file for auditing.

---

### 2. PDF Support

The loader now supports `.pdf`, `.txt`, and `.md` in addition to `.docx`. PDF loading uses **pdfplumber**, which gives superior table extraction compared to pypdf or pdfminer — important for financial documents where tables carry most of the key numbers.

The loader:
1. Detects table bounding boxes on each page and crops them out before extracting prose, preventing duplicate/garbled content
2. Extracts tables separately as pipe-separated rows (`Revenue | $65.6B | +16% YoY`), consistent with the existing docx table format

Place any supported file in `data/` and run `/update document` — no other changes needed.

---

### 3. Auto Document Type Detection

`update_documents()` now identifies and logs the file type before processing:

```
Detected [PDF]: annual_report.pdf
  total_tokens=48231  →  auto chunk_size=219
  Generating document summary...
Detected [Word Document]: PressReleaseFY26Q1.docx
  total_tokens=3841  →  auto chunk_size=80
```

The detected type is stored in the `.pkl` file alongside chunk metadata.

---

### 4. Global vs. Local Query Routing

**Problem:** Retrieval-based QA only pulls `top_k` semantically matched chunks. Broad questions like "what is this document about?" get arbitrary chunks and the LLM can't synthesize a full picture.

**Solution:** Two-path routing based on question type.

**At index time:** A full document summary is generated by the LLM and stored in the `.pkl` file (one API call per document, done once).

**At query time:** `is_global_query()` classifies the question:
- **Global** (broad/overview) → skip retrieval, use stored summary as context → `global.md` prompt
- **Local** (specific fact) → existing retrieval path → user-selected prompt type

Global triggers include: `"summarize"`, `"what did this document talk about"`, `"overview"`, `"key topics"`, `"what happened"`, `"describe the report"`, and 30+ other common phrasings.

```
User: "what did this document talk about"   → Global path (summary context)
User: "what was Microsoft's revenue?"       → Local path  (retrieval)
User: "what are the key themes?"            → Global path (summary context)
User: "how much did cloud revenue grow?"    → Local path  (retrieval)
```

No user action required — routing is fully automatic.

---

## Recent Update: Persistent Vector Store

### Problem

Previously the FAISS index was rebuilt from scratch on every query — loading all embeddings from the `.pkl` file, running KMeans clustering, then searching. This meant latency scaled with document size and made cross-document search impossible.

### Solution

A shared `VectorStore` class backed by a persistent FAISS flat index:

```
vector_store/
  index.faiss   ← all chunk embeddings across all documents
  meta.json     ← parallel metadata: chunk text, doc name, chunk id, source path
```

**At index time** (`/update document`): embeddings are added to the store and saved to disk once.

**At query time**: the store is loaded and searched in one step — no rebuild, no KMeans.

```python
vector_store = VectorStore()
results = vector_store.search(query_embedding, top_k=5, doc_name="report.pdf")
```

Passing `doc_name` filters results to the selected document. Omitting it searches across all documents (foundation for future multi-document Q&A).

### Before vs After

| | Before | After |
|---|---|---|
| Query latency | Rebuild FAISS + KMeans every call | Load index → search |
| Cross-document | Not supported | `search()` without `doc_name` |
| Storage | Embeddings duplicated in each `.pkl` | Single shared `index.faiss` |

---
