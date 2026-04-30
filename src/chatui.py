# src/chatui.py

from pathlib import Path
from datetime import datetime

from rag_pipeline import ask
from document_manager import update_documents, list_chunked_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECORD_DIR = PROJECT_ROOT / "records"
RECORD_DIR.mkdir(exist_ok=True)


def create_record_file():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_path = RECORD_DIR / f"rag_session_{ts}.md"

    with open(record_path, "w", encoding="utf-8") as f:
        f.write("# RAG Chat Session\n\n")
        f.write(f"Started: {datetime.now()}\n\n")

    return record_path


def append_record(record_path, question, answer, sources, current_doc):
    source_text = "\n".join(
        f"- [{s['rank']}] Chunk {s['chunk_id']}: {s['preview']}"
        for s in sources
    )

    block = f"""
---

## Question
{question}

## Document
{current_doc["name"]}

## Answer
{answer}

## Retrieved Chunks
{source_text}

"""

    with open(record_path, "a", encoding="utf-8") as f:
        f.write(block)


def close_record(record_path):
    with open(record_path, "a", encoding="utf-8") as f:
        f.write(f"\n---\n\nEnded: {datetime.now()}\n")


def print_help():
    print("""
Commands:
  /update document     Chunk + embed new documents
  /docs                Show available documents
  /use 1               Select document
  /k 5                 Set top_k default 5
  /prompt role|cot|fewshot   Set prompt type
  /record              Show current record file
  /help                Show commands
  exit                 Quit
""")


def format_sources(sources):
    return " | ".join(
        f"[{s['rank']}] {s['preview']}..."
        for s in sources
    )


def main():
    record_path = create_record_file()

    print("=" * 60)
    print("LLM Finance RAG Chat")
    print("=" * 60)
    print(f"Session record: {record_path}")
    print_help()

    top_k = 5
    current_doc = None
    prompt_type = "role"

    while True:
        user_input = input("\n> ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            close_record(record_path)
            print(f"Session saved to: {record_path}")
            print("Goodbye.")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/record":
            print(f"Current record file: {record_path.resolve()}")
            continue

        if user_input == "/update document":
            updated = update_documents()

            if updated:
                print("\nUpdated documents:")
                for name in updated:
                    print(f"- {name}")
            else:
                print("\nNo new documents found.")
            continue

        if user_input == "/docs":
            docs = list_chunked_documents()

            if not docs:
                print("No chunked documents found. Run /update document first.")
                continue

            print("\nAvailable documents:")
            for idx, doc in enumerate(docs, start=1):
                print(f"{idx}. {doc['name']}")
            continue

        if user_input.startswith("/use"):
            docs = list_chunked_documents()

            try:
                doc_idx = int(user_input.split()[1]) - 1
                current_doc = docs[doc_idx]
                print(f"Current document: {current_doc['name']}")
            except Exception:
                print("Usage: /use 1")
            continue

        if user_input.startswith("/k"):
            try:
                top_k = int(user_input.split()[1])
                print(f"top_k = {top_k}")
            except Exception:
                print("Usage: /k 5")
            continue

        if user_input.startswith("/prompt"):
            try:
                prompt_type = user_input.split()[1]

                if prompt_type not in ["role", "cot", "fewshot"]:
                    print("Usage: /prompt role | cot | fewshot")
                    continue

                print(f"prompt_type = {prompt_type}")

            except Exception:
                print("Usage: /prompt role | cot | fewshot")
            continue

        if not user_input:
            continue

        if current_doc is None:
            print("Please select a document first: /docs then /use 1")
            continue

        try:
            answer, sources = ask(
                question=user_input,
                chunk_path=current_doc["chunk_path"],
                top_k=top_k,
                prompt_type=prompt_type
            )

            print("\nAnswer:")
            print(answer)

            print("\nSources:")
            print(format_sources(sources))

            append_record(
                record_path=record_path,
                question=user_input,
                answer=answer,
                sources=sources,
                current_doc=current_doc
            )

        except Exception as e:
            print("\nError:")
            print(e)


if __name__ == "__main__":
    main()