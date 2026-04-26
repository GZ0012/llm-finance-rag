# src/loader.py

from pathlib import Path
from docx import Document


def load_docx(file_path: str) -> str:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".docx":
        raise ValueError(f"Only .docx files are supported for now: {file_path}")

    doc = Document(file_path)

    texts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            texts.append(text)

    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_text.append(cell_text)

            if row_text:
                texts.append(" | ".join(row_text))

    return "\n".join(texts)


if __name__ == "__main__":
    file_path = "data/Microsoft/PressReleaseFY26Q1.docx"

    text = load_docx(file_path)

    print("Loaded document successfully.")
    print(f"Total characters: {len(text)}")
    print("\nPreview:")
    print(text[:1000])