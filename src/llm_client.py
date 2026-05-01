from pathlib import Path
from openai import OpenAI
from config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "src" / "prompts"


def load_prompt_template(prompt_type: str) -> str:
    path = PROMPT_DIR / f"{prompt_type}.md"

    if not path.exists():
        raise ValueError(f"Prompt not found: {prompt_type}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(context: str, question: str, prompt_type: str) -> str:
    template = load_prompt_template(prompt_type)

    prompt = template.replace("{{context}}", context)
    prompt = prompt.replace("{{question}}", question)

    return prompt


def ask_llm(context: str, question: str, prompt_type: str = "role") -> str: #"role", "cot", "fewshot", "global"
    prompt = build_prompt(context, question, prompt_type)

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a financial assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


def generate_summary(chunks: list, max_context_chars: int = 24000) -> str:
    """
    Summarize a document from its chunks at indexing time.

    Joins chunks up to max_context_chars so the summary prompt doesn't
    overflow the LLM's context window. For most financial press releases
    and earnings transcripts this limit is never hit.
    """
    joined = "\n\n".join(chunks)

    # Truncate if the document is very long (e.g. annual reports / 10-Ks).
    if len(joined) > max_context_chars:
        joined = joined[:max_context_chars] + "\n\n[document truncated]"

    prompt = (
        "You are a professional financial analyst.\n"
        "Read the following document and write a structured summary that covers:\n"
        "- The company and reporting period\n"
        "- Key financial results (revenue, profit, growth rates)\n"
        "- Major business highlights and segment performance\n"
        "- Forward-looking statements or guidance\n\n"
        f"{joined}\n\nSummary:"
    )

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content