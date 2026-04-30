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


def ask_llm(context: str, question: str, prompt_type: str = "role") -> str: #"role", "cot", "fewshot"
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