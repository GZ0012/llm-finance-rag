# src/llm_client.py

from openai import OpenAI
from config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def build_prompt(context: str, question: str) -> str:
    return f"""
You are a financial analyst assistant.

Answer the question ONLY based on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""


def ask_llm(context: str, question: str) -> str:
    prompt = build_prompt(context, question)

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content