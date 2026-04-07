import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Hent API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API key not found")

client = Groq(api_key=api_key)

MODEL_NAME = "llama-3.3-70b-versatile"

def _call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=MODEL_NAME,
    )
    return response.choices[0].message.content.strip()


def summarize_article(text: str) -> str:
    prompt = f"""
You are a news assistant.

Summarize the following news article in 2-3 sentences.

Article:
{text}
"""
    return _call_llm(prompt)


def extract_topic(text: str) -> str:
    prompt = f"""
Extract the main topic of the following news article.

Return ONLY a short topic (max 5 words).

Article:
{text}
"""
    return _call_llm(prompt)


def generate_enriched_summary(article: str, summary: str, wiki_context: str) -> str:
    prompt = f"""
You are a helpful assistant.

Given a news article, a short summary, and background knowledge, create an enriched summary.

Keep it concise (3-5 sentences) and include relevant background context.

Article:
{article}

Summary:
{summary}

Background:
{wiki_context}
"""
    return _call_llm(prompt)