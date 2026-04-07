# app/llm.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"


def _call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
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
Extract the main entity, person, organization, event, or subject of the article.
Return ONLY one short Wikipedia-friendly topic.

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