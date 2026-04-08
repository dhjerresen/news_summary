# app/production/summarize.py
from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

Cluster = dict[str, Any]
SummaryRecord = dict[str, Any]


def get_groq_client(api_key: str | None = None) -> Groq:
    """
    Create and return a Groq client.
    """
    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return Groq(api_key=resolved_api_key)


def build_summary_prompt(cluster: Cluster) -> str:
    """
    Build the production summarization prompt for one cluster.
    """
    title = str(cluster.get("title", "") or "").strip()
    text = str(cluster.get("text", "") or "").strip()
    supporting_sources = cluster.get("supporting_sources", [])

    supporting_sources_text = ", ".join(
        str(source).strip() for source in supporting_sources if str(source).strip()
    )

    return f"""
You are a news assistant.

Summarize the following news cluster in 2-3 concise sentences.
Focus on the most important facts only.
Do not speculate.
Do not add information not supported by the source text.

Title:
{title}

Supporting sources:
{supporting_sources_text}

Cluster text:
{text}
""".strip()


def summarize_cluster(
    client: Groq,
    cluster: Cluster,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_output_tokens: int = 400,
    prompt_version: str = "summary_prompt_v1",
) -> SummaryRecord:
    """
    Summarize a single enriched cluster with Groq.
    """
    prompt = build_summary_prompt(cluster)
    started_at = time.perf_counter()

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    latency_seconds = round(time.perf_counter() - started_at, 3)

    summary_text = ""
    if response.choices and response.choices[0].message:
        summary_text = (response.choices[0].message.content or "").strip()

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return {
        "cluster_id": cluster.get("cluster_id"),
        "cluster_rank": cluster.get("cluster_rank"),
        "title": cluster.get("title"),
        "url": cluster.get("url"),
        "source_name": cluster.get("source_name"),
        "published_at": cluster.get("published_at"),
        "supporting_sources": cluster.get("supporting_sources", []),
        "summary": summary_text,
        "summary_length": len(summary_text),
        "model_name": model_name,
        "prompt_version": prompt_version,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "latency_seconds": latency_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "success": bool(summary_text),
    }


def summarize_clusters(
    clusters: list[Cluster],
    groq_api_key: str | None = None,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_output_tokens: int = 400,
    prompt_version: str = "summary_prompt_v1",
) -> list[SummaryRecord]:
    """
    Summarize a list of enriched clusters.
    """
    client = get_groq_client(api_key=groq_api_key)
    summaries: list[SummaryRecord] = []

    for cluster in clusters:
        summary_record = summarize_cluster(
            client=client,
            cluster=cluster,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            prompt_version=prompt_version,
        )
        summaries.append(summary_record)

    return summaries