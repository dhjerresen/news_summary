# app/production/summarize.py
from __future__ import annotations

import os
import time
from typing import Any

from groq import Groq


# Type aliases for clarity
Cluster = dict[str, Any]
SummaryRecord = dict[str, Any]


# Initializes and returns a Groq client using API key
def get_groq_client(api_key: str | None = None) -> Groq:
    """
    Create and return a Groq client.
    """
    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
    
    # Raise error if API key is missing
    if not resolved_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    return Groq(api_key=resolved_api_key)


# Builds the prompt used for summarizing a cluster
def build_summary_prompt(cluster: Cluster) -> str:
    """
    Build the production summarization prompt for one cluster.
    """
    # Extract title and text from cluster
    title = str(cluster.get("title", "") or "").strip()
    text = str(cluster.get("text", "") or "").strip()

    # Construct instruction prompt for the LLM
    return f"""
You are a news assistant.

Summarize the following news article cluster in 2-3 concise sentences.
Focus only on the most important facts.
Do not speculate.
Do not add information not supported by the input text.

Title:
{title}

Cluster text:
{text}
""".strip()


# Summarizes a single cluster using the Groq API
def summarize_cluster(
    client: Groq,
    cluster: Cluster,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_output_tokens: int = 400,
    prompt_version: str = "summary_prompt_v1",
) -> SummaryRecord:
    """
    Summarize a single cluster with Groq.
    """
    # Build prompt and start latency timer
    prompt = build_summary_prompt(cluster)
    started_at = time.perf_counter()

    # Send request to LLM
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    # Measure response time
    latency_seconds = round(time.perf_counter() - started_at, 3)

    # Extract generated summary text
    summary_text = ""
    if response.choices and response.choices[0].message:
        summary_text = (response.choices[0].message.content or "").strip()

    # Extract token usage statistics if available
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    # Build cluster ID from rank
    cluster_id = f"cluster_{int(cluster['cluster_rank']):03d}"

    # Return structured summary result
    return {
        "cluster_id": cluster_id,
        "summary": summary_text,
        "summary_length": len(summary_text),
        "latency_seconds": latency_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "success": bool(summary_text),  # True if summary is not empty
        "error": None,
        "model_name": model_name,
        "prompt_version": prompt_version,
    }


# Summarizes multiple clusters and handles failures gracefully
def summarize_clusters(
    clusters: list[Cluster],
    groq_api_key: str | None = None,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_output_tokens: int = 400,
    prompt_version: str = "summary_prompt_v1",
) -> list[SummaryRecord]:
    """
    Summarize a list of clusters.

    Continues even if individual cluster summaries fail.
    """
    # Initialize Groq client
    client = get_groq_client(api_key=groq_api_key)
    summaries: list[SummaryRecord] = []

    # Loop through each cluster
    for cluster in clusters:
        cluster_id = f"cluster_{int(cluster['cluster_rank']):03d}"

        try:
            # Attempt to summarize cluster
            summary_record = summarize_cluster(
                client=client,
                cluster=cluster,
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                prompt_version=prompt_version,
            )
        except Exception as exc:
            # If summarization fails, create a fallback record with error info
            summary_record = {
                "cluster_id": cluster_id,
                "summary": "",
                "summary_length": 0,
                "latency_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "success": False,
                "error": str(exc),
                "model_name": model_name,
                "prompt_version": prompt_version,
            }

        # Store result
        summaries.append(summary_record)

    return summaries