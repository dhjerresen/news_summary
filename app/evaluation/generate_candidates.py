# app/evaluation/generate_candidates.py
from __future__ import annotations

from typing import Any

from app.production.summarize import summarize_clusters


Cluster = dict[str, Any]
CandidateSummary = dict[str, Any]


def generate_candidate_summaries(
    clusters: list[Cluster],
    groq_api_key: str | None,
    generation_models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Generate candidate summaries for each cluster using multiple models.

    Returns one record per cluster:
    {
      "cluster_id": "...",
      "title": "...",
      "source_name": "...",
      "published_at": "...",
      "supporting_sources": [...],
      "supporting_articles": [...],
      "text_preview": "...",
      "candidates": [...]
    }
    """
    cluster_map: dict[str, dict[str, Any]] = {}

    for model_cfg in generation_models:
        model_name = model_cfg.get("model_name", "llama-3.3-70b-versatile")
        temperature = model_cfg.get("temperature", 0.2)
        max_output_tokens = model_cfg.get("max_output_tokens", 400)
        prompt_version = model_cfg.get("prompt_version", "summary_prompt_v1")

        summaries = summarize_clusters(
            clusters=clusters,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            prompt_version=prompt_version,
        )

        for cluster, summary in zip(clusters, summaries, strict=False):
            cluster_id = str(cluster.get("cluster_id"))

            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = {
                    "cluster_id": cluster.get("cluster_id"),
                    "cluster_rank": cluster.get("cluster_rank"),
                    "title": cluster.get("title"),
                    "url": cluster.get("url"),
                    "source_name": cluster.get("source_name"),
                    "published_at": cluster.get("published_at"),
                    "text": cluster.get("text"),
                    "text_preview": cluster.get("text_preview"),
                    "supporting_sources": cluster.get("supporting_sources", []),
                    "supporting_articles": cluster.get("supporting_articles", []),
                    "candidates": [],
                }

            cluster_map[cluster_id]["candidates"].append(summary)

    return list(cluster_map.values())