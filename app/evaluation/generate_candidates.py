# app/evaluation/generate_candidates.py
from __future__ import annotations

from typing import Any

from app.production.summarize import summarize_clusters


# Type aliases for clarity
Cluster = dict[str, Any]
CandidateSummary = dict[str, Any]


# Builds a unique cluster ID based on cluster rank
def build_cluster_id(cluster: Cluster) -> str:
    rank = cluster.get("cluster_rank")
    
    # Format rank as zero-padded ID (e.g., cluster_001)
    if isinstance(rank, int):
        return f"cluster_{rank:03d}"
    
    # Fallback if rank is missing or invalid
    return "cluster_unknown"


# Truncates long text to a maximum length without cutting words in half
def truncate_text(text: str, max_length: int = 200) -> str:
    text = text.strip()
    
    # Return as-is if already short enough
    if len(text) <= max_length:
        return text

    # Cut text and avoid breaking words
    truncated = text[:max_length].rsplit(" ", 1)[0].strip()
    
    # Fallback if no space found (single long word)
    if not truncated:
        truncated = text[:max_length].strip()

    return f"{truncated}..."


# Extracts unique source names from supporting articles
def extract_supporting_sources(cluster: Cluster) -> list[str]:
    supporting_articles = cluster.get("supporting_articles", [])
    
    # Validate structure
    if not isinstance(supporting_articles, list):
        return []

    seen: set[str] = set()
    sources: list[str] = []

    # Collect unique source names
    for article in supporting_articles:
        if not isinstance(article, dict):
            continue

        source_name = str(article.get("source_name", "") or "").strip()
        
        # Avoid duplicates
        if source_name and source_name not in seen:
            seen.add(source_name)
            sources.append(source_name)

    return sources


# Generates summaries for each cluster using multiple models
def generate_candidate_summaries(
    clusters: list[Cluster],
    groq_api_key: str | None,
    generation_models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Generate candidate summaries for each cluster using multiple models.
    """
    
    # Map to group results per cluster
    cluster_map: dict[str, dict[str, Any]] = {}

    # Loop through each model configuration
    for model_cfg in generation_models:
        model_name = model_cfg.get("model_name", "llama-3.3-70b-versatile")
        temperature = model_cfg.get("temperature", 0.2)
        max_output_tokens = model_cfg.get("max_output_tokens", 400)
        prompt_version = model_cfg.get("prompt_version", "summary_prompt_v1")

        # Generate summaries for all clusters using the current model
        summaries = summarize_clusters(
            clusters=clusters,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            prompt_version=prompt_version,
        )

        # Combine clusters with their generated summaries
        for cluster, summary in zip(clusters, summaries, strict=False):
            cluster_id = build_cluster_id(cluster)

            # Initialize cluster entry if not already created
            if cluster_id not in cluster_map:
                text = str(cluster.get("text", "") or "")
                cluster_map[cluster_id] = {
                    "cluster_id": cluster_id,
                    "cluster_rank": cluster.get("cluster_rank"),
                    "title": cluster.get("title"),
                    "url": cluster.get("url"),
                    "source_name": cluster.get("source_name"),
                    "published_at": cluster.get("published_at"),
                    "text": text,
                    "text_preview": truncate_text(text, max_length=200),
                    "supporting_sources": extract_supporting_sources(cluster),
                    "supporting_articles": cluster.get("supporting_articles", []),
                    "candidates": [],  # Store summaries from different models
                }

            # Add the current model's summary to the cluster's candidates
            cluster_map[cluster_id]["candidates"].append(summary)

    # Return list of clusters with all candidate summaries
    return list(cluster_map.values())