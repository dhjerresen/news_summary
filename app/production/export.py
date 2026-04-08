# app/production/export.py
from __future__ import annotations

from typing import Any


Cluster = dict[str, Any]
SummaryRecord = dict[str, Any]


def build_cluster_id(cluster: Cluster) -> str:
    rank = cluster.get("cluster_rank")
    if isinstance(rank, int):
        return f"cluster_{rank:03d}"
    return "cluster_unknown"


def truncate_text(text: str, max_length: int = 200) -> str:
    text = text.strip()
    if len(text) <= max_length:
        return text

    truncated = text[:max_length].rsplit(" ", 1)[0].strip()
    if not truncated:
        truncated = text[:max_length].strip()

    return f"{truncated}..."


def extract_supporting_sources(cluster: Cluster) -> list[str]:
    supporting_articles = cluster.get("supporting_articles", [])
    if not isinstance(supporting_articles, list):
        return []

    seen: set[str] = set()
    sources: list[str] = []

    for article in supporting_articles:
        if not isinstance(article, dict):
            continue

        source_name = str(article.get("source_name", "") or "").strip()
        if source_name and source_name not in seen:
            seen.add(source_name)
            sources.append(source_name)

    return sources


def build_frontend_record(cluster: Cluster, summary_record: SummaryRecord) -> dict[str, Any]:
    supporting_articles = cluster.get("supporting_articles", [])
    if not isinstance(supporting_articles, list):
        supporting_articles = []

    supporting_sources = extract_supporting_sources(cluster)
    text = str(cluster.get("text", "") or "")

    return {
        "cluster_id": build_cluster_id(cluster),
        "cluster_rank": cluster.get("cluster_rank"),
        "title": cluster.get("title"),
        "url": cluster.get("url"),
        "source_name": cluster.get("source_name"),
        "published_at": cluster.get("published_at"),
        "text_preview": truncate_text(text, max_length=200),
        "supporting_sources": supporting_sources,
        "supporting_articles": supporting_articles,
        "num_supporting_sources": len(supporting_sources),
        "num_supporting_articles": len(supporting_articles),
        "summary": summary_record.get("summary", ""),
        "summary_length": summary_record.get("summary_length", 0),
        "model_name": summary_record.get("model_name"),
        "prompt_version": summary_record.get("prompt_version"),
        "latency_seconds": summary_record.get("latency_seconds"),
        "input_tokens": summary_record.get("input_tokens"),
        "output_tokens": summary_record.get("output_tokens"),
        "total_tokens": summary_record.get("total_tokens"),
        "success": summary_record.get("success", False),
        "error": summary_record.get("error"),
    }


def build_frontend_payload(
    clusters: list[Cluster],
    summaries: list[SummaryRecord],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the final frontend payload for downstream consumption.
    """
    summaries_by_cluster_id = {
        str(summary.get("cluster_id")): summary
        for summary in summaries
        if summary.get("cluster_id") is not None
    }

    records: list[dict[str, Any]] = []

    for cluster in clusters:
        cluster_id = build_cluster_id(cluster)
        summary_record = summaries_by_cluster_id.get(cluster_id)

        if not summary_record:
            continue

        records.append(build_frontend_record(cluster, summary_record))

    return {
        "metadata": metadata,
        "items": records,
    }