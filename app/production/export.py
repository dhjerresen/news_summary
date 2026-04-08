# app/production/export.py
from __future__ import annotations

from typing import Any


Cluster = dict[str, Any]
SummaryRecord = dict[str, Any]


def build_frontend_record(cluster: Cluster, summary_record: SummaryRecord) -> dict[str, Any]:
    """
    Build a frontend-friendly record by combining cluster metadata and summary output.
    """
    return {
        "cluster_id": cluster.get("cluster_id"),
        "cluster_rank": cluster.get("cluster_rank"),
        "title": cluster.get("title"),
        "url": cluster.get("url"),
        "source_name": cluster.get("source_name"),
        "published_at": cluster.get("published_at"),
        "text_preview": cluster.get("text_preview"),
        "supporting_sources": cluster.get("supporting_sources", []),
        "num_supporting_sources": cluster.get("num_supporting_sources", 0),
        "num_supporting_articles": cluster.get("num_supporting_articles", 0),
        "summary": summary_record.get("summary", ""),
        "summary_length": summary_record.get("summary_length", 0),
        "model_name": summary_record.get("model_name"),
        "prompt_version": summary_record.get("prompt_version"),
        "latency_seconds": summary_record.get("latency_seconds"),
    }


def build_frontend_payload(
    enriched_clusters: list[Cluster],
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

    for cluster in enriched_clusters:
        cluster_id = str(cluster.get("cluster_id"))
        summary_record = summaries_by_cluster_id.get(cluster_id)

        if not summary_record:
            continue

        records.append(build_frontend_record(cluster, summary_record))

    return {
        "metadata": metadata,
        "items": records,
    }