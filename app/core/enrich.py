# app/core/enrich.py
from __future__ import annotations

from typing import Any


Cluster = dict[str, Any]


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length, preserving whole-word readability where possible.
    """
    text = text.strip()
    if len(text) <= max_length:
        return text

    truncated = text[:max_length].rsplit(" ", 1)[0].strip()
    if not truncated:
        truncated = text[:max_length].strip()

    return f"{truncated}..."


def build_cluster_id(cluster: Cluster) -> str:
    """
    Build a stable cluster identifier from the cluster rank if available.
    """
    rank = cluster.get("cluster_rank")
    if isinstance(rank, int):
        return f"cluster_{rank:03d}"
    return "cluster_unknown"


def count_supporting_sources(cluster: Cluster) -> int:
    """
    Count unique supporting sources in a cluster.
    """
    sources = cluster.get("supporting_sources", [])
    if not isinstance(sources, list):
        return 0

    unique_sources = {str(source).strip() for source in sources if str(source).strip()}
    return len(unique_sources)


def enrich_cluster(cluster: Cluster, preview_length: int = 200) -> Cluster:
    """
    Add derived metadata to a single processed cluster.
    """
    enriched = dict(cluster)

    text = str(enriched.get("text", "") or "")
    supporting_articles = enriched.get("supporting_articles", [])
    supporting_sources = enriched.get("supporting_sources", [])

    enriched["cluster_id"] = enriched.get("cluster_id") or build_cluster_id(enriched)
    enriched["text_preview"] = truncate_text(text, max_length=preview_length)
    enriched["num_supporting_articles"] = (
        len(supporting_articles) if isinstance(supporting_articles, list) else 0
    )
    enriched["num_supporting_sources"] = count_supporting_sources(enriched)
    enriched["primary_source"] = str(enriched.get("source_name", "") or "").strip()
    enriched["has_supporting_articles"] = bool(supporting_articles)
    enriched["has_supporting_sources"] = bool(supporting_sources)

    return enriched


def enrich_clusters(clusters: list[Cluster], preview_length: int = 200) -> list[Cluster]:
    """
    Add derived metadata to a list of processed clusters.
    """
    return [enrich_cluster(cluster, preview_length=preview_length) for cluster in clusters]


def build_display_record(cluster: Cluster) -> dict[str, Any]:
    """
    Build a lightweight display-friendly record from an enriched cluster.
    Useful for frontend JSON output or summary overviews.
    """
    return {
        "cluster_id": cluster.get("cluster_id"),
        "cluster_rank": cluster.get("cluster_rank"),
        "title": cluster.get("title"),
        "url": cluster.get("url"),
        "source_name": cluster.get("source_name"),
        "published_at": cluster.get("published_at"),
        "text_preview": cluster.get("text_preview"),
        "num_supporting_articles": cluster.get("num_supporting_articles", 0),
        "num_supporting_sources": cluster.get("num_supporting_sources", 0),
        "supporting_sources": cluster.get("supporting_sources", []),
    }


def build_display_records(clusters: list[Cluster]) -> list[dict[str, Any]]:
    """
    Build display-friendly records for a list of clusters.
    """
    return [build_display_record(cluster) for cluster in clusters]