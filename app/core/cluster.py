# app/core/cluster.py
from __future__ import annotations

from typing import Any

from app.core.preprocess import (
    extract_source_name,
    get_article_text,
    is_valid_article,
    safe_str,
)


Article = dict[str, Any]
Cluster = dict[str, Any]


def score_article(article: Article) -> tuple[int, int]:
    """
    Score article quality for representative selection.

    Priority:
    1. article has full text
    2. longer usable text
    """
    full_text = safe_str(article.get("text"))
    fallback_summary = safe_str(article.get("summary"))
    usable_text = full_text or fallback_summary

    has_full_text = 1 if full_text else 0
    text_length = len(usable_text)

    return has_full_text, text_length


def select_representative_article(news_items: list[Article]) -> Article | None:
    """
    Select the best representative article from a list of articles.
    """
    valid_items = [article for article in news_items if is_valid_article(article)]

    if not valid_items:
        return None

    return max(valid_items, key=score_article)


def build_supporting_articles(news_items: list[Article]) -> tuple[list[str], list[Article]]:
    """
    Build supporting source list and lightweight supporting article metadata.
    """
    supporting_sources: list[str] = []
    seen_sources: set[str] = set()
    supporting_articles: list[Article] = []

    for item in news_items:
        item_title = safe_str(item.get("title"))
        item_url = safe_str(item.get("url"))
        item_source = extract_source_name(item)
        item_published_at = item.get("publish_date")

        if item_source and item_source not in seen_sources:
            seen_sources.add(item_source)
            supporting_sources.append(item_source)

        supporting_articles.append(
            {
                "title": item_title,
                "url": item_url,
                "source_name": item_source,
                "published_at": item_published_at,
            }
        )

    return supporting_sources, supporting_articles


def preprocess_clusters(
    raw_data: dict[str, Any],
    max_clusters: int = 10,
    min_text_length: int = 80,
) -> list[Cluster]:
    """
    Process World News API /top-news cluster data.

    Returns one representative record per cluster.
    """
    groups = raw_data.get("top_news", [])
    if not isinstance(groups, list):
        return []

    processed: list[Cluster] = []

    for cluster_rank, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            continue

        news_items = group.get("news", [])
        if not isinstance(news_items, list) or not news_items:
            continue

        representative = select_representative_article(news_items)
        if representative is None:
            continue

        title = safe_str(representative.get("title"))
        url = safe_str(representative.get("url"))
        source_name = extract_source_name(representative)
        text = get_article_text(representative)

        if not title or not text:
            continue

        if len(text) < min_text_length:
            continue

        supporting_sources, supporting_articles = build_supporting_articles(news_items)

        processed_cluster: Cluster = {
            "cluster_rank": cluster_rank,
            "cluster_size": len(news_items),
            "title": title,
            "url": url,
            "source_name": source_name,
            "published_at": representative.get("publish_date"),
            "text": text,
            "text_length": len(text),
            "has_full_text": bool(safe_str(representative.get("text"))),
            "has_summary_field": bool(safe_str(representative.get("summary"))),
            "supporting_sources": supporting_sources,
            "supporting_articles": supporting_articles,
        }
        processed.append(processed_cluster)

        if len(processed) >= max_clusters:
            break

    return processed