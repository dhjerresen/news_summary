# app/preprocessing.py
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def get_article_text(article: dict[str, Any]) -> str:
    """
    Pick the best available text field from an article.
    Priority:
    1. full text
    2. summary
    3. title
    """
    candidates = [
        article.get("text", ""),
        article.get("summary", ""),
        article.get("title", ""),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return normalize_whitespace(candidate)

    return ""


def extract_source_name(article: dict[str, Any]) -> str:
    """
    Use source_name if present, otherwise derive it from the URL hostname.
    """
    explicit_source = normalize_whitespace(str(article.get("source_name", "") or ""))
    if explicit_source:
        return explicit_source

    url = str(article.get("url", "") or "").strip()
    if not url:
        return ""

    try:
        hostname = urlparse(url).netloc.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


def score_article(article: dict[str, Any]) -> tuple[int, int]:
    """
    Score article quality for representative selection.
    Priority:
    1. full text exists
    2. longer usable text
    """
    full_text = normalize_whitespace(str(article.get("text", "") or ""))
    fallback_summary = normalize_whitespace(str(article.get("summary", "") or ""))
    usable_text = full_text or fallback_summary

    has_full_text = 1 if full_text else 0
    length = len(usable_text)

    return (has_full_text, length)


def select_representative_article(news_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Select the best single article from a cluster.
    """
    valid_items: list[dict[str, Any]] = []

    for article in news_items:
        title = normalize_whitespace(str(article.get("title", "") or ""))
        text = get_article_text(article)
        if title and text:
            valid_items.append(article)

    if not valid_items:
        return None

    return max(valid_items, key=score_article)


def preprocess_clusters(
    raw_data: dict[str, Any],
    max_clusters: int = 10,
    min_text_length: int = 80,
) -> list[dict[str, Any]]:
    """
    Process World News API /top-news as clusters, not as a flat article list.

    Returns one representative record per cluster.
    """
    groups = raw_data.get("top_news", [])
    processed: list[dict[str, Any]] = []

    for cluster_rank, group in enumerate(groups, start=1):
        news_items = group.get("news", [])
        if not isinstance(news_items, list) or not news_items:
            continue

        representative = select_representative_article(news_items)
        if representative is None:
            continue

        title = normalize_whitespace(str(representative.get("title", "") or ""))
        url = normalize_whitespace(str(representative.get("url", "") or ""))
        source_name = extract_source_name(representative)
        text = get_article_text(representative)

        if not title or not text:
            continue

        if len(text) < min_text_length:
            continue

        supporting_sources: list[str] = []
        seen_sources: set[str] = set()
        supporting_articles: list[dict[str, Any]] = []

        for item in news_items:
            item_title = normalize_whitespace(str(item.get("title", "") or ""))
            item_url = normalize_whitespace(str(item.get("url", "") or ""))
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

        processed_cluster = {
            "cluster_rank": cluster_rank,
            "cluster_size": len(news_items),
            "title": title,
            "url": url,
            "source_name": source_name,
            "published_at": representative.get("publish_date"),
            "text": text,
            "text_length": len(text),
            "has_full_text": bool(representative.get("text")),
            "has_summary_field": bool(representative.get("summary")),
            "supporting_sources": supporting_sources,
            "supporting_articles": supporting_articles,
        }
        processed.append(processed_cluster)

        if len(processed) >= max_clusters:
            break

    return processed