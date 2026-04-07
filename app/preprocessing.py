# app/preprocessing.py
from __future__ import annotations

from typing import Any

def flatten_top_news(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten the World News API top_news structure into a simple article list.
    Expected structure:
    {
      "top_news": [
        {"news": [article1, article2, ...]},
        ...
      ]
    }
    """
    groups = raw_data.get("top_news", [])
    articles: list[dict[str, Any]] = []

    for group in groups:
        news_items = group.get("news", [])
        if isinstance(news_items, list):
            articles.extend(news_items)

    return articles


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def get_article_text(article: dict[str, Any]) -> str:
    """
    Pick the best available text field from an article.
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


def preprocess_articles(
    raw_data: dict[str, Any],
    max_articles: int = 10,
    min_text_length: int = 80,
) -> list[dict[str, Any]]:
    """
    Basic preprocessing:
    - flatten API response
    - remove articles without usable text/title
    - normalize whitespace
    - deduplicate by url/title
    - limit number of articles
    """
    flat_articles = flatten_top_news(raw_data)
    seen_keys: set[str] = set()
    processed: list[dict[str, Any]] = []

    for article in flat_articles:
        title = normalize_whitespace(str(article.get("title", "") or ""))
        url = normalize_whitespace(str(article.get("url", "") or ""))
        source = normalize_whitespace(str(article.get("source_name", "") or ""))

        text = get_article_text(article)
        if not title or not text:
            continue

        if len(text) < min_text_length:
            continue

        dedupe_key = url or title.lower()
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        processed_article = {
            "title": title,
            "url": url,
            "source_name": source,
            "published_at": article.get("publish_date"),
            "text": text,
            "text_length": len(text),
            "has_full_text": bool(article.get("text")),
            "has_summary_field": bool(article.get("summary")),
        }
        processed.append(processed_article)

        if len(processed) >= max_articles:
            break

    return processed