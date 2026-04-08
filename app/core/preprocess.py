# app/core/preprocess.py
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


Article = dict[str, Any]


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    return " ".join(text.split()).strip()


def safe_str(value: Any) -> str:
    """Convert a value to a cleaned string, returning an empty string for None."""
    if value is None:
        return ""
    return normalize_whitespace(str(value))


def get_article_text(article: Article) -> str:
    """
    Return the best available text field from an article.

    Priority:
    1. text
    2. summary
    3. title
    """
    candidates = (
        article.get("text"),
        article.get("summary"),
        article.get("title"),
    )

    for candidate in candidates:
        cleaned = safe_str(candidate)
        if cleaned:
            return cleaned

    return ""


def extract_source_name(article: Article) -> str:
    """
    Return source_name if present, otherwise derive it from the article URL hostname.
    """
    explicit_source = safe_str(article.get("source_name"))
    if explicit_source:
        return explicit_source

    url = safe_str(article.get("url"))
    if not url:
        return ""

    try:
        hostname = urlparse(url).netloc.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


def is_valid_article(article: Article, min_text_length: int = 1) -> bool:
    """Check whether an article has enough usable content to be processed."""
    title = safe_str(article.get("title"))
    text = get_article_text(article)
    return bool(title and text and len(text) >= min_text_length)