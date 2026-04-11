# app/core/preprocess.py
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


# Type alias for readability
Article = dict[str, Any]


# Cleans text by removing extra whitespace and trimming spaces
def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    return " ".join(text.split()).strip()


# Safely converts any value to a cleaned string
def safe_str(value: Any) -> str:
    """Convert a value to a cleaned string, returning an empty string for None."""
    if value is None:
        return ""
    
    # Convert to string and normalize whitespace
    return normalize_whitespace(str(value))


# Extracts the best available text content from an article
def get_article_text(article: Article) -> str:
    """
    Return the best available body text from an article.

    Priority:
    1. text
    2. summary
    """
    # Try full text first, then fallback to summary
    for candidate in (article.get("text"), article.get("summary")):
        cleaned = safe_str(candidate)
        if cleaned:
            return cleaned

    # Return empty string if no usable text is found
    return ""


# Extracts the source name of an article
def extract_source_name(article: Article) -> str:
    """
    Return source_name if present, otherwise derive it from the article URL hostname.
    """
    # Use explicit source name if available
    explicit_source = safe_str(article.get("source_name"))
    if explicit_source:
        return explicit_source

    # Otherwise extract from URL
    url = safe_str(article.get("url"))
    if not url:
        return ""

    try:
        # Parse hostname from URL
        hostname = urlparse(url).netloc.lower()
        
        # Remove "www." prefix if present
        if hostname.startswith("www."):
            hostname = hostname[4:]
        
        return hostname
    except Exception:
        # Return empty string if URL parsing fails
        return ""


# Validates whether an article is usable for processing
def is_valid_article(article: Article, min_text_length: int = 1) -> bool:
    """Check whether an article has enough usable content to be processed."""
    
    # Extract and clean required fields
    title = safe_str(article.get("title"))
    text = get_article_text(article)
    
    # Valid if title exists, text exists, and text meets minimum length
    return bool(title and text and len(text) >= min_text_length)