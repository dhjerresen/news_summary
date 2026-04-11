# app/core/cluster.py
from __future__ import annotations

from typing import Any

from app.core.preprocess import (
    extract_source_name,
    get_article_text,
    is_valid_article,
    safe_str,
)


# Type aliases for clarity
Article = dict[str, Any]
Cluster = dict[str, Any]


# Scores an article based on quality for selecting a representative article
def score_article(article: Article) -> tuple[int, int]:
    """
    Score article quality for representative selection.

    Priority:
    1. article has full text
    2. longer usable text
    """
    # Extract full text and fallback summary safely
    full_text = safe_str(article.get("text"))
    fallback_summary = safe_str(article.get("summary"))
    
    # Use full text if available, otherwise fallback to summary
    usable_text = full_text or fallback_summary

    # Binary score: 1 if full text exists, otherwise 0
    has_full_text = 1 if full_text else 0
    
    # Secondary score: length of usable text
    text_length = len(usable_text)

    return has_full_text, text_length


# Selects the best article from a cluster based on the scoring function
def select_representative_article(news_items: list[Article]) -> Article | None:
    """
    Select the best representative article from a list of articles.
    """
    # Filter out invalid articles using validation function
    valid_items = [article for article in news_items if is_valid_article(article)]

    # Return None if no valid articles exist
    if not valid_items:
        return None

    # Select the article with the highest score
    return max(valid_items, key=score_article)


# Creates a simplified list of supporting articles (metadata only)
def build_supporting_articles(news_items: list[Article]) -> list[Article]:
    """
    Build lightweight supporting article metadata.
    """
    supporting_articles: list[Article] = []

    # Extract relevant metadata for each article in the cluster
    for item in news_items:
        supporting_articles.append(
            {
                "title": safe_str(item.get("title")),
                "url": safe_str(item.get("url")),
                "source_name": extract_source_name(item),
                "published_at": item.get("publish_date"),
            }
        )

    return supporting_articles


# Main function to preprocess clusters from raw API data
def preprocess_clusters(
    raw_data: dict[str, Any],
    max_clusters: int = 10,
    min_text_length: int = 80,
) -> list[Cluster]:
    """
    Process World News API /top-news cluster data.

    Returns one representative record per cluster.
    """
    # Extract cluster groups from raw data
    groups = raw_data.get("top_news", [])
    
    # Validate structure: must be a list
    if not isinstance(groups, list):
        return []

    processed: list[Cluster] = []

    # Iterate through each cluster group with ranking
    for cluster_rank, group in enumerate(groups, start=1):
        # Skip invalid group structures
        if not isinstance(group, dict):
            continue

        # Extract news articles in the cluster
        news_items = group.get("news", [])
        if not isinstance(news_items, list) or not news_items:
            continue

        # Select the best representative article for the cluster
        representative = select_representative_article(news_items)
        if representative is None:
            continue

        # Extract key fields from the representative article
        title = safe_str(representative.get("title"))
        url = safe_str(representative.get("url"))
        source_name = extract_source_name(representative)
        text = get_article_text(representative)

        # Skip if required fields are missing
        if not title or not text:
            continue

        # Filter out articles that are too short
        if len(text) < min_text_length:
            continue

        # Build list of supporting articles (metadata only)
        supporting_articles = build_supporting_articles(news_items)

        # Create the processed cluster object
        processed_cluster: Cluster = {
            "cluster_rank": cluster_rank,
            "title": title,
            "url": url,
            "source_name": source_name,
            "published_at": representative.get("publish_date"),
            "text": text,
            "supporting_articles": supporting_articles,
        }
        processed.append(processed_cluster)

        # Stop early if maximum number of clusters is reached
        if len(processed) >= max_clusters:
            break

    # Return processed clusters ready for downstream pipeline steps
    return processed