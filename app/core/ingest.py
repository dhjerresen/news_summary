# app/core/ingest.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

# Default timeout for HTTP requests (in seconds)
DEFAULT_TIMEOUT_SECONDS = 30


# Custom exception for ingestion-related errors
class IngestError(Exception):
    """Raised when data ingestion fails."""


# Fetches JSON data from a given HTTP endpoint
def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Fetch JSON data from an HTTP endpoint.

    Raises:
        IngestError: if the request fails or the response is not valid JSON.
    """
    try:
        # Send GET request with optional parameters and headers
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        # Raise exception for HTTP errors (e.g., 404, 500)
        response.raise_for_status()
    except requests.RequestException as exc:
        # Wrap request-related errors in a custom exception
        raise IngestError(f"HTTP request failed: {exc}") from exc

    try:
        # Parse response as JSON
        data = response.json()
    except ValueError as exc:
        # Handle invalid JSON responses
        raise IngestError("Response was not valid JSON.") from exc

    # Ensure the JSON response is a dictionary
    if not isinstance(data, dict):
        raise IngestError("Expected JSON object response.")

    return data


# Loads JSON data from a local file
def load_raw_json(path: str | Path) -> dict[str, Any]:
    """
    Load raw JSON payload from disk.

    Raises:
        IngestError: if the file cannot be read or parsed.
    """
    path = Path(path)

    try:
        # Open and parse JSON file
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        # Handle file read errors
        raise IngestError(f"Could not read file: {path}") from exc
    except ValueError as exc:
        # Handle invalid JSON content
        raise IngestError(f"Invalid JSON in file: {path}") from exc

    # Ensure the loaded data is a dictionary
    if not isinstance(data, dict):
        raise IngestError("Expected JSON object in file.")

    return data


# Fetches clustered top news data from the World News API
def fetch_world_news_top_news(
    api_key: str,
    source_country: str = "us",
    language: str = "en",
    date: str | None = None,
    news_sources: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Fetch clustered top-news data from World News API.
    """
    # API endpoint URL
    url = "https://api.worldnewsapi.com/top-news"

    # Required query parameters
    params: dict[str, Any] = {
        "source-country": source_country,
        "language": language,
        "api-key": api_key,
    }

    # Optional filters
    if date:
        params["date"] = date

    if news_sources:
        params["news-sources"] = news_sources

    # Use generic fetch_json function to make the request
    return fetch_json(url=url, params=params, timeout=timeout)


# Validates that the payload has the expected structure
def validate_top_news_payload(payload: dict[str, Any]) -> bool:
    """
    Validate that the payload looks like a World News API top-news response.
    """
    top_news = payload.get("top_news")
    
    # Check that "top_news" exists and is a list
    return isinstance(top_news, list)


# Extracts valid cluster groups from the payload
def extract_top_news_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract top_news groups from a World News API payload.

    Returns an empty list if the payload shape is invalid.
    """
    # Get "top_news" field, default to empty list
    top_news = payload.get("top_news", [])
    
    # Validate structure
    if not isinstance(top_news, list):
        return []

    # Return only valid dictionary entries (clusters)
    return [group for group in top_news if isinstance(group, dict)]