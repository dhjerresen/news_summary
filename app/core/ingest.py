# app/core/ingest.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

DEFAULT_TIMEOUT_SECONDS = 30


class IngestError(Exception):
    """Raised when data ingestion fails."""


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
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise IngestError(f"HTTP request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise IngestError("Response was not valid JSON.") from exc

    if not isinstance(data, dict):
        raise IngestError("Expected JSON object response.")

    return data


def load_raw_json(path: str | Path) -> dict[str, Any]:
    """
    Load raw JSON payload from disk.

    Raises:
        IngestError: if the file cannot be read or parsed.
    """
    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise IngestError(f"Could not read file: {path}") from exc
    except ValueError as exc:
        raise IngestError(f"Invalid JSON in file: {path}") from exc

    if not isinstance(data, dict):
        raise IngestError("Expected JSON object in file.")

    return data


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
    url = "https://api.worldnewsapi.com/top-news"

    params: dict[str, Any] = {
        "source-country": source_country,
        "language": language,
        "api-key": api_key,
    }

    if date:
        params["date"] = date

    if news_sources:
        params["news-sources"] = news_sources

    return fetch_json(url=url, params=params, timeout=timeout)


def validate_top_news_payload(payload: dict[str, Any]) -> bool:
    """
    Validate that the payload looks like a World News API top-news response.
    """
    top_news = payload.get("top_news")
    return isinstance(top_news, list)


def extract_top_news_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract top_news groups from a World News API payload.

    Returns an empty list if the payload shape is invalid.
    """
    top_news = payload.get("top_news", [])
    if not isinstance(top_news, list):
        return []

    return [group for group in top_news if isinstance(group, dict)]