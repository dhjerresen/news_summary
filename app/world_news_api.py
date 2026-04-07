# app/world_news_api.py
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

def fetch_news() -> dict:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("WORLDNEWS_API_KEY")
    if not api_key:
        raise ValueError("WORLDNEWS_API_KEY not found in .env")

    today = date.today().isoformat()
    url = "https://api.worldnewsapi.com/top-news"

    params = {
        "source-country": "us",
        "language": "en",
        "date": today,
    }

    headers = {"x-api-key": api_key}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    result = fetch_news()
    print(result)