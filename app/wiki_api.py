# app/wiki_api.py
import requests

WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

HEADERS = {
    "User-Agent": "news-enrichment-mvp/0.1"
}


def search_wikipedia(topic: str, limit: int = 5) -> list[dict]:
    params = {
        "q": topic,
        "limit": limit
    }

    response = requests.get(
        WIKIPEDIA_SEARCH_URL,
        headers=HEADERS,
        params=params,
        timeout=10
    )
    response.raise_for_status()

    data = response.json()
    return data.get("pages", [])


def get_wikipedia_summary(topic: str) -> str | None:
    pages = search_wikipedia(topic, limit=1)

    if not pages:
        return None

    title = pages[0].get("title")
    if not title:
        return None

    summary_response = requests.get(
        f"{WIKIPEDIA_SUMMARY_URL}{title}",
        headers=HEADERS,
        timeout=10
    )
    summary_response.raise_for_status()

    summary_data = summary_response.json()
    return summary_data.get("extract")