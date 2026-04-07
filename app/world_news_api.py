# app/world_news_api.py
import os
from dotenv import load_dotenv
import requests
from pathlib import Path
from datetime import date

def fetch_news():
    # Find projektets root (én mappe op fra denne fil)
    env_path = Path(__file__).resolve().parent.parent / ".env"

    load_dotenv(dotenv_path=env_path)

    # Hent API key
    api_key = os.getenv("WORLDNEWS_API_KEY")

    if not api_key:
        return "Error: API key not found in .env"

    # Dagens dato (automatisk)
    today = date.today().isoformat()

    url = "https://api.worldnewsapi.com/top-news"

    params = {
        "source-country": "us",  # Danmark
        "language": "en",        # Dansk
        "date": today            # Dynamisk dato
    }

    headers = {
        "x-api-key": api_key
    }

    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"


if __name__ == "__main__":
    result = my_custom_function()
    print(result)