# app/pipeline.py
from app.llm import summarize_article, extract_topic, generate_enriched_summary
from app.world_news_api import fetch_news
from app.wiki_api import get_wikipedia_summary


def run_pipeline():
    raw_data = fetch_news()
    articles = raw_data.get("top_news", [])

    results = []

    for article in articles[:1]:  # MVP: kun 1 artikel
        text = article.get("text", "") or article.get("summary", "") or article.get("title", "")

        if not text:
            continue

        summary = summarize_article(text)
        topic = extract_topic(text)
        wiki_context = get_wikipedia_summary(topic) or "No additional background information found."

        enriched = generate_enriched_summary(text, summary, wiki_context)

        results.append({
            "title": article.get("title"),
            "summary": summary,
            "topic": topic,
            "wiki_context": wiki_context,
            "enriched_summary": enriched,
        })

    return results