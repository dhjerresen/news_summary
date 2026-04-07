# app/pipeline.py
from app.llm import summarize_article, extract_topic, generate_enriched_summary
from app.world_news_api import fetch_news
from app.wiki_api import get_wikipedia_summary


def run_pipeline():
    raw_data = fetch_news()

    if not isinstance(raw_data, dict):
        print("Unexpected API response:", raw_data)
        return []

    top_news_groups = raw_data.get("top_news", [])
    results = []

    for group in top_news_groups:
        news_items = group.get("news", [])

        for article in news_items[:1]:  # MVP: kun 1 artikel
            text = article.get("text", "") or article.get("summary", "") or article.get("title", "")

            if not text:
                print("Skipping article with no usable text:", article)
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

            return results  # stop efter første artikel i hele pipelinen

    return results