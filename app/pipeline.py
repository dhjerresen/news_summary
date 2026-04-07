# app/pipeline.py
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from app.llm import (
    MODEL_NAME,
    extract_topic,
    generate_enriched_summary,
    summarize_article,
)
from app.preprocessing import preprocess_articles
from app.utils import (
    append_text_line,
    create_run_id,
    ensure_dir,
    save_json,
    save_text,
    utc_now_iso,
)
from app.wiki_api import get_wikipedia_summary
from app.world_news_api import fetch_news

PROMPT_VERSION = "v1"
ARTIFACTS_BASE_DIR = Path("artifacts") / "runs"


def _safe_process_article(article: dict[str, Any]) -> dict[str, Any]:
    text = article["text"]

    summary = summarize_article(text)
    topic = extract_topic(text)
    wiki_context = get_wikipedia_summary(topic) or "No additional background information found."
    enriched_summary = generate_enriched_summary(text, summary, wiki_context)

    return {
        "title": article["title"],
        "url": article["url"],
        "source_name": article["source_name"],
        "published_at": article["published_at"],
        "text_length": article["text_length"],
        "topic": topic,
        "summary": summary,
        "wiki_context": wiki_context,
        "enriched_summary": enriched_summary,
    }


def run_pipeline(max_articles: int = 5) -> dict[str, Any]:
    """
    Full run-based pipeline:
    1. create run_id
    2. fetch raw news
    3. save raw input artifact
    4. preprocess
    5. save processed articles
    6. summarize + enrich all processed articles
    7. save outputs
    8. save metadata and metrics
    """
    run_id = create_run_id()
    run_dir = ensure_dir(ARTIFACTS_BASE_DIR / run_id)
    log_path = run_dir / "logs.txt"

    started_at = utc_now_iso()
    append_text_line(f"[{started_at}] Pipeline started. run_id={run_id}", log_path)

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "model_name": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "max_articles_requested": max_articles,
        "num_raw_articles": 0,
        "num_processed_articles": 0,
        "num_successful_summaries": 0,
        "num_failed_articles": 0,
        "wiki_miss_count": 0,
        "artifacts": {
            "raw_news": str(run_dir / "raw_news.json"),
            "processed_articles": str(run_dir / "processed_articles.json"),
            "summaries": str(run_dir / "summaries.json"),
            "failed_articles": str(run_dir / "failed_articles.json"),
            "metadata": str(run_dir / "metadata.json"),
            "logs": str(log_path),
        },
    }

    try:
        raw_data = fetch_news()

        if not isinstance(raw_data, dict):
            raise ValueError(f"Unexpected API response from fetch_news(): {raw_data}")

        save_json(raw_data, run_dir / "raw_news.json")
        append_text_line(f"[{utc_now_iso()}] Saved raw_news.json", log_path)

        processed_articles = preprocess_articles(raw_data, max_articles=max_articles)
        save_json(processed_articles, run_dir / "processed_articles.json")
        append_text_line(
            f"[{utc_now_iso()}] Preprocessed {len(processed_articles)} articles",
            log_path,
        )

        raw_groups = raw_data.get("top_news", [])
        raw_count = 0
        for group in raw_groups:
            raw_count += len(group.get("news", []))

        metadata["num_raw_articles"] = raw_count
        metadata["num_processed_articles"] = len(processed_articles)

        summaries: list[dict[str, Any]] = []
        failed_articles: list[dict[str, Any]] = []

        for idx, article in enumerate(processed_articles, start=1):
            try:
                append_text_line(
                    f"[{utc_now_iso()}] Processing article {idx}/{len(processed_articles)}: {article['title']}",
                    log_path,
                )

                result = _safe_process_article(article)
                summaries.append(result)

                if result["wiki_context"] == "No additional background information found.":
                    metadata["wiki_miss_count"] += 1

                append_text_line(
                    f"[{utc_now_iso()}] Success: {article['title']}",
                    log_path,
                )

            except Exception as article_error:
                failed_articles.append(
                    {
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "error": str(article_error),
                        "traceback": traceback.format_exc(),
                    }
                )
                append_text_line(
                    f"[{utc_now_iso()}] Failed: {article.get('title')} | {article_error}",
                    log_path,
                )

        save_json(summaries, run_dir / "summaries.json")
        save_json(failed_articles, run_dir / "failed_articles.json")

        metadata["num_successful_summaries"] = len(summaries)
        metadata["num_failed_articles"] = len(failed_articles)
        metadata["status"] = "success"
        metadata["finished_at"] = utc_now_iso()

        save_json(metadata, run_dir / "metadata.json")

        latest_payload = {
            "run_id": run_id,
            "metadata": metadata,
            "summaries": summaries,
        }
        save_json(latest_payload, Path("data") / "output.json")
        save_text(run_id, Path("artifacts") / "latest_run.txt")

        append_text_line(f"[{utc_now_iso()}] Pipeline finished successfully", log_path)

        return latest_payload

    except Exception as e:
        metadata["status"] = "failed"
        metadata["finished_at"] = utc_now_iso()
        metadata["error"] = str(e)

        save_json(metadata, run_dir / "metadata.json")
        append_text_line(f"[{utc_now_iso()}] Pipeline failed: {e}", log_path)
        append_text_line(traceback.format_exc(), log_path)

        raise