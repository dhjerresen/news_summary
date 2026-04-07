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
from app.preprocessing import preprocess_clusters
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

PROMPT_VERSION = "v2_clustered_top_news"
ARTIFACTS_BASE_DIR = Path("artifacts") / "runs"
WIKI_FALLBACK_TEXT = "No additional background information found."


def _safe_process_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    """
    Create summary artifacts for one top-news cluster.
    Uses the representative article text as the main input.
    """
    text = cluster["text"]

    summary = summarize_article(text)
    topic = extract_topic(text)
    wiki_context = get_wikipedia_summary(topic) or WIKI_FALLBACK_TEXT
    enriched_summary = generate_enriched_summary(text, summary, wiki_context)

    return {
        "cluster_rank": cluster["cluster_rank"],
        "cluster_size": cluster["cluster_size"],
        "title": cluster["title"],
        "url": cluster["url"],
        "source_name": cluster["source_name"],
        "published_at": cluster["published_at"],
        "text_length": cluster["text_length"],
        "supporting_sources": cluster["supporting_sources"],
        "supporting_articles": cluster["supporting_articles"],
        "topic": topic,
        "summary": summary,
        "wiki_context": wiki_context,
        "enriched_summary": enriched_summary,
    }


def _count_raw_articles(raw_data: dict[str, Any]) -> int:
    groups = raw_data.get("top_news", [])
    raw_count = 0

    for group in groups:
        news_items = group.get("news", [])
        if isinstance(news_items, list):
            raw_count += len(news_items)

    return raw_count


def _validate_pipeline_counts(
    processed_clusters: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    failed_clusters: list[dict[str, Any]],
) -> None:
    expected_successes = len(processed_clusters) - len(failed_clusters)

    if len(summaries) != expected_successes:
        raise ValueError(
            "Inconsistent pipeline counts: "
            f"processed_clusters={len(processed_clusters)}, "
            f"failed_clusters={len(failed_clusters)}, "
            f"expected_successes={expected_successes}, "
            f"actual_summaries={len(summaries)}"
        )


def run_pipeline(max_clusters: int = 5) -> dict[str, Any]:
    """
    Full cluster-based pipeline:
    1. create run_id
    2. fetch raw top-news clusters
    3. save raw input artifact
    4. preprocess into one representative item per cluster
    5. save processed clusters
    6. summarize + enrich all processed clusters
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
        "max_clusters_requested": max_clusters,
        "num_raw_clusters": 0,
        "num_raw_articles": 0,
        "num_processed_clusters": 0,
        "num_successful_summaries": 0,
        "num_failed_clusters": 0,
        "wiki_miss_count": 0,
        "artifacts": {
            "raw_news": str(run_dir / "raw_news.json"),
            "processed_clusters": str(run_dir / "processed_clusters.json"),
            "summaries": str(run_dir / "summaries.json"),
            "failed_clusters": str(run_dir / "failed_clusters.json"),
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

        raw_groups = raw_data.get("top_news", [])
        if not isinstance(raw_groups, list):
            raise ValueError("Expected raw_data['top_news'] to be a list")

        metadata["num_raw_clusters"] = len(raw_groups)
        metadata["num_raw_articles"] = _count_raw_articles(raw_data)

        processed_clusters = preprocess_clusters(
            raw_data,
            max_clusters=max_clusters,
            min_text_length=80,
        )

        save_json(processed_clusters, run_dir / "processed_clusters.json")
        append_text_line(
            f"[{utc_now_iso()}] Preprocessed {len(processed_clusters)} clusters",
            log_path,
        )

        metadata["num_processed_clusters"] = len(processed_clusters)

        summaries: list[dict[str, Any]] = []
        failed_clusters: list[dict[str, Any]] = []

        for idx, cluster in enumerate(processed_clusters, start=1):
            try:
                append_text_line(
                    (
                        f"[{utc_now_iso()}] Processing cluster "
                        f"{idx}/{len(processed_clusters)}: "
                        f"rank={cluster['cluster_rank']} | "
                        f"size={cluster['cluster_size']} | "
                        f"title={cluster['title']}"
                    ),
                    log_path,
                )

                result = _safe_process_cluster(cluster)
                summaries.append(result)

                if result["wiki_context"] == WIKI_FALLBACK_TEXT:
                    metadata["wiki_miss_count"] += 1

                append_text_line(
                    f"[{utc_now_iso()}] Success: {cluster['title']}",
                    log_path,
                )

            except Exception as cluster_error:
                failed_clusters.append(
                    {
                        "cluster_rank": cluster.get("cluster_rank"),
                        "cluster_size": cluster.get("cluster_size"),
                        "title": cluster.get("title"),
                        "url": cluster.get("url"),
                        "error": str(cluster_error),
                        "traceback": traceback.format_exc(),
                    }
                )
                append_text_line(
                    f"[{utc_now_iso()}] Failed: {cluster.get('title')} | {cluster_error}",
                    log_path,
                )

        _validate_pipeline_counts(processed_clusters, summaries, failed_clusters)

        save_json(summaries, run_dir / "summaries.json")
        save_json(failed_clusters, run_dir / "failed_clusters.json")

        metadata["num_successful_summaries"] = len(summaries)
        metadata["num_failed_clusters"] = len(failed_clusters)
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