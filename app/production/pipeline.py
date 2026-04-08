# app/production/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.cluster import preprocess_clusters
from app.core.ingest import (
    IngestError,
    fetch_world_news_top_news,
    load_raw_json,
    validate_top_news_payload,
)
from app.core.utils import create_run_id, ensure_dir, save_json, utc_now_iso
from app.production.export import build_frontend_payload
from app.production.summarize import summarize_clusters


def build_production_artifact_dir(base_dir: str | Path = "artifacts/production") -> Path:
    run_id = create_run_id()
    artifact_dir = Path(base_dir) / run_id
    ensure_dir(artifact_dir)
    return artifact_dir


def build_production_metadata(
    run_id: str,
    source: str,
    raw_cluster_count: int,
    processed_cluster_count: int,
    max_clusters: int,
    min_text_length: int,
    model_name: str,
    prompt_version: str,
    temperature: float,
    max_output_tokens: int,
    summaries: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    num_successful_summaries = sum(1 for item in summaries if item.get("success"))
    num_failed_summaries = len(summaries) - num_successful_summaries

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "pipeline_type": "production",
        "timestamp_utc": utc_now_iso(),
        "source": source,
        "raw_cluster_count": raw_cluster_count,
        "processed_cluster_count": processed_cluster_count,
        "max_clusters": max_clusters,
        "min_text_length": min_text_length,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "num_successful_summaries": num_successful_summaries,
        "num_failed_summaries": num_failed_summaries,
    }

    if extra:
        metadata.update(extra)

    return metadata


def run_production_pipeline(
    worldnews_api_key: str | None = None,
    groq_api_key: str | None = None,
    input_path: str | Path | None = None,
    artifact_base_dir: str | Path = "artifacts/production",
    max_clusters: int = 10,
    min_text_length: int = 80,
    source_country: str = "us",
    language: str = "en",
    date: str | None = None,
    news_sources: str | None = None,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_output_tokens: int = 400,
    prompt_version: str = "summary_prompt_v1",
    save_intermediate_artifacts: bool = False,
) -> dict[str, Any]:
    """
    Run the production pipeline:

    1. Ingest raw data
    2. Preprocess clusters
    3. Generate summaries
    4. Build frontend payload
    5. Save artifacts
    """
    artifact_dir = build_production_artifact_dir(artifact_base_dir)
    run_id = artifact_dir.name

    if input_path is not None:
        raw_data = load_raw_json(input_path)
        source = "file"
        source_details: dict[str, Any] = {"input_path": str(input_path)}
    else:
        if not worldnews_api_key:
            raise ValueError("worldnews_api_key is required when input_path is not provided.")

        try:
            raw_data = fetch_world_news_top_news(
                api_key=worldnews_api_key,
                source_country=source_country,
                language=language,
                date=date,
                news_sources=news_sources,
            )
        except IngestError:
            raise

        source = "world_news_api"
        source_details = {
            "source_country": source_country,
            "language": language,
            "date": date,
            "news_sources": news_sources,
        }

    if not validate_top_news_payload(raw_data):
        raise ValueError("Invalid top-news payload: missing or invalid 'top_news' field.")

    raw_cluster_count = len(raw_data.get("top_news", []))

    clusters = preprocess_clusters(
        raw_data=raw_data,
        max_clusters=max_clusters,
        min_text_length=min_text_length,
    )

    if not groq_api_key:
        raise ValueError("groq_api_key is required for summarization.")

    summaries = summarize_clusters(
        clusters=clusters,
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        prompt_version=prompt_version,
    )

    metadata = build_production_metadata(
        run_id=run_id,
        source=source,
        raw_cluster_count=raw_cluster_count,
        processed_cluster_count=len(clusters),
        max_clusters=max_clusters,
        min_text_length=min_text_length,
        model_name=model_name,
        prompt_version=prompt_version,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        summaries=summaries,
        extra=source_details,
    )

    frontend_payload = build_frontend_payload(
        clusters=clusters,
        summaries=summaries,
        metadata=metadata,
    )

    if save_intermediate_artifacts:
        save_json(raw_data, artifact_dir / "raw_data.json")
        save_json(clusters, artifact_dir / "processed_clusters.json")
        save_json(summaries, artifact_dir / "summaries.json")

    save_json(frontend_payload, artifact_dir / "frontend_payload.json")
    save_json(metadata, artifact_dir / "metadata.json")

    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "frontend_payload_path": str(artifact_dir / "frontend_payload.json"),
        "metadata_path": str(artifact_dir / "metadata.json"),
        "metadata": metadata,
        "processed_clusters": clusters,
        "summaries": summaries,
        "frontend_payload": frontend_payload,
    }


run_pipeline = run_production_pipeline