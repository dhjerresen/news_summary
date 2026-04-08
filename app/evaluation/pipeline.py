# app/evaluation/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.utils import create_run_id, ensure_dir, load_json, save_json, utc_now_iso
from app.evaluation.compare import aggregate_judge_results
from app.evaluation.generate_candidates import generate_candidate_summaries
from app.evaluation.judge import judge_candidates


def build_evaluation_artifact_dir(base_dir: str | Path = "artifacts/evaluation") -> Path:
    run_id = create_run_id()
    artifact_dir = Path(base_dir) / run_id
    ensure_dir(artifact_dir)
    return artifact_dir


def build_evaluation_metadata(
    run_id: str,
    input_path: str,
    num_clusters: int,
    generation_models: list[dict[str, Any]],
    judge_model_name: str,
    judge_prompt_version: str,
    judge_results: list[dict[str, Any]],
) -> dict[str, Any]:
    num_successful_judgments = sum(1 for item in judge_results if item.get("success"))
    num_failed_judgments = len(judge_results) - num_successful_judgments

    return {
        "run_id": run_id,
        "pipeline_type": "evaluation",
        "timestamp_utc": utc_now_iso(),
        "input_path": input_path,
        "num_clusters": num_clusters,
        "generation_models": generation_models,
        "judge_model_name": judge_model_name,
        "judge_prompt_version": judge_prompt_version,
        "num_successful_judgments": num_successful_judgments,
        "num_failed_judgments": num_failed_judgments,
    }


def run_evaluation_pipeline(
    input_path: str | Path = "data/eval/fixed_eval_clusters.json",
    groq_api_key: str | None = None,
    artifact_base_dir: str | Path = "artifacts/evaluation",
    generation_models: list[dict[str, Any]] | None = None,
    judge_model_name: str = "llama-3.3-70b-versatile",
    judge_temperature: float = 0.0,
    judge_max_output_tokens: int = 600,
    judge_prompt_version: str = "judge_prompt_v1",
) -> dict[str, Any]:
    if generation_models is None:
        generation_models = [
            {
                "model_name": "llama-3.3-70b-versatile",
                "temperature": 0.2,
                "max_output_tokens": 400,
                "prompt_version": "summary_prompt_v1",
            },
            {
                "model_name": "llama-3.1-8b-instant",
                "temperature": 0.2,
                "max_output_tokens": 400,
                "prompt_version": "summary_prompt_v1",
            },
        ]

    artifact_dir = build_evaluation_artifact_dir(artifact_base_dir)
    run_id = artifact_dir.name

    clusters = load_json(input_path)
    if not isinstance(clusters, list):
        raise ValueError("Evaluation dataset must be a JSON list of clusters.")

    candidate_summaries = generate_candidate_summaries(
        clusters=clusters,
        groq_api_key=groq_api_key,
        generation_models=generation_models,
    )

    judge_results = judge_candidates(
        candidate_records=candidate_summaries,
        groq_api_key=groq_api_key,
        judge_model_name=judge_model_name,
        temperature=judge_temperature,
        max_output_tokens=judge_max_output_tokens,
        judge_prompt_version=judge_prompt_version,
    )

    successful_judge_results = [item for item in judge_results if item.get("success")]
    aggregated_results = aggregate_judge_results(successful_judge_results)

    metadata = build_evaluation_metadata(
        run_id=run_id,
        input_path=str(input_path),
        num_clusters=len(clusters),
        generation_models=generation_models,
        judge_model_name=judge_model_name,
        judge_prompt_version=judge_prompt_version,
        judge_results=judge_results,
    )

    save_json(clusters, artifact_dir / "input_clusters.json")
    save_json(candidate_summaries, artifact_dir / "candidate_summaries.json")
    save_json(judge_results, artifact_dir / "judge_results.json")
    save_json(aggregated_results, artifact_dir / "aggregated_results.json")
    save_json(metadata, artifact_dir / "metadata.json")

    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "input_clusters_path": str(artifact_dir / "input_clusters.json"),
        "candidate_summaries_path": str(artifact_dir / "candidate_summaries.json"),
        "judge_results_path": str(artifact_dir / "judge_results.json"),
        "aggregated_results_path": str(artifact_dir / "aggregated_results.json"),
        "metadata_path": str(artifact_dir / "metadata.json"),
        "metadata": metadata,
        "aggregated_results": aggregated_results,
    }