# app/utils/wandb_eval_logger.py
from __future__ import annotations

import os
from statistics import mean
from typing import Any

import wandb


# Initializes a Weights & Biases (W&B) run specifically for evaluation experiments
def init_wandb_eval_run(
    metadata: dict[str, Any],
    project: str = "news-summary-evaluation",
    entity: str | None = None,
) -> Any:
    """Initialize a W&B run for the evaluation pipeline."""
    
    # Prepare configuration dictionary from metadata to track experiment settings
    config = {
        "run_id": metadata.get("run_id"),
        "pipeline_type": metadata.get("pipeline_type"),
        "input_path": metadata.get("input_path"),
        "num_clusters": metadata.get("num_clusters"),
        "judge_model_name": metadata.get("judge_model_name"),
        "judge_temperature": metadata.get("judge_temperature"),
        "judge_max_output_tokens": metadata.get("judge_max_output_tokens"),
        "judge_prompt_version": metadata.get("judge_prompt_version"),
        "generation_models": metadata.get("generation_models"),
    }

    # Start and return the W&B run with project details and configuration
    return wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=metadata.get("run_id"),
    )


# Logs aggregated evaluation metrics to W&B
def log_eval_metrics(
    metadata: dict[str, Any],
    judge_results: list[dict[str, Any]],
    aggregated_results: dict[str, Any],
) -> None:
    """Log simple aggregated evaluation metrics."""
    
    # Filter only successful judge results
    successful = [item for item in judge_results if item.get("success")]

    # Extract latency and token usage values, ignoring missing values
    latencies = [
        item.get("latency_seconds")
        for item in successful
        if item.get("latency_seconds") is not None
    ]
    input_tokens = [
        item.get("input_tokens")
        for item in successful
        if item.get("input_tokens") is not None
    ]
    output_tokens = [
        item.get("output_tokens")
        for item in successful
        if item.get("output_tokens") is not None
    ]
    total_tokens = [
        item.get("total_tokens")
        for item in successful
        if item.get("total_tokens") is not None
    ]

    # Get counts of successful and failed judgments
    num_successful = metadata.get("num_successful_judgments", 0) or 0
    num_failed = metadata.get("num_failed_judgments", 0) or 0
    total = num_successful + num_failed

    # Create dictionary of aggregated metrics
    metrics: dict[str, Any] = {
        "num_clusters": metadata.get("num_clusters"),
        "num_successful_judgments": num_successful,
        "num_failed_judgments": num_failed,
        "judge_success_rate": (num_successful / total) if total > 0 else 0,
        "num_judgments": aggregated_results.get("num_judgments", 0),
        "avg_judge_latency_seconds": mean(latencies) if latencies else 0,
        "avg_judge_input_tokens": mean(input_tokens) if input_tokens else 0,
        "avg_judge_output_tokens": mean(output_tokens) if output_tokens else 0,
        "avg_judge_total_tokens": mean(total_tokens) if total_tokens else 0,
    }

    # Add per-model leaderboard statistics
    for model_stats in aggregated_results.get("models", []):
        model_name = str(model_stats.get("model_name", "unknown_model"))
        
        # Sanitize model name for safe metric keys (no dots or dashes)
        safe_name = model_name.replace(".", "_").replace("-", "_")

        # Log pairwise comparison statistics
        metrics[f"leaderboard/{safe_name}/pairwise_wins"] = model_stats.get("pairwise_wins", 0)
        metrics[f"leaderboard/{safe_name}/pairwise_losses"] = model_stats.get("pairwise_losses", 0)
        metrics[f"leaderboard/{safe_name}/pairwise_ties"] = model_stats.get("pairwise_ties", 0)
        metrics[f"leaderboard/{safe_name}/judged_pairs"] = model_stats.get("judged_pairs", 0)

        # Log average score if available
        avg_score = model_stats.get("avg_total_score")
        if avg_score is not None:
            metrics[f"leaderboard/{safe_name}/avg_total_score"] = avg_score

    # Log overall winning model if one exists
    winner_model = aggregated_results.get("winner_model")
    if winner_model:
        metrics["winner_model_name"] = winner_model

    # Send all metrics to W&B
    wandb.log(metrics)


# Logs detailed judge results as a structured W&B table
def log_judge_results_table(judge_results: list[dict[str, Any]]) -> None:
    """Log a simple W&B table for judge results."""
    
    # Define table columns for structured evaluation data
    table = wandb.Table(
        columns=[
            "cluster_id",
            "cluster_rank",
            "judge_model_name",
            "judge_prompt_version",
            "model_a",
            "model_b",
            "winner",
            "summary_a_total",
            "summary_b_total",
            "short_reason",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "success",
            "error",
        ]
    )

    # Add one row per judge result
    for item in judge_results:
        parsed = item.get("parsed_result", {}) or {}
        scores = parsed.get("scores", {}) or {}

        summary_a = scores.get("summary_a", {}) or {}
        summary_b = scores.get("summary_b", {}) or {}

        table.add_data(
            item.get("cluster_id"),
            item.get("cluster_rank"),
            item.get("judge_model_name"),
            item.get("judge_prompt_version"),
            parsed.get("model_a"),
            parsed.get("model_b"),
            parsed.get("winner"),
            summary_a.get("total"),
            summary_b.get("total"),
            parsed.get("short_reason"),
            item.get("latency_seconds"),
            item.get("input_tokens"),
            item.get("output_tokens"),
            item.get("total_tokens"),
            item.get("success"),
            item.get("error"),
        )

    # Log the table to W&B
    wandb.log({"judge_results_table": table})


# Logs evaluation-related files (JSON outputs) as a W&B artifact
def log_eval_artifacts(file_paths: dict[str, str | None], run_id: str) -> None:
    """Log evaluation JSON files as a W&B artifact."""
    
    # Create a new artifact to group evaluation files
    artifact = wandb.Artifact(name=f"eval-run-{run_id}", type="evaluation-run")

    # Add each existing file to the artifact
    for name, path in file_paths.items():
        if path and os.path.exists(path):
            artifact.add_file(path, name=f"{name}.json")

    # Upload the artifact to W&B
    wandb.log_artifact(artifact)


# Ends the current W&B run
def finish_eval_run() -> None:
    wandb.finish()