# app/utils/wandb_logger.py
import os
import wandb
from statistics import mean


# Initializes a Weights & Biases (W&B) run for the main pipeline
def init_wandb_run(metadata: dict):
    return wandb.init(
        project="news-summary-mlops",
        config={
            # Store metadata as configuration so the run can be reproduced and analyzed later
            "run_id": metadata.get("run_id"),
            "pipeline_type": metadata.get("pipeline_type"),
            "source": metadata.get("source"),
            "model_name": metadata.get("model_name"),
            "prompt_version": metadata.get("prompt_version"),
            "temperature": metadata.get("temperature"),
            "max_output_tokens": metadata.get("max_output_tokens"),
            "max_clusters": metadata.get("max_clusters"),
            "min_text_length": metadata.get("min_text_length"),
            "source_country": metadata.get("source_country"),
            "language": metadata.get("language"),
        },
    )


# Logs aggregated performance metrics from the pipeline run
def log_aggregated_metrics(metadata: dict, items: list):
    # Collect latency values (ignore missing values)
    latencies = [i["latency_seconds"] for i in items if i.get("latency_seconds") is not None]
    
    # Collect token usage values (ignore missing values)
    total_tokens = [i["total_tokens"] for i in items if i.get("total_tokens") is not None]

    # Log key metrics to W&B
    wandb.log({
        "raw_cluster_count": metadata.get("raw_cluster_count"),
        "processed_cluster_count": metadata.get("processed_cluster_count"),
        "num_successful_summaries": metadata.get("num_successful_summaries"),
        "num_failed_summaries": metadata.get("num_failed_summaries"),
        "avg_latency_seconds": mean(latencies) if latencies else 0,
        "avg_total_tokens": mean(total_tokens) if total_tokens else 0,
    })


# Logs individual summary results as a structured table in W&B
def log_summary_table(items: list):
    # Define table columns for each summary result
    table = wandb.Table(columns=[
        "cluster_id",
        "cluster_rank",
        "title",
        "success",
        "latency_seconds",
        "total_tokens",
        "summary",
    ])

    # Add one row per processed item
    for item in items:
        table.add_data(
            item.get("cluster_id"),
            item.get("cluster_rank"),
            item.get("title"),
            item.get("success"),
            item.get("latency_seconds"),
            item.get("total_tokens"),
            item.get("summary"),
        )

    # Log the table to W&B
    wandb.log({"summary_table": table})


# Logs output files (artifacts) from the pipeline run to W&B
def log_artifacts(file_paths: dict, run_id: str):
    # Create a new artifact to group files from this run
    artifact = wandb.Artifact(name=f"run-{run_id}", type="pipeline-run")

    # Add each file if it exists
    for name, path in file_paths.items():
        if path and os.path.exists(path):
            artifact.add_file(path, name=f"{name}.json")

    # Upload the artifact to W&B
    wandb.log_artifact(artifact)


# Ends the current W&B run
def finish_run():
    wandb.finish()