# app/utils/wandb_logger.py
import os
import wandb
from statistics import mean


def init_wandb_run(metadata: dict):
    return wandb.init(
        project="news-summary-mlops",
        config={
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


def log_aggregated_metrics(metadata: dict, items: list):
    latencies = [i["latency_seconds"] for i in items if i.get("latency_seconds") is not None]
    input_tokens = [i["input_tokens"] for i in items if i.get("input_tokens") is not None]
    output_tokens = [i["output_tokens"] for i in items if i.get("output_tokens") is not None]
    total_tokens = [i["total_tokens"] for i in items if i.get("total_tokens") is not None]
    summary_lengths = [i["summary_length"] for i in items if i.get("summary_length") is not None]

    wandb.log({
        "raw_cluster_count": metadata.get("raw_cluster_count"),
        "processed_cluster_count": metadata.get("processed_cluster_count"),
        "num_successful_summaries": metadata.get("num_successful_summaries"),
        "num_failed_summaries": metadata.get("num_failed_summaries"),
        "avg_latency_seconds": mean(latencies) if latencies else 0,
        "avg_input_tokens": mean(input_tokens) if input_tokens else 0,
        "avg_output_tokens": mean(output_tokens) if output_tokens else 0,
        "avg_total_tokens": mean(total_tokens) if total_tokens else 0,
        "avg_summary_length": mean(summary_lengths) if summary_lengths else 0,
    })


def log_summary_table(items: list):
    table = wandb.Table(columns=[
        "cluster_id",
        "cluster_rank",
        "title",
        "source_name",
        "published_at",
        "num_supporting_sources",
        "num_supporting_articles",
        "summary_length",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "success",
        "summary",
    ])

    for item in items:
        table.add_data(
            item.get("cluster_id"),
            item.get("cluster_rank"),
            item.get("title"),
            item.get("source_name"),
            item.get("published_at"),
            item.get("num_supporting_sources"),
            item.get("num_supporting_articles"),
            item.get("summary_length"),
            item.get("latency_seconds"),
            item.get("input_tokens"),
            item.get("output_tokens"),
            item.get("total_tokens"),
            item.get("success"),
            item.get("summary"),
        )

    wandb.log({"summary_table": table})


def log_artifacts(file_paths: dict, run_id: str):
    artifact = wandb.Artifact(name=f"run-{run_id}", type="pipeline-run")

    for name, path in file_paths.items():
        if path and os.path.exists(path):
            artifact.add_file(path, name=f"{name}.json")

    wandb.log_artifact(artifact)


def finish_run():
    wandb.finish()