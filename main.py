from __future__ import annotations

import json
import os
import shutil
import sys

from dotenv import load_dotenv

from app.production.pipeline import run_pipeline
from app.utils.utils import ensure_dir
from app.utils.wandb_logger import (
    finish_run,
    init_wandb_run,
    log_aggregated_metrics,
    log_artifacts,
    log_summary_table,
)


if __name__ == "__main__":
    try:
        load_dotenv()

        worldnews_api_key = os.getenv("WORLDNEWS_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        input_path = os.getenv("INPUT_PATH")

        result = run_pipeline(
            worldnews_api_key=worldnews_api_key,
            groq_api_key=groq_api_key,
            input_path=input_path,
            max_clusters=5,
        )

        metadata = result["metadata"]
        items = result["frontend_payload"]["items"]

        print("Pipeline completed successfully.")
        print(f"Run ID: {result['run_id']}")
        print(f"Successful summaries: {metadata['num_successful_summaries']}")
        print(f"Failed summaries: {metadata['num_failed_summaries']}")

        latest_output_path = result["frontend_payload_path"]
        ensure_dir("docs")
        shutil.copyfile(latest_output_path, "docs/latest.json")
        print(f"Copied {latest_output_path} to docs/latest.json")

        wandb_started = False
        try:
            init_wandb_run(metadata)
            wandb_started = True

            log_aggregated_metrics(metadata, items)
            log_summary_table(items)
            log_artifacts(
                file_paths={
                    "raw_news": result.get("raw_data_path"),
                    "metadata": result.get("metadata_path"),
                    "summaries": result.get("summaries_path"),
                    "frontend_payload": result.get("frontend_payload_path"),
                },
                run_id=metadata["run_id"],
            )
            print("W&B logging completed.")
        except Exception as wandb_error:
            print(f"W&B logging failed: {wandb_error}")
        finally:
            if wandb_started:
                finish_run()

        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)