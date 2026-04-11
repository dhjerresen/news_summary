# app/evaluation/run_evaluation.py
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

from app.evaluation.pipeline import run_evaluation_pipeline
from app.utils.wandb_eval_logger import (
    finish_eval_run,
    init_wandb_eval_run,
    log_eval_artifacts,
    log_eval_metrics,
    log_judge_results_table,
)


# Entry point: this code only runs when the file is executed directly
if __name__ == "__main__":
    try:
        # Load environment variables from the .env file
        load_dotenv()

        # Read API key and optional evaluation input path from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        input_path = os.getenv("EVAL_INPUT_PATH", "data/eval/fixed_eval_clusters.json")

        # Run the full evaluation pipeline
        result = run_evaluation_pipeline(
            input_path=input_path,
            groq_api_key=groq_api_key,
        )

        # Extract important outputs from the pipeline result
        metadata = result["metadata"]
        judge_results = result["judge_results"]
        aggregated_results = result["aggregated_results"]

        # Print basic information about the completed evaluation run
        print(f"Evaluation pipeline completed: {result['artifact_dir']}")
        print(f"Winner model: {aggregated_results['winner_model']}")

        try:
            # Start a Weights & Biases run for evaluation logging
            init_wandb_eval_run(metadata)

            # Log evaluation metrics, detailed judge table, and generated artifacts
            log_eval_metrics(metadata, judge_results, aggregated_results)
            log_judge_results_table(judge_results)
            log_eval_artifacts(
                file_paths={
                    "input_clusters": result.get("input_clusters_path"),
                    "candidate_summaries": result.get("candidate_summaries_path"),
                    "judge_results": result.get("judge_results_path"),
                    "aggregated_results": result.get("aggregated_results_path"),
                    "metadata": result.get("metadata_path"),
                },
                run_id=metadata["run_id"],
            )

            # Finish the W&B run cleanly
            finish_eval_run()
            print("W&B evaluation logging completed.")
        except Exception as wandb_error:
            # If W&B logging fails, print the error but keep the evaluation result
            print(f"W&B evaluation logging failed: {wandb_error}")

        # Print metadata and aggregated leaderboard results as formatted JSON
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(json.dumps(aggregated_results, indent=2, ensure_ascii=False))

    except Exception as e:
        # Catch any fatal error, print it, and exit with error status
        print(f"Evaluation pipeline failed: {e}")
        sys.exit(1)