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


if __name__ == "__main__":
    try:
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY")
        input_path = os.getenv("EVAL_INPUT_PATH", "data/eval/fixed_eval_clusters.json")

        result = run_evaluation_pipeline(
            input_path=input_path,
            groq_api_key=groq_api_key,
        )

        metadata = result["metadata"]
        judge_results = result["judge_results"]
        aggregated_results = result["aggregated_results"]

        print(f"Evaluation pipeline completed: {result['artifact_dir']}")
        print(f"Winner model: {aggregated_results['winner_model']}")

        try:
            init_wandb_eval_run(metadata)
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
            finish_eval_run()
            print("W&B evaluation logging completed.")
        except Exception as wandb_error:
            print(f"W&B evaluation logging failed: {wandb_error}")

        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(json.dumps(aggregated_results, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Evaluation pipeline failed: {e}")
        sys.exit(1)