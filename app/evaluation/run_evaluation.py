# app/evaluation/run_evaluation.py
from __future__ import annotations

import os

from app.evaluation.pipeline import run_evaluation_pipeline


if __name__ == "__main__":
    groq_api_key = os.getenv("GROQ_API_KEY")
    input_path = os.getenv("EVAL_INPUT_PATH", "data/eval/fixed_eval_clusters.json")

    result = run_evaluation_pipeline(
        input_path=input_path,
        groq_api_key=groq_api_key,
    )

    print(f"Evaluation pipeline completed: {result['artifact_dir']}")
    print(f"Winner model: {result['aggregated_results']['winner_model']}")