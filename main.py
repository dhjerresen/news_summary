# main.py
from __future__ import annotations

import json
import os
import shutil

from app.core.utils import ensure_dir
from app.production.pipeline import run_pipeline


if __name__ == "__main__":
    try:
        worldnews_api_key = os.getenv("WORLDNEWS_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        input_path = os.getenv("INPUT_PATH")

        result = run_pipeline(
            worldnews_api_key=worldnews_api_key,
            groq_api_key=groq_api_key,
            input_path=input_path,
            max_clusters=5,
        )

        print("Pipeline completed successfully.")
        print(f"Run ID: {result['run_id']}")
        print(f"Successful summaries: {result['metadata']['num_successful_summaries']}")
        print(f"Failed clusters: {result['metadata']['num_failed_clusters']}")

        latest_output_path = result["frontend_payload_path"]
        ensure_dir("docs")
        shutil.copyfile(latest_output_path, "docs/latest.json")
        print(f"Copied {latest_output_path} to docs/latest.json")

        print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Pipeline failed: {e}")