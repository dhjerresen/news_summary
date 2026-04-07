# main.py
from __future__ import annotations

import json
import shutil

from app.pipeline import run_pipeline


if __name__ == "__main__":
    try:
        result = run_pipeline(max_clusters=5)

        print("Pipeline completed successfully.")
        print(f"Run ID: {result['run_id']}")
        print(f"Successful summaries: {result['metadata']['num_successful_summaries']}")
        print(f"Failed clusters: {result['metadata']['num_failed_clusters']}")
        print("Latest output saved to data/output.json")
        shutil.copyfile("data/output.json", "docs/latest.json")
        print("Copied data/output.json to docs/latest.json")

        print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Pipeline failed: {e}")