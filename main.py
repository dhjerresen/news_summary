# main.py
from app.pipeline import run_pipeline
import json


if __name__ == "__main__":
    try:
        results = run_pipeline()

        with open("data/output.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Pipeline completed! Saved {len(results)} article(s).")

    except Exception as e:
        print(f"Pipeline failed: {e}")