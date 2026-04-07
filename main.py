from app.pipeline import run_pipeline
import json


if __name__ == "__main__":
    results = run_pipeline()

    with open("data/output.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Pipeline completed!")