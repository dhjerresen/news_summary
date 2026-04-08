# app/production/run_production

from __future__ import annotations

import os

from app.production.pipeline import run_production_pipeline


if __name__ == "__main__":
    worldnews_api_key = os.getenv("WORLDNEWS_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    input_path = os.getenv("INPUT_PATH")

    result = run_production_pipeline(
        world_news_api_key=worldnews_api_key,
        groq_api_key=groq_api_key,
        input_path=input_path,
        max_clusters=10,
        min_text_length=80,
        source_country="us",
        language="en",
        model_name="gpt-4.1-mini",
        temperature=0.2,
        max_output_tokens=400,
        prompt_version="summary_prompt_v1",
    )

    print(f"Production pipeline completed: {result['artifact_dir']}")