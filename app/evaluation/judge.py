# app/evaluation/judge.py
from __future__ import annotations

import json
import os
import time
from itertools import combinations
from typing import Any

from groq import Groq


ClusterCandidateRecord = dict[str, Any]
JudgeResult = dict[str, Any]


def get_groq_client(api_key: str | None = None) -> Groq:
    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return Groq(api_key=resolved_api_key)


def build_judge_prompt(
    cluster_record: ClusterCandidateRecord,
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
) -> str:
    source_title = str(cluster_record.get("title", "") or "").strip()
    source_text = str(cluster_record.get("text", "") or "").strip()

    summary_a_text = str(summary_a.get("summary", "") or "").strip()
    summary_b_text = str(summary_b.get("summary", "") or "").strip()

    model_a = str(summary_a.get("model_name", "") or "").strip()
    model_b = str(summary_b.get("model_name", "") or "").strip()

    return f"""
You are an expert evaluator of news summaries.

You will receive:
1. A source news cluster
2. Summary A
3. Summary B

Evaluate both summaries on:
- factual_consistency
- coverage
- relevance
- clarity
- conciseness

Scoring scale: 1 to 5.
Prioritize factual consistency above style.
Be strict.

Return ONLY valid JSON in this exact format:
{{
  "model_a": "{model_a}",
  "model_b": "{model_b}",
  "scores": {{
    "summary_a": {{
      "factual_consistency": 0,
      "coverage": 0,
      "relevance": 0,
      "clarity": 0,
      "conciseness": 0,
      "total": 0
    }},
    "summary_b": {{
      "factual_consistency": 0,
      "coverage": 0,
      "relevance": 0,
      "clarity": 0,
      "conciseness": 0,
      "total": 0
    }}
  }},
  "winner": "summary_a",
  "short_reason": "..."
}}

Source title:
{source_title}

Source cluster text:
{source_text}

Summary A ({model_a}):
{summary_a_text}

Summary B ({model_b}):
{summary_b_text}
""".strip()


def safe_parse_judge_json(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    return json.loads(raw_text)


def judge_pair(
    client: Groq,
    cluster_record: ClusterCandidateRecord,
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    judge_model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_output_tokens: int = 600,
    judge_prompt_version: str = "judge_prompt_v1",
) -> JudgeResult:
    prompt = build_judge_prompt(cluster_record, summary_a, summary_b)
    started_at = time.perf_counter()

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=judge_model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    latency_seconds = round(time.perf_counter() - started_at, 3)
    raw_output = ""
    if response.choices and response.choices[0].message:
        raw_output = (response.choices[0].message.content or "").strip()

    parsed = safe_parse_judge_json(raw_output)

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return {
        "cluster_id": cluster_record.get("cluster_id"),
        "cluster_rank": cluster_record.get("cluster_rank"),
        "judge_model_name": judge_model_name,
        "judge_prompt_version": judge_prompt_version,
        "latency_seconds": latency_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw_judge_output": raw_output,
        "parsed_result": parsed,
        "success": True,
        "error": None,
    }


def judge_candidates(
    candidate_records: list[ClusterCandidateRecord],
    groq_api_key: str | None,
    judge_model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_output_tokens: int = 600,
    judge_prompt_version: str = "judge_prompt_v1",
) -> list[JudgeResult]:
    client = get_groq_client(api_key=groq_api_key)
    all_results: list[JudgeResult] = []

    for cluster_record in candidate_records:
        candidates = cluster_record.get("candidates", [])
        if len(candidates) < 2:
            continue

        for summary_a, summary_b in combinations(candidates, 2):
            try:
                result = judge_pair(
                    client=client,
                    cluster_record=cluster_record,
                    summary_a=summary_a,
                    summary_b=summary_b,
                    judge_model_name=judge_model_name,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    judge_prompt_version=judge_prompt_version,
                )
            except Exception as exc:
                result = {
                    "cluster_id": cluster_record.get("cluster_id"),
                    "cluster_rank": cluster_record.get("cluster_rank"),
                    "judge_model_name": judge_model_name,
                    "judge_prompt_version": judge_prompt_version,
                    "latency_seconds": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "raw_judge_output": "",
                    "parsed_result": {},
                    "success": False,
                    "error": str(exc),
                }

            all_results.append(result)

    return all_results