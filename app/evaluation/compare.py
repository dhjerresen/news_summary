# app/evaluation/compare.py
from __future__ import annotations

from collections import defaultdict
from typing import Any


JudgeResult = dict[str, Any]


def aggregate_judge_results(judge_results: list[JudgeResult]) -> dict[str, Any]:
    """
    Aggregate pairwise LLM-as-a-judge results into per-model leaderboard metrics.
    """
    model_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "pairwise_wins": 0,
            "pairwise_losses": 0,
            "pairwise_ties": 0,
            "judged_pairs": 0,
            "total_score_sum": 0.0,
            "score_count": 0,
        }
    )

    for item in judge_results:
        parsed = item.get("parsed_result", {})
        model_a = parsed.get("model_a")
        model_b = parsed.get("model_b")
        winner = parsed.get("winner")

        scores = parsed.get("scores", {})
        score_a = scores.get("summary_a", {}).get("total")
        score_b = scores.get("summary_b", {}).get("total")

        if model_a:
            model_totals[model_a]["judged_pairs"] += 1
        if model_b:
            model_totals[model_b]["judged_pairs"] += 1

        if isinstance(score_a, (int, float)) and model_a:
            model_totals[model_a]["total_score_sum"] += score_a
            model_totals[model_a]["score_count"] += 1

        if isinstance(score_b, (int, float)) and model_b:
            model_totals[model_b]["total_score_sum"] += score_b
            model_totals[model_b]["score_count"] += 1

        if winner == "summary_a" and model_a and model_b:
            model_totals[model_a]["pairwise_wins"] += 1
            model_totals[model_b]["pairwise_losses"] += 1
        elif winner == "summary_b" and model_a and model_b:
            model_totals[model_b]["pairwise_wins"] += 1
            model_totals[model_a]["pairwise_losses"] += 1
        else:
            if model_a:
                model_totals[model_a]["pairwise_ties"] += 1
            if model_b:
                model_totals[model_b]["pairwise_ties"] += 1

    models: list[dict[str, Any]] = []
    for model_name, stats in model_totals.items():
        score_count = stats["score_count"]
        avg_total_score = (
            round(stats["total_score_sum"] / score_count, 3) if score_count else None
        )

        models.append(
            {
                "model_name": model_name,
                "pairwise_wins": int(stats["pairwise_wins"]),
                "pairwise_losses": int(stats["pairwise_losses"]),
                "pairwise_ties": int(stats["pairwise_ties"]),
                "judged_pairs": int(stats["judged_pairs"]),
                "avg_total_score": avg_total_score,
            }
        )

    models.sort(
        key=lambda x: (
            x["pairwise_wins"],
            x["avg_total_score"] if x["avg_total_score"] is not None else -1,
        ),
        reverse=True,
    )

    winner_model = models[0]["model_name"] if models else None

    return {
        "models": models,
        "winner_model": winner_model,
        "num_judgments": len(judge_results),
    }