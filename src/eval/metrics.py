from math import ceil
from statistics import fmean

from src.agents.graph import State


def score(state: State, gold: set[str]) -> dict[str, float | None]:
    answer = state["result"]
    retrieved = {p.source for p in state["candidates"]}
    cited = {p.source for p in answer.citations}
    return {
        "source_recall_at_k": len(gold & retrieved) / len(gold) if gold else None,
        "citation_precision": len(gold & cited) / len(cited) if cited else None,
        "abstention_accuracy": float(answer.abstained == (not gold)),
        "rewrite_rate": float(answer.rewrites > 0),
        "answer_coverage": float(not answer.abstained),
    }


def aggregate(rows: list[dict]) -> dict:
    summary = {}
    for key in rows[0]["metrics"]:
        values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        summary[key] = {"mean": fmean(values) if values else None, "n": len(values)}
    times = sorted(r["latency_s"] for r in rows)
    summary["latency_p95_s"] = times[ceil(0.95 * len(times)) - 1]
    return summary
