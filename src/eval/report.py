"""Report macro averages over the same eligible questions for both systems."""

from itertools import pairwise
from statistics import fmean, median

from src.eval.ragas_metrics import METRICS


def compare(rows: list[dict]) -> dict:
    systems = {
        system: {row["question"]: row for row in rows if row["system"] == system}
        for system in ("baseline", "corrective")
    }
    summary = {}
    for metric in (*METRICS, "answer_coverage"):
        pairs = [
            (left["metrics"][metric], systems["corrective"][q]["metrics"][metric])
            for q, left in systems["baseline"].items()
            if q in systems["corrective"]
        ]
        valid = [(a, b) for a, b in pairs if a is not None and b is not None]
        left = fmean(a for a, _ in valid) if valid else None
        right = fmean(b for _, b in valid) if valid else None
        summary[metric] = {
            "baseline": left,
            "corrective": right,
            "delta": right - left if valid else None,
            "n_pairs": len(valid),
        }
    return summary


def outcome(answerable: bool, abstained: bool) -> str:
    if answerable:
        return "false_abstention" if abstained else "answered"
    return "correct_abstention" if abstained else "hallucination"


def rewrite_causes(steps: list[dict]) -> list[str]:
    """Why each rewrite happened, read from a case's graph steps."""
    causes = []
    for step, following in pairwise(steps):
        if step["node"] == "grade_documents" and following["node"] == "rewrite_query":
            if not step.get("grading_valid", True):
                causes.append("invalid_grading")
            elif not step["evidence"]:
                causes.append("zero_relevant")
            else:
                causes.append("below_threshold")
    return causes


def breakdown(rows: list[dict]) -> dict:
    """Abstention outcomes per system, and rewrite causes for the corrective one."""
    result = {}
    for system in ("baseline", "corrective"):
        outcomes = [r["outcome"] for r in rows if r["system"] == system and r.get("outcome")]
        unanswerable = sum(o in ("correct_abstention", "hallucination") for o in outcomes)
        answerable = len(outcomes) - unanswerable
        result[system] = {
            "unanswerable": unanswerable,
            "correct_abstention": outcomes.count("correct_abstention"),
            "hallucination": outcomes.count("hallucination"),
            "answerable": answerable,
            "false_abstention": outcomes.count("false_abstention"),
        }
    causes = [c for r in rows if r["system"] == "corrective" for c in r.get("rewrite_causes", [])]
    result["rewrite_causes"] = {c: causes.count(c) for c in sorted(set(causes))}
    return result


def breakdown_table(info: dict) -> str:
    def rate(count: int, total: int) -> str:
        return f"{count}/{total} ({count / total:.0%})" if total else "—"

    lines = [
        "| Outcome | Baseline RAG | Agentic Corrective RAG |",
        "| --- | ---: | ---: |",
    ]
    for label, key, total in (
        ("Correct abstention (unanswerable)", "correct_abstention", "unanswerable"),
        ("Hallucination (unanswerable)", "hallucination", "unanswerable"),
        ("False abstention (answerable)", "false_abstention", "answerable"),
    ):
        cells = [rate(info[s][key], info[s][total]) for s in ("baseline", "corrective")]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    causes = ", ".join(f"{k}: {v}" for k, v in info["rewrite_causes"].items()) or "none"
    return "\n".join([*lines, "", f"Corrective rewrite causes: {causes}"])


def variants_table(reports: list[dict]) -> str:
    """One row per run: the first run's baseline, then every run's corrective system."""

    def system_row(label: str, report: dict, system: str) -> str:
        rows = [r for r in report["cases"] if r["system"] == system]
        rewrites = sum(bool(r["answer"] and r["answer"]["rewrites"]) for r in rows)
        latency = median(r["latency_s"] for r in rows if r["latency_s"] is not None)
        cells = []
        for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            value = report["summary"][metric][system]
            cells.append("—" if value is None else f"{value:.4f}")
        info = report["breakdown"][system]
        unanswerable = info["unanswerable"]
        abstain = f"{info['correct_abstention']}/{unanswerable}" if unanswerable else "—"
        return " | ".join(
            [f"| {label}", *cells, f"{rewrites}/{len(rows)}", f"{latency:.1f}", abstain + " |"]
        )

    header = (
        "| Variant | Faithfulness | Answer relevancy | Context precision | Context recall "
        "| Rewrites | Median latency (s) | Correct abstentions |"
    )
    lines = [header, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines.append(system_row("baseline", reports[0], "baseline"))
    lines += [system_row(r.get("label") or "corrective", r, "corrective") for r in reports]
    return "\n".join(lines)


def table(summary: dict) -> str:
    header = "| Metric | Baseline RAG | Agentic Corrective RAG | Delta | Pairs |"
    lines = [header, "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in summary.items():
        cells = [
            "—" if values[k] is None else f"{values[k]:.4f}"
            for k in ("baseline", "corrective", "delta")
        ]
        lines.append(" | ".join(["| " + name, *cells, str(values["n_pairs"]) + " |"]))
    return "\n".join(lines)
