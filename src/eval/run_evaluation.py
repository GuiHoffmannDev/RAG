"""Compare a single-pass RAG with the corrective graph on one index snapshot."""

import argparse
import asyncio
import csv
import hashlib
import json
import os
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Self, TextIO

from langgraph.graph.state import CompiledStateGraph
from openai import AsyncOpenAI, OpenAIError
from pydantic import Field, StringConstraints, model_validator
from qdrant_client.http.exceptions import ApiException
from ragas.metrics.collections.base import BaseMetric

from src.agents.graph import PROMPT_VERSION, build_graph, generate
from src.agents.runtime import Runtime
from src.config import Settings
from src.eval.ragas_metrics import METRICS, make_metrics, score_response
from src.eval.report import (
    breakdown,
    breakdown_table,
    compare,
    outcome,
    rewrite_causes,
    table,
    variants_table,
)
from src.models import Query
from src.observability.tracing import QueryTrace, record_scores

TRACE_COLUMNS = ("trace_id", "trace_url", "total_tokens", "estimated_cost_usd")


class GoldenCase(Query):
    # An unanswerable question has no reference answer and is scored only on abstention.
    answerable: bool = True
    ground_truth: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    gold_sources: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_reference(self) -> Self:
        if self.answerable and not self.ground_truth:
            raise ValueError("An answerable case needs a ground_truth.")
        return self


def load_cases(raw: bytes) -> list[GoldenCase]:
    cases = []
    for number, line in enumerate(raw.splitlines(), 1):
        if line.strip():
            try:
                cases.append(GoldenCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid golden case on line {number}.") from exc
    if not cases or len({c.question.casefold() for c in cases}) != len(cases):
        raise ValueError("Provide a nonempty JSONL dataset with unique questions.")
    return cases


async def run_case(
    case: GoldenCase,
    system: str,
    graph: CompiledStateGraph,
    metrics: dict[str, BaseMetric],
    traces: TextIO,
    trace: QueryTrace,
) -> dict:
    steps: list[dict] = []
    errors: dict[str, str] = {}
    try:
        with trace.observe(case.question) as config:
            state = generate(graph, case.question, config, steps)
            trace.answer = answer = state["result"]
    except Exception as exc:  # noqa: BLE001 - one failed graph must not discard other cases.
        errors["generation"] = type(exc).__name__
        answer = None
    row = {
        **case.model_dump(),
        "system": system,
        "latency_s": trace.metrics.latency_s,
        "observability": trace.metrics.model_dump(),
        "answer": answer.model_dump(exclude={"observability"}) if answer else None,
        "errors": errors,
        "outcome": outcome(case.answerable, answer.abstained) if answer else None,
        "rewrite_causes": rewrite_causes(steps),
    }
    traces.write(json.dumps({**row, "steps": steps}, ensure_ascii=False, allow_nan=False) + "\n")
    traces.flush()
    scores: dict[str, float | None] = dict.fromkeys((*METRICS, "answer_coverage"))
    if answer is not None and case.answerable:
        judged, failures = await score_response(
            metrics, case.question, case.ground_truth, answer, [p.text for p in state["evidence"]]
        )
        scores.update(judged, answer_coverage=float(not answer.abstained))
        errors.update(failures)
    record_scores(trace.metrics.trace_id, scores)
    return {**row, "metrics": scores}


async def evaluate(
    dataset: Path,
    output_root: Path,
    cfg: Settings,
    provider: str,
    model: str,
    embedding: str,
    label: str = "",
) -> tuple[Path, dict, bool]:
    raw = dataset.read_bytes()
    cases = load_cases(raw)
    local = provider == "ollama"
    options = (
        {"base_url": str(cfg.ollama_url).rstrip("/") + "/v1", "api_key": "ollama"} if local else {}
    )
    rows = []
    with closing(Runtime(cfg)) as runtime:
        async with AsyncOpenAI(timeout=180 if local else 45, max_retries=2, **options) as client:
            metrics = make_metrics(client, model, embedding, local)
            aliases = runtime.index.client.get_aliases().aliases
            snapshot = next(
                (a.collection_name for a in aliases if a.alias_name == cfg.collection),
                cfg.collection,
            )
            runtime.index.client.get_collection(snapshot)
            runtime.index.settings = cfg.model_copy(update={"collection": snapshot})
            graphs = {
                "baseline": build_graph(runtime.index.search, runtime.model, 0, baseline=True),
                "corrective": runtime.graph,
            }
            out = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            out.mkdir(parents=True, exist_ok=False)
            print(f"Writing evaluation to {out}", flush=True)
            with (
                (out / "traces.jsonl").open("x", encoding="utf-8") as traces,
                (out / "results.csv").open("x", newline="", encoding="utf-8") as csv_file,
            ):
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "system",
                        "question",
                        "ground_truth",
                        "answer",
                        "latency_s",
                        *TRACE_COLUMNS,
                        *METRICS,
                        "answer_coverage",
                        "outcome",
                        "errors",
                    ],
                )
                writer.writeheader()
                for i, case in enumerate(cases, 1):
                    for system, graph in graphs.items():
                        trace = QueryTrace(
                            runtime.index.settings,
                            f"evaluation-{system}",
                            system=system,
                            prompt_version=PROMPT_VERSION,
                            evaluation_run=out.name,
                            max_rewrites=0 if system == "baseline" else cfg.max_rewrites,
                        )
                        row = await run_case(case, system, graph, metrics, traces, trace)
                        writer.writerow(
                            {
                                "system": system,
                                "question": case.question,
                                "ground_truth": case.ground_truth,
                                "answer": row["answer"]["answer"] if row["answer"] else None,
                                "latency_s": row["latency_s"],
                                **{key: row["observability"][key] for key in TRACE_COLUMNS},
                                **row["metrics"],
                                "outcome": row["outcome"],
                                "errors": json.dumps(row["errors"]),
                            }
                        )
                        csv_file.flush()
                        rows.append(row)
                        print(f"[{i}/{len(cases)}] {system}: {row['errors'] or 'ok'}", flush=True)

    summary = compare(rows)
    info = breakdown(rows)
    report = {
        "label": label,
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "case_count": len(cases),
        "snapshot": snapshot,
        "settings": cfg.model_dump(mode="json", exclude={"ollama_url", "qdrant_url"}),
        "judge_provider": provider,
        "judge_model": model,
        "judge_embedding_model": embedding,
        "ragas_version": version("ragas"),
        "summary": summary,
        "breakdown": info,
        "cases": rows,
    }
    (out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "summary.md").write_text(
        table(summary) + "\n\n" + breakdown_table(info) + "\n", encoding="utf-8"
    )
    return out, summary, any(r["errors"] for r in rows)


def main(argv: Sequence[str] | None = None) -> None:
    cfg = Settings.load()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("evals/golden_set.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"))
    parser.add_argument("--judge-provider", choices=("openai", "ollama"), default="openai")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
    parser.add_argument("--judge-embedding-model")
    parser.add_argument("--label", default="", help="Name of this variant in comparison tables.")
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="RUN_DIR",
        help="Print a variants table from finished run directories and exit.",
    )
    args = parser.parse_args(argv)
    if args.compare:
        reports = [json.loads((d / "results.json").read_text("utf-8")) for d in args.compare]
        print(variants_table(reports))
        return
    local = args.judge_provider == "ollama"
    if not local and not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY for the hosted judge or choose --judge-provider ollama.")
    model = args.judge_model or ("qwen3.5:4b" if local else "gpt-4o-mini")
    embedding = args.judge_embedding_model or (
        "qwen3-embedding:0.6b" if local else "text-embedding-3-small"
    )
    try:
        out, _, errors = asyncio.run(
            evaluate(
                args.cases, args.output_dir, cfg, args.judge_provider, model, embedding, args.label
            )
        )
    except ValueError as exc:
        parser.exit(2, f"Evaluation configuration or dataset error: {exc}\n")
    except (OSError, OpenAIError, ApiException) as exc:
        parser.exit(
            2,
            f"Evaluation setup failed ({type(exc).__name__}). Check dataset, index and model credentials.\n",
        )
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"Artifacts: {out}")
    if errors:
        parser.exit(1, "Partial evaluation. See per-case errors in results.json.\n")


if __name__ == "__main__":
    main()
