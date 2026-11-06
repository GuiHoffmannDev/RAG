"""Evaluate the same corpus snapshot with a chosen rewrite budget."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from src.agents.runtime import Runtime
from src.config import Settings
from src.eval.metrics import aggregate, score
from src.models import Query
from src.observability.tracing import record_scores


class Case(Query):
    gold_sources: list[str]


async def run(path: Path, budget: int | None) -> dict:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    raw = path.read_bytes()
    cases = [Case.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    if not cases:
        raise ValueError("Evaluation requires at least one labeled question.")
    cfg = Settings.load()
    if budget is not None:
        cfg = Settings.model_validate({**cfg.model_dump(), "max_rewrites": budget})
    judge_model = os.getenv(
        "JUDGE_MODEL", cfg.chat_model if cfg.model_provider == "ollama" else "gpt-4o-mini"
    )
    runtime = Runtime(cfg)
    try:
        alias = cfg.collection
        snapshot = next(
            (
                a.collection_name
                for a in runtime.index.client.get_aliases().aliases
                if a.alias_name == alias
            ),
            alias,
        )
        runtime.index.settings = cfg.model_copy(update={"collection": snapshot})
        rows = []
        async with AsyncOpenAI(
            timeout=180 if cfg.model_provider == "ollama" else 45,
            max_retries=2,
            **cfg.client_options,
        ) as client:
            # Keep the local judge's token budget available for the structured verdict.
            judge_options = {"reasoning_effort": "none"} if cfg.model_provider == "ollama" else {}
            faithfulness = Faithfulness(
                llm=llm_factory(judge_model, client=client, **judge_options)
            )
            for case in cases:
                start = perf_counter()
                state = runtime.ask(case, name="evaluation-rewrite-budget")
                elapsed = perf_counter() - start
                answer = state["result"]
                metrics = score(state, set(case.gold_sources))
                metrics["faithfulness"] = None
                if not answer.abstained:
                    result = await faithfulness.ascore(
                        user_input=case.question,
                        response=answer.answer,
                        retrieved_contexts=[p.text for p in state["evidence"]],
                    )
                    if not math.isfinite(result.value):
                        raise ValueError("RAGAS returned a non-finite faithfulness score.")
                    metrics["faithfulness"] = float(result.value)
                record_scores(answer.observability.trace_id, metrics)
                rows.append(
                    {
                        **case.model_dump(),
                        "candidate_sources": [p.source for p in state["candidates"]],
                        "evidence": [p.model_dump() for p in state["evidence"]],
                        **answer.model_dump(),
                        "latency_s": elapsed,
                        "metrics": metrics,
                    }
                )
    finally:
        runtime.close()
    return {
        "settings": cfg.model_dump(mode="json", exclude={"qdrant_url", "ollama_url"}),
        "snapshot": snapshot,
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "judge_model": judge_model,
        "ragas_version": version("ragas"),
        "case_count": len(rows),
        "summary": aggregate(rows),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("evals/smoke.jsonl"))
    parser.add_argument("--max-rewrites", type=int, choices=range(4))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to preserve previous results.")
    report = asyncio.run(run(args.cases, args.max_rewrites))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
