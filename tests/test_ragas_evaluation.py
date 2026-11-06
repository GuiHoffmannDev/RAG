import asyncio
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.runnables import RunnableLambda

from src.agents.graph import Claim, Draft, Grade, Rewrite, Verdict, abstention, build_graph
from src.config import Settings
from src.eval import run_evaluation
from src.eval.ragas_metrics import METRICS, score_response
from src.models import ScoredPassage


def test_golden_set_has_unique_questions_and_indexable_sources():
    cases = run_evaluation.load_cases(Path("evals/golden_set.jsonl").read_bytes())
    assert 15 <= len(cases) <= 40
    assert any(not c.answerable for c in cases) and any(c.answerable for c in cases)
    for case in cases:
        # An unanswerable case has no source and no reference answer.
        assert bool(case.gold_sources) == case.answerable == bool(case.ground_truth)
        assert all((Path("assets/corpus") / source).is_file() for source in case.gold_sources)
    with pytest.raises(ValueError, match="unique"):
        run_evaluation.load_cases((cases[0].model_dump_json() + "\n").encode() * 2)


def test_answerable_case_requires_a_reference():
    with pytest.raises(ValueError, match="line 1"):
        run_evaluation.load_cases(b'{"question": "Q?"}\n')
    case = run_evaluation.load_cases(b'{"question": "Q?", "answerable": false}\n')[0]
    assert not case.answerable and case.ground_truth == ""


def test_empty_evidence_and_abstention_have_explicit_denominators():
    metrics = {key: Mock() for key in METRICS}
    metrics["answer_relevancy"].ascore = AsyncMock(return_value=SimpleNamespace(value=0))
    scores, errors = asyncio.run(
        score_response(metrics, "Question", "Reference", abstention(0), [])
    )
    assert not errors
    assert scores == {
        "faithfulness": None,
        "answer_relevancy": 0,
        "context_precision": 0,
        "context_recall": 0,
    }


def test_pipeline_preserves_traces_and_compares_only_valid_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    hit = ScoredPassage(source="policy.md", score=1, text="Retrieved evidence", chunk=1)
    runtime = Mock()
    runtime.index.search.return_value = [hit]
    runtime.index.client.get_aliases.return_value.aliases = [
        SimpleNamespace(alias_name="demo", collection_name="snapshot")
    ]
    draft = Draft(claims=[Claim(text="Generated claim.", passage_ids=[1])])
    responders = {
        Grade: RunnableLambda(lambda _: Grade(verdicts=[Verdict(passage_id=1, relevant=True)])),
        Draft: RunnableLambda(lambda _: draft),
        Rewrite: RunnableLambda(lambda _: pytest.fail("unexpected rewrite")),
    }
    runtime.model.with_structured_output.side_effect = responders.__getitem__
    runtime.graph = build_graph(runtime.index.search, runtime.model, 1)
    monkeypatch.setattr(run_evaluation, "Runtime", lambda _: runtime)
    metrics = {
        key: Mock(ascore=AsyncMock(return_value=SimpleNamespace(value=0.7))) for key in METRICS
    }
    metrics["context_recall"].ascore.side_effect = [TimeoutError(), SimpleNamespace(value=1)]
    monkeypatch.setattr(run_evaluation, "make_metrics", lambda *args: metrics)
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question": "Question?",
                "ground_truth": "Reference",
                "contexts": ["Gold-only passage"],
            }
        )
    )
    cfg = Settings.load().model_copy(update={"collection": "demo"})
    out, summary, errors = asyncio.run(
        run_evaluation.evaluate(dataset, tmp_path, cfg, "ollama", "chat", "embedding")
    )
    assert runtime.index.settings.collection == "snapshot"
    assert runtime.index.search.call_count == 2
    assert runtime.close.called and errors
    for call in metrics["context_precision"].ascore.await_args_list:
        assert call.kwargs == {
            "user_input": "Question?",
            "reference": "Reference",
            "retrieved_contexts": [hit.text],
        }
    assert summary["context_recall"] == {
        "baseline": None,
        "corrective": None,
        "delta": None,
        "n_pairs": 0,
    }
    assert summary["faithfulness"]["n_pairs"] == 1
    traces = [json.loads(line) for line in (out / "traces.jsonl").read_text().splitlines()]
    assert [[s["node"] for s in r["steps"]] for r in traces] == [
        ["retrieve", "generate_answer"],
        ["retrieve", "grade_documents", "generate_answer"],
    ]
    with (out / "results.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["total_tokens"] == "0" and rows[0]["trace_url"] == ""
    assert "observability" in traces[0]
    assert rows[0]["context_recall"] == "" and float(rows[1]["context_recall"]) == 1
    report = json.loads((out / "results.json").read_text())
    assert report["cases"][0]["errors"] == {"context_recall": "TimeoutError"}
    assert report["cases"][1]["answer"]["citations"][0]["text"] == hit.text
    assert "Baseline RAG | Agentic Corrective RAG" in (out / "summary.md").read_text()
