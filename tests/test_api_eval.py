import json
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
from langchain_core.exceptions import ModelRefusalError
from qdrant_client.http.exceptions import ResponseHandlingException

from src.agents.graph import abstention
from src.api.main import app
from src.eval.metrics import aggregate, score
from src.models import Answer, Citation, Passage, QueryResponse


def test_http_contract(monkeypatch):
    runtime = Mock()
    runtime.ask.return_value = {"result": abstention(1)}
    monkeypatch.setattr("src.api.main.Runtime", lambda _: runtime)
    with TestClient(app) as api:
        assert api.get("/health").status_code == 200
        assert api.post("/query", json={"question": "  "}).status_code == 422
        result = api.post("/query", json={"question": "Unknown?"})
        assert result.status_code == 200 and result.json()["abstained"]
        runtime.ask.side_effect = ResponseHandlingException(RuntimeError("internal detail"))
        assert api.post("/query", json={"question": "Retry?"}).status_code == 503
        runtime.ask.side_effect = ModelRefusalError("model refused")
        assert api.post("/query", json={"question": "Retry?"}).status_code == 422
    runtime.close.assert_called_once()


def test_metrics_and_conditional_denominators():
    passage = Passage(source="a.md", chunk=1, text="Evidence")
    answer = Answer(
        answer="Claim [1]",
        citations=[Citation(label=1, **passage.model_dump())],
        abstained=False,
        rewrites=1,
    )
    hit = score({"candidates": [passage], "result": answer}, {"a.md", "b.md"})
    miss = score({"candidates": [], "result": abstention(0)}, set())
    summary = aggregate([{"metrics": hit, "latency_s": 1}, {"metrics": miss, "latency_s": 3}])
    assert summary["source_recall_at_k"] == {"mean": 0.5, "n": 1}
    assert summary["citation_precision"] == {"mean": 1, "n": 1}
    assert summary["answer_coverage"]["mean"] == 0.5
    assert summary["abstention_accuracy"]["mean"] == 1
    assert summary["latency_p95_s"] == 3


def test_query_response_schema_matches_committed_contract():
    snapshot = Path("tests/contracts/query_response.schema.json")
    assert json.loads(snapshot.read_text()) == QueryResponse.model_json_schema(), (
        "The /query response schema changed. If intentional, run "
        "`python -m tests.contracts.regenerate` and commit the snapshot."
    )


def test_query_response_fields_and_types(monkeypatch):
    passage = Passage(source="a.md", chunk=2, text="Evidence")
    answer = Answer(
        answer="Claim [1]",
        citations=[Citation(label=1, score=0.5, **passage.model_dump())],
        abstained=False,
        rewrites=1,
    )
    runtime = Mock()
    runtime.index.settings.chat_models = ["m"]
    runtime.ask.return_value = {"result": answer}
    monkeypatch.setattr("src.api.main.Runtime", lambda _: runtime)
    with TestClient(app) as api:
        body = api.post("/query", json={"question": "Q?"}).json()
    assert body == {
        "answer": "Claim [1]",
        "citations": [{"source": "a.md", "chunk": 2, "text": "Evidence", "score": 0.5, "label": 1}],
        "abstained": False,
        "rewrites": 1,
        "observability": None,
        "feedback_token": None,
    }
