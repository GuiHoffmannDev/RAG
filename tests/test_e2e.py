"""POST /query through the real app, Runtime and graph. Only the model and index are fake."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.agents.runtime import Runtime
from src.api.main import app
from src.config import Settings
from tests.conftest import PASSAGE, FakeChat


@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    cfg = Settings.load().model_copy(update={"max_rewrites": 2})
    search = Mock(return_value=[PASSAGE])

    def start(grades):
        chat = FakeChat(grades)
        index = SimpleNamespace(search=search, settings=cfg, client=Mock())
        monkeypatch.setattr("src.agents.runtime.HybridIndex", lambda _: index)
        monkeypatch.setattr(Runtime, "chat_model", staticmethod(lambda *a, **k: chat.model))
        return TestClient(app), search, chat

    return start


def test_rewrite_then_cited_answer(serve):
    client, search, chat = serve([[(1, False)], [(1, True)]])
    with client as api:
        body = api.post("/query", json={"question": "What is ARC-17?", "inspect_run": True}).json()
    assert body["abstained"] is False and body["rewrites"] == 1
    assert body["answer"] == "The checksum differs. [1]"
    assert body["citations"][0]["source"] == "errors.md"
    assert [c.args[0] for c in search.call_args_list] == ["What is ARC-17?", "archive checksum"]
    steps = body["inspection"]["steps"]
    assert [s["node"] for s in steps].count("grade_documents") == 2
    # The raw grader verdicts reach the inspection payload.
    assert steps[1]["verdicts"] == [{"passage_id": 1, "relevant": False}]
    assert len(chat.prompts) == 2


def test_abstains_when_evidence_never_arrives(serve):
    client, search, _ = serve([[(1, False)]] * 3)
    with client as api:
        response = api.post("/query", json={"question": "Enterprise price?"})
    body = response.json()
    assert response.status_code == 200
    assert body["abstained"] is True and body["citations"] == [] and body["rewrites"] == 2
    assert search.call_count == 3
