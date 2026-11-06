from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

from src.agents.graph import Claim, Draft, Grade, Rewrite, Verdict
from src.agents.runtime import Runtime
from src.api.main import app
from src.models import ScoredPassage


def test_streamlit_inspects_real_graph_and_restores_history(monkeypatch):
    hit = ScoredPassage(source="errors.md", chunk=1, text="Checksum mismatch.", score=1)
    grades = iter([False, True, False])
    model = Mock()
    model.with_structured_output.side_effect = {
        Grade: RunnableLambda(
            lambda _: Grade(verdicts=[Verdict(passage_id=1, relevant=next(grades))])
        ),
        Rewrite: RunnableLambda(lambda _: Rewrite(query="archive checksum")),
        Draft: RunnableLambda(lambda _: Draft(claims=[Claim(text=hit.text, passage_ids=[1])])),
    }.__getitem__
    create_model = Mock(return_value=model)
    search = Mock(return_value=[hit])
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setenv("API_URL", "http://testserver")
    monkeypatch.setenv("CHAT_MODEL", "default-chat")
    monkeypatch.setenv("ADDITIONAL_CHAT_MODELS", "alternative-chat")
    monkeypatch.setenv("MAX_REWRITES", "2")
    monkeypatch.setattr(Runtime, "chat_model", staticmethod(create_model))
    monkeypatch.setattr(
        "src.agents.runtime.HybridIndex",
        lambda cfg: SimpleNamespace(settings=cfg, search=search, client=Mock()),
    )
    with TestClient(app) as api:
        monkeypatch.setattr(httpx, "get", lambda url, **_: api.get(url))
        post = Mock(side_effect=lambda url, json, **_: api.post(url, json=json))
        monkeypatch.setattr(httpx, "post", post)
        ui = AppTest.from_file("../app/main.py").run()
        ui.button[1].click().run()
        question = ui.text_area[0].value
        assert "ARC-17" in question and not post.called
        ui.selectbox[0].set_value("alternative-chat")
        ui.slider[0].set_value(0.4)
        ui.slider[1].set_value(1)
        ui.button[0].click().run()
        assert not ui.exception
        assert post.call_args.kwargs["json"] == {
            "question": question,
            "inspect_run": True,
            "temperature": 0.4,
            "model": "alternative-chat",
            "max_rewrites": 1,
        }
        cfg, temperature = create_model.call_args.args
        assert (cfg.chat_model, temperature, cfg.max_rewrites) == ("alternative-chat", 0.4, 1)
        assert [call.args[0] for call in search.call_args_list] == [question, "archive checksum"]
        assert {item.value for item in ui.text} >= {
            "Rejeitado · errors.md · trecho 1",
            "Aceito · errors.md · trecho 1",
            "archive checksum",
            "Checksum mismatch. [1]",
            hit.text,
        }
        previous = ui.session_state.selected
        ui.toggle[0].set_value(True).run()
        assert len(ui.json) == 1 and post.call_count == 1
        state = ui.session_state.history[0]["answer"]["inspection"]["state"]
        assert state["question"] == question and state["query"] == "archive checksum"
        assert state["candidates"] == state["evidence"] == [hit.model_dump()]
        assert state["result"]["citations"][0]["text"] == hit.text

        ui.slider[1].set_value(0)
        ui.text_area[0].set_value("Second question?")
        ui.button[0].click().run()
        assert ui.session_state.history[0]["answer"]["abstained"]
        ui.selectbox(key="selected").set_value(previous).run()
        assert not ui.exception and post.call_count == 2 and ui.session_state.selected == previous
        assert "archive checksum" in [item.value for item in ui.text]
        assert app.state.runtime.index.settings.chat_model == "default-chat"
        assert app.state.runtime.index.settings.max_rewrites == 2
        for invalid in ({"model": "unknown"}, {"temperature": 2}, {"max_rewrites": -1}):
            assert api.post("/query", json={"question": question, **invalid}).status_code == 422
