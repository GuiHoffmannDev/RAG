import asyncio
import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.agents.graph import abstention, build_graph
from src.agents.runtime import Runtime
from src.api.main import app
from src.config import Settings
from src.eval.run_evaluation import GoldenCase, run_case
from src.models import Query, QueryMetrics, ScoredPassage
from src.observability import tracing


def test_offline_trace_contains_nested_calls_routes_and_query_cost(monkeypatch):
    """Exercise the real SDK and LangGraph callbacks with in-memory OTLP and HTTP transports."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-offline")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-offline")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    exporter = InMemorySpanExporter()
    monkeypatch.setattr("langfuse._client.span_processor.OTLPSpanExporter", lambda **_: exporter)
    provider = TracerProvider()
    client = Langfuse(public_key="pk-offline", secret_key="sk-offline", tracer_provider=provider)
    monkeypatch.setattr(tracing, "langfuse_client", lambda: client)
    monkeypatch.setattr(
        client, "get_trace_url", lambda *, trace_id: f"https://example.org/{trace_id}"
    )
    cfg = Settings.load().model_copy(
        update={
            "model_provider": "openai",
            "chat_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_rewrites": 1,
        }
    )
    grades = iter([False, True])

    def respond(request):
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "model": cfg.embedding_model,
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
            )
        schema = json.loads(request.content)["response_format"]["json_schema"]["name"]
        if schema == "Grade":
            data = {"verdicts": [{"passage_id": 1, "relevant": next(grades)}]}
        elif schema == "Rewrite":
            data = {"query": "archive checksum"}
        else:
            data = {"claims": [{"text": "Checksum mismatch.", "passage_ids": [1]}]}
        return httpx.Response(
            200,
            json={
                "id": "chat-offline",
                "object": "chat.completion",
                "created": 1,
                "model": cfg.chat_model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(data)},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    try:
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            options = {
                "api_key": "offline",
                "http_client": http,
                "base_url": "https://example.org/v1",
            }
            model = ChatOpenAI(model=cfg.chat_model, **options)
            embed = OpenAIEmbeddings(
                model=cfg.embedding_model, check_embedding_ctx_length=False, **options
            )
            embed.client = tracing.MeteredEmbeddings(embed.client)
            hit = ScoredPassage(source="errors.md", chunk=1, text="Checksum mismatch.", score=0.75)

            def search(query):
                embed.embed_query(query)
                return [hit]

            runtime = Runtime.__new__(Runtime)
            runtime.index = SimpleNamespace(settings=cfg)
            runtime.graph = build_graph(search, model, cfg.max_rewrites)
            answer = runtime.ask(Query(question="What is ARC-17?"))["result"]
        client.flush()
        spans = exporter.get_finished_spans()
        metrics = answer.observability
        assert answer.rewrites == 1 and not answer.abstained
        assert (
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.embedding_tokens,
            metrics.total_tokens,
        ) == (40, 20, 6, 66)
        assert metrics.estimated_cost_usd == pytest.approx((40 * 0.15 + 20 * 0.60 + 6 * 0.02) / 1e6)
        assert {f"{s.context.trace_id:032x}" for s in spans} == {metrics.trace_id}
        assert metrics.trace_url.endswith(metrics.trace_id)
        assert {s.name for s in spans} >= {
            "retrieve",
            "grade_documents",
            "rewrite_query",
            "generate_answer",
            "embed_query",
        }
        by_id = {s.context.span_id: s for s in spans}
        assert [by_id[s.parent.span_id].name for s in spans if s.name == "embed_query"] == [
            "retrieve"
        ] * 2
        output = "langfuse.observation.output"
        assert [
            s.attributes[output].strip('"') for s in spans if s.name == "decide_to_generate"
        ] == ["rewrite_query", "generate_answer"]
        retrieval = next(s for s in spans if s.name == "retrieve")
        assert json.loads(retrieval.attributes[output])["candidates"] == [hit.model_dump()]
        assert (
            len([s for s in spans if s.attributes.get("langfuse.observation.type") == "generation"])
            == 4
        )
        root = next(s for s in spans if s.name == "corrective-rag")
        assert root.attributes["langfuse.trace.metadata.rewrites"] in ("1", 1)
        # A failed evaluation retains its own trace and all preceding node outputs.
        graph = build_graph(lambda _: [hit], model, 0, baseline=True)
        steps = StringIO()
        case = GoldenCase(question="What happened?", ground_truth="Checksum mismatch.")
        trace = tracing.QueryTrace(cfg, "evaluation-baseline")
        # The transport is closed, so this graph fails in generation after retrieval.
        row = asyncio.run(run_case(case, "baseline", graph, {}, steps, trace))
        assert row["errors"]["generation"] and row["observability"]["trace_id"] != metrics.trace_id
        assert row["observability"]["trace_id"]
        assert json.loads(steps.getvalue())["steps"][0]["node"] == "retrieve"
    finally:
        client.shutdown()
        provider.shutdown()


def test_missing_usage_and_price_remain_unknown(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    cfg = Settings.load().model_copy(update={"model_provider": "openai", "chat_model": "custom"})
    trace = tracing.QueryTrace(cfg, "test")
    with trace.observe("test"):
        trace.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content="hello"))]])
        )
    assert trace.metrics.total_tokens is None and trace.metrics.estimated_cost_usd is None
    assert trace.metrics.trace_id is None and trace.metrics.latency_s >= 0
    trace = tracing.QueryTrace(cfg, "test")
    with trace.observe("test"):
        trace.input_tokens = 10
    assert trace.metrics.total_tokens == 10 and trace.metrics.estimated_cost_usd is None


def test_feedback_validation_retry_and_trace_correlation(monkeypatch):
    runtime = Mock()
    answer = abstention(1)
    answer.observability = QueryMetrics(
        latency_s=1,
        trace_id="a" * 32,
        input_tokens=2,
        output_tokens=3,
        embedding_tokens=1,
        total_tokens=6,
        estimated_cost_usd=0,
    )
    runtime.ask.return_value = {"result": answer}
    monkeypatch.setattr("src.api.main.Runtime", lambda _: runtime)
    client = Mock()
    monkeypatch.setattr(tracing, "langfuse_client", lambda: client)
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-offline")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-offline")
    with TestClient(app) as api:
        result = api.post("/query", json={"question": "Test?"}).json()
        payload = {
            "trace_id": result["observability"]["trace_id"],
            "feedback_token": result["feedback_token"],
            "positive": False,
        }
        assert api.post("/feedback", json={**payload, "trace_id": "b" * 32}).status_code == 403
        assert api.post("/feedback", json={**payload, "positive": 5}).status_code == 422
        assert api.post("/feedback", json={**payload, "trace_id": "invalid"}).status_code == 422
        assert api.post("/feedback", json=payload).is_success
        assert api.post("/feedback", json={**payload, "positive": True}).is_success
        votes = [c.kwargs["request"] for c in client.api.score.create.call_args_list]
        assert votes[0].id == votes[1].id and [v.value for v in votes] == [0, 1]
        assert all(v.trace_id == payload["trace_id"] and v.name == "user_feedback" for v in votes)
        client.api.score.create.side_effect = TimeoutError()
        assert api.post("/feedback", json=payload).status_code == 503
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
        assert api.post("/feedback", json=payload).status_code == 503


def test_streamlit_feedback_keeps_answer_without_repeating_query(monkeypatch):
    from streamlit.testing.v1 import AppTest

    answer = {
        "answer": "Example",
        "abstained": False,
        "rewrites": 1,
        "citations": [],
        "feedback_token": "example-receipt",
        "observability": {
            "trace_id": "a" * 32,
            "trace_url": "https://example.org/trace",
            "latency_s": 1.5,
            "estimated_cost_usd": 0,
            "total_tokens": 42,
        },
    }
    monkeypatch.setenv("API_URL", "http://test-feedback")
    monkeypatch.setattr(httpx, "get", Mock(side_effect=httpx.ConnectError("offline")))
    post = Mock(
        side_effect=[
            Mock(json=Mock(return_value=answer)),
            httpx.ConnectError("feedback temporarily offline"),
            Mock(),
        ]
    )
    monkeypatch.setattr(httpx, "post", post)
    ui = AppTest.from_file("../app/main.py").run()
    ui.text_area[0].set_value("Example question")
    ui.button[0].click().run()
    assert not ui.exception and ui.link_button[0].url == answer["observability"]["trace_url"]
    assert [m.value for m in ui.metric] == ["1.50 s", "42", "$0.000000"]
    ui.feedback[0].set_value(0).run()
    assert ui.warning and post.call_count == 2
    ui.run()
    assert post.call_count == 2
    next(button for button in ui.button if button.label == "Reenviar avaliação").click().run()
    ui.run()
    assert not ui.exception and not ui.warning and post.call_count == 3
    assert post.call_args.kwargs["json"] == {
        "trace_id": "a" * 32,
        "feedback_token": "example-receipt",
        "positive": False,
    }
