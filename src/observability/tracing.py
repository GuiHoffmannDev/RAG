"""Per-query telemetry shared by the API, CLI and evaluation runners."""

import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import cache
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, LangfuseAgent
from langfuse.api import CreateScoreRequest
from langfuse.langchain import CallbackHandler
from openai.resources.embeddings import Embeddings
from opentelemetry.trace import get_current_span

from src.config import Settings
from src.models import Answer, QueryMetrics

log = logging.getLogger(__name__)
active_query: ContextVar["QueryTrace | None"] = ContextVar("rag_query", default=None)


def tracing_enabled() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return False
    public, secret = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if bool(public) != bool(secret):
        raise ValueError("Langfuse tracing requires both public and secret keys.")
    return bool(public)


@cache
def langfuse_client() -> Langfuse:
    # The SDK caches the project ID for trace links on the client instance.
    return Langfuse()


@contextmanager
def optional_telemetry(operation: str) -> Iterator[None]:
    """Isolate SDK failures from retrieval, generation and local evaluation artifacts."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - optional SDK operations must not mask graph errors.
        log.warning("Langfuse %s failed (%s)", operation, type(exc).__name__)


class QueryTrace(BaseCallbackHandler):
    """One instance per graph execution. LangGraph propagates its context into nodes."""

    def __init__(self, cfg: Settings, name: str, **metadata: object):
        self.cfg = cfg
        self.name = name
        self.metadata = {
            "model": cfg.chat_model,
            "embedding_model": cfg.embedding_model,
            "provider": cfg.model_provider,
            "collection": cfg.collection,
            "max_rewrites": cfg.max_rewrites,
            **metadata,
        }
        self.input_tokens = self.output_tokens = self.embedding_tokens = 0
        self.chat_complete = self.embedding_complete = True
        self.client: Langfuse | None = None
        self.answer: Answer | None = None
        self.metrics: QueryMetrics

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # Each structured model invocation in this graph uses a single prompt and completion.
        gen = (
            response.generations[0][0] if response.generations and response.generations[0] else None
        )
        usage = (
            getattr(gen.message, "usage_metadata", None)
            if isinstance(gen, ChatGeneration)
            else None
        )
        if usage is None:
            self.chat_complete = False
        else:
            self.input_tokens += usage["input_tokens"]
            self.output_tokens += usage["output_tokens"]

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        self.chat_complete = False

    def estimate_cost(self) -> float | None:
        cfg = self.cfg
        if cfg.model_provider == "ollama":
            return 0.0
        if not (self.chat_complete and self.embedding_complete):
            return None
        default_chat = cfg.chat_model in {"gpt-4o-mini", "gpt-4o-mini-2024-07-18"}
        charges = (
            (self.input_tokens, cfg.chat_input_usd_per_million, 0.15 if default_chat else None),
            (self.output_tokens, cfg.chat_output_usd_per_million, 0.60 if default_chat else None),
            (
                self.embedding_tokens,
                cfg.embedding_usd_per_million,
                0.02 if cfg.embedding_model == "text-embedding-3-small" else None,
            ),
        )
        cost = 0.0
        for count, override, default in charges:
            rate = default if override is None else override
            if count and rate is None:
                return None
            cost += count * (rate or 0)
        return cost / 1_000_000

    @contextmanager
    def observe(self, question: str) -> Iterator[RunnableConfig]:
        enabled = tracing_enabled()
        callbacks = [self]
        trace_id = None
        root = None
        with ExitStack() as stack:
            if enabled:
                with optional_telemetry("trace initialization"):
                    client = langfuse_client()
                    root = stack.enter_context(
                        client.start_as_current_observation(
                            name=self.name,
                            as_type="agent",
                            input=question,
                            metadata=self.metadata,
                        )
                    )
                    if get_current_span().is_recording():
                        callbacks.append(
                            CallbackHandler(public_key=os.environ["LANGFUSE_PUBLIC_KEY"])
                        )
                        self.client, trace_id = client, root.trace_id
            token = active_query.set(self)
            start = perf_counter()
            try:
                yield RunnableConfig(
                    run_name="rag-graph", callbacks=callbacks, metadata=self.metadata
                )
            finally:
                elapsed = perf_counter() - start
                active_query.reset(token)
                self.finish(elapsed, question, trace_id, root)

    def finish(
        self, elapsed: float, question: str, trace_id: str | None, root: LangfuseAgent | None
    ) -> None:
        complete = self.chat_complete and self.embedding_complete
        total = self.input_tokens + self.output_tokens + self.embedding_tokens
        self.metrics = QueryMetrics(
            latency_s=elapsed,
            input_tokens=self.input_tokens if self.chat_complete else None,
            output_tokens=self.output_tokens if self.chat_complete else None,
            embedding_tokens=self.embedding_tokens if self.embedding_complete else None,
            total_tokens=total if complete else None,
            estimated_cost_usd=self.estimate_cost(),
            trace_id=trace_id,
        )
        if self.client and root:
            with optional_telemetry("trace finalization"):
                summary = {
                    **self.metadata,
                    **self.metrics.model_dump(exclude={"trace_id", "trace_url"}),
                }
                output = None
                if self.answer is not None:
                    summary["rewrites"] = self.answer.rewrites
                    output = self.answer.model_dump(exclude={"observability"})
                root.update(output=output, metadata=summary)
                root.update_trace(name=self.name, input=question, output=output, metadata=summary)
                self.metrics.trace_url = self.client.get_trace_url(trace_id=trace_id)
        log.info(
            "rag_query name=%s model=%s status=%s %s",
            self.name,
            self.cfg.chat_model,
            "ok" if self.answer else "error",
            self.metrics.model_dump_json(exclude={"trace_url"}),
        )


class MeteredEmbeddings:
    """Read provider usage before LangChain extracts vectors and discards the response."""

    def __init__(self, client: Embeddings):
        self.client = client

    def create(self, **kwargs):
        query = active_query.get()
        if query is None:
            return self.client.create(**kwargs)
        with ExitStack() as stack:
            span = None
            if query.client:
                with optional_telemetry("embedding span"):
                    span = stack.enter_context(
                        query.client.start_as_current_observation(
                            name="embed_query",
                            as_type="embedding",
                            model=query.cfg.embedding_model,
                        )
                    )
            try:
                response = self.client.create(**kwargs)
            except Exception:
                query.embedding_complete = False
                raise
            count = getattr(getattr(response, "usage", None), "prompt_tokens", None)
            if count is None:
                query.embedding_complete = False
            else:
                query.embedding_tokens += count
                if span:
                    with optional_telemetry("embedding usage"):
                        span.update(usage_details={"input": count, "total": count})
            return response


def record_scores(trace_id: str | None, scores: dict[str, float | None]) -> None:
    if not trace_id:
        return
    with optional_telemetry("score export"):
        for name, value in scores.items():
            if value is not None:
                langfuse_client().create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    data_type="NUMERIC",
                    score_id=str(uuid5(NAMESPACE_URL, f"{trace_id}/{name}")),
                )


def submit_feedback(trace_id: str, positive: bool) -> None:
    # Wait for Langfuse to accept the vote so the UI can offer a retry on failure.
    langfuse_client().api.score.create(
        request=CreateScoreRequest(
            id=str(uuid5(NAMESPACE_URL, f"{trace_id}/user_feedback")),
            trace_id=trace_id,
            name="user_feedback",
            value=int(positive),
            data_type="BOOLEAN",
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "default"),
        )
    )


def flush_traces() -> None:
    if tracing_enabled():
        with optional_telemetry("flush"):
            langfuse_client().flush()
