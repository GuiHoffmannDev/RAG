import hmac
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from langchain_core.exceptions import ModelError, ModelRefusalError, OutputParserException
from openai import OpenAIError
from pydantic import ValidationError
from qdrant_client.http.exceptions import ApiException

from src.agents.runtime import Runtime
from src.config import Settings
from src.models import Feedback, InteractiveQuery, QueryResponse, RunInspection
from src.observability.tracing import submit_feedback, tracing_enabled

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    runtime = Runtime(Settings.load())
    app.state.runtime = runtime
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(title="Agentic Corrective RAG", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    index = request.app.state.runtime.index
    try:
        info = index.client.get_collection(index.settings.collection)
        if index.settings.dense_name not in info.config.params.vectors:
            raise HTTPException(503, "Reindex the corpus for the configured embedding model.")
    except ApiException as exc:
        raise HTTPException(503, "Index unavailable. Check Qdrant and run ingestion.") from exc
    return {"status": "ready"}


def feedback_token(trace_id: str) -> str:
    key = os.environ["LANGFUSE_SECRET_KEY"].encode()
    return hmac.digest(key, f"rag-feedback/{trace_id}".encode(), "sha256").hex()


@app.get("/settings")
def settings(request: Request) -> dict:
    cfg = request.app.state.runtime.index.settings
    return {
        "provider": cfg.model_provider,
        "models": cfg.chat_models,
        "max_rewrites": cfg.max_rewrites,
    }


@app.post("/query", response_model=QueryResponse, response_model_exclude_unset=True)
def query(body: InteractiveQuery, request: Request) -> QueryResponse:
    try:
        runtime = request.app.state.runtime
        if body.model and body.model not in runtime.index.settings.chat_models:
            raise HTTPException(422, "Select a configured chat model.")
        steps = [] if body.inspect_run else None
        state = runtime.ask(
            body,
            model=body.model,
            temperature=body.temperature,
            max_rewrites=body.max_rewrites,
            steps=steps,
        )
        answer = state["result"]
        trace_id = answer.observability.trace_id if answer.observability else None
        response = QueryResponse(
            **answer.model_dump(),
            feedback_token=feedback_token(trace_id) if trace_id else None,
        )
        if steps is not None:
            response.inspection = RunInspection(steps=steps, state=state)
        return response
    except ModelRefusalError as exc:
        raise HTTPException(422, "The model declined this request.") from exc
    except (ModelError, OpenAIError, ApiException, OutputParserException, ValidationError) as exc:
        log.warning("Query dependency failed (%s)", type(exc).__name__)
        raise HTTPException(
            503, "Query dependency unavailable. Check the index and provider."
        ) from exc


@app.post("/feedback")
def feedback(body: Feedback) -> dict[str, str]:
    if not tracing_enabled():
        raise HTTPException(503, "Langfuse feedback is disabled.")
    if not hmac.compare_digest(body.feedback_token, feedback_token(body.trace_id)):
        raise HTTPException(403, "Feedback token does not match this response.")
    try:
        submit_feedback(body.trace_id, body.positive)
    except Exception as exc:
        log.warning("Could not submit feedback (%s)", type(exc).__name__)
        raise HTTPException(503, "Feedback is temporarily unavailable.") from exc
    return {"status": "accepted"}
