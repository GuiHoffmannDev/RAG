"""Score the answer and the ordered evidence actually supplied to its generator."""

import math

from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.metrics.collections.base import BaseMetric

from src.models import Answer

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def make_metrics(
    client: AsyncOpenAI, model: str, embedding_model: str, local: bool
) -> dict[str, BaseMetric]:
    options = {"reasoning_effort": "none"} if local else {}
    llm = llm_factory(model, client=client, **options)
    embeddings = embedding_factory(
        "openai", model=embedding_model, client=client, interface="modern"
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_response(
    metrics: dict[str, BaseMetric],
    question: str,
    reference: str,
    answer: Answer,
    evidence: list[str],
) -> tuple[dict[str, float | None], dict[str, str]]:
    scores: dict[str, float | None] = dict.fromkeys(METRICS)
    errors: dict[str, str] = {}
    for name, metric in metrics.items():
        if name == "faithfulness" and answer.abstained:
            continue  # No factual answer to check.
        if name == "answer_relevancy":
            inputs = {"response": answer.answer}
        elif name == "faithfulness":
            inputs = {"response": answer.answer, "retrieved_contexts": evidence}
        elif not evidence:
            scores[name] = 0.0  # Empty retrieval cannot support an answerable reference.
            continue
        else:
            inputs = {"reference": reference, "retrieved_contexts": evidence}
        try:
            result = await metric.ascore(user_input=question, **inputs)
            value = float(result.value)
            if not math.isfinite(value):
                raise ValueError("Non-finite metric result")
            scores[name] = value
        except Exception as exc:  # noqa: BLE001 - isolate each external evaluation call.
            # Provider responses may echo credentials or prompts. Persist only the error class.
            errors[name] = type(exc).__name__
    return scores, errors
