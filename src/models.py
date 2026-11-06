from typing import Annotated

from pydantic import BaseModel, Field, StrictBool, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class Query(BaseModel):
    question: Question


class InteractiveQuery(Query):
    model: str | None = Field(default=None, min_length=1, max_length=200)
    temperature: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    max_rewrites: int | None = Field(default=None, ge=0, le=3)
    inspect_run: bool = False


class Passage(BaseModel):
    source: str
    chunk: int = Field(ge=1)
    text: str


class ScoredPassage(Passage):
    score: float = Field(allow_inf_nan=False, description="Qdrant reciprocal rank fusion score.")


class Citation(Passage):
    score: float | None = Field(default=None, description="Qdrant reciprocal rank fusion score.")
    label: int


class QueryMetrics(BaseModel):
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    embedding_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None
    trace_id: str | None = None
    trace_url: str | None = None


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    rewrites: int
    observability: QueryMetrics | None = None


class RunInspection(BaseModel):
    steps: list[dict]
    state: dict


class QueryResponse(Answer):
    inspection: RunInspection | None = None
    feedback_token: str | None = None


class Feedback(BaseModel):
    feedback_token: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    positive: StrictBool
