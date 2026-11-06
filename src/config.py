import os
import tomllib
from importlib.resources import files
from typing import Literal, Self
from urllib.parse import quote

from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, SecretStr, model_validator


class Settings(BaseModel):
    model_provider: Literal["openai", "ollama"]
    ollama_url: HttpUrl
    chat_model: str = Field(min_length=1)
    additional_chat_models: str = ""
    embedding_model: str = Field(min_length=1)
    qdrant_url: HttpUrl
    # Required only for Qdrant Cloud or a self-hosted server with an API key.
    qdrant_api_key: SecretStr | None = None
    collection: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    candidate_depth: int = Field(ge=1, le=100)
    top_k: int = Field(ge=1, le=20)
    max_rewrites: int = Field(ge=0, le=3)
    # Grader variants. Defaults reproduce the original behavior.
    grade_rule: Literal["majority", "any"] = "majority"
    grader_prompt: Literal["v1", "fewshot"] = "v1"
    grader_model: str | None = Field(default=None, min_length=1)
    chunk_size: int = Field(ge=32, le=6000)
    chunk_overlap: int = Field(ge=0)

    chat_input_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    chat_output_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    embedding_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        if self.chunk_overlap >= self.chunk_size or self.top_k > self.candidate_depth:
            raise ValueError("Overlap must be below chunk size and top_k within candidate depth.")
        return self

    @property
    def chat_models(self) -> list[str]:
        extra = (name.strip() for name in self.additional_chat_models.split(","))
        return list(dict.fromkeys([self.chat_model, *filter(None, extra)]))

    @property
    def client_options(self) -> dict[str, str]:
        if self.model_provider == "ollama":
            # Ollama accepts the OpenAI protocol. The SDK requires a placeholder key.
            return {"base_url": str(self.ollama_url).rstrip("/") + "/v1", "api_key": "ollama"}
        return {}

    @property
    def dense_name(self) -> str:
        # A changed embedding model must never silently query the previous vector space.
        if self.model_provider == "ollama":
            return "ollama-" + quote(self.embedding_model, safe="")
        return f"dense-{self.embedding_model}"

    @classmethod
    def load(cls) -> Self:
        load_dotenv(".env", override=False)
        defaults = tomllib.loads(files("configs").joinpath("defaults.toml").read_text())
        local = defaults.pop("ollama")
        if os.getenv("MODEL_PROVIDER", defaults["model_provider"]) == "ollama":
            defaults.update(local)
        return cls.model_validate(
            {k: os.getenv(k.upper(), defaults.get(k)) for k in cls.model_fields}
        )
