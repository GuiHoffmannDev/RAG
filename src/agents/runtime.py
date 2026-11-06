import logging

from langchain_openai import ChatOpenAI

from src.agents.graph import PROMPT_VERSION, State, build_graph, generate
from src.config import Settings
from src.models import Query
from src.observability.tracing import QueryTrace, flush_traces, tracing_enabled
from src.retrieval.hybrid import HybridIndex


class Runtime:
    def __init__(self, settings: Settings):
        tracing_enabled()  # Validate the key pair before serving queries.
        logging.basicConfig()
        logging.getLogger("src.observability.tracing").setLevel(logging.INFO)
        self.index = HybridIndex(settings)
        self.model = self.chat_model(settings, 0)
        self.graph = self.build(settings, self.model, settings.max_rewrites)

    def build(self, cfg: Settings, chat: ChatOpenAI, budget: int):
        grader = self.chat_model(cfg, 0, cfg.grader_model) if cfg.grader_model else None
        return build_graph(
            self.index.search,
            chat,
            budget,
            grader_model=grader,
            grade_rule=cfg.grade_rule,
            grader_prompt=cfg.grader_prompt,
        )

    @staticmethod
    def chat_model(settings: Settings, temperature: float, name: str | None = None) -> ChatOpenAI:
        return ChatOpenAI(
            model=name or settings.chat_model,
            temperature=temperature,
            reasoning_effort="none" if settings.model_provider == "ollama" else None,
            timeout=180 if settings.model_provider == "ollama" else 30,
            max_retries=2,
            **settings.client_options,
        )

    def ask(
        self,
        request: Query,
        *,
        name: str = "corrective-rag",
        model: str | None = None,
        temperature: float = 0,
        max_rewrites: int | None = None,
        steps: list[dict] | None = None,
    ) -> State:
        cfg = self.index.settings
        graph = self.graph
        model = model or cfg.chat_model
        budget = cfg.max_rewrites if max_rewrites is None else max_rewrites
        if (model, temperature, budget) != (cfg.chat_model, 0, cfg.max_rewrites):
            overrides = {"chat_model": model, "max_rewrites": budget}
            change_chat = model != cfg.chat_model or temperature != 0
            if model != cfg.chat_model:
                # The deployment's custom rates apply only to its default chat model.
                overrides.update(chat_input_usd_per_million=None, chat_output_usd_per_million=None)
            cfg = cfg.model_copy(update=overrides)
            chat = self.chat_model(cfg, temperature) if change_chat else self.model
            graph = self.build(cfg, chat, budget)
        trace = QueryTrace(cfg, name, prompt_version=PROMPT_VERSION, temperature=temperature)
        with trace.observe(request.question) as config:
            if steps is None:
                state = graph.invoke(
                    {"question": request.question, "query": request.question, "rewrites": 0},
                    config=config,
                )
            else:
                state = generate(graph, request.question, config, steps)
            trace.answer = state["result"]
        state["result"].observability = trace.metrics
        return state

    def close(self) -> None:
        self.index.client.close()
        flush_traces()
