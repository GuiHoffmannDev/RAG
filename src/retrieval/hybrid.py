from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models

from src.config import Settings
from src.models import ScoredPassage
from src.observability.tracing import MeteredEmbeddings


def bm25(text: str) -> models.Document:
    return models.Document(text=text, model="qdrant/bm25")


class HybridIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        # The SDK flag delegates BM25 to the local Qdrant server.
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        self.client = QdrantClient(
            url=str(settings.qdrant_url),
            api_key=api_key or None,
            cloud_inference=True,
            timeout=15,
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            request_timeout=120 if settings.model_provider == "ollama" else 30,
            max_retries=2,
            check_embedding_ctx_length=settings.model_provider != "ollama",
            **settings.client_options,
        )

        self.embeddings.client = MeteredEmbeddings(self.embeddings.client)

    def search(self, query: str) -> list[ScoredPassage]:
        cfg = self.settings
        vector = self.embeddings.embed_query(query)
        hits = self.client.query_points(
            collection_name=cfg.collection,
            prefetch=[
                models.Prefetch(query=vector, using=cfg.dense_name, limit=cfg.candidate_depth),
                models.Prefetch(query=bm25(query), using="bm25", limit=cfg.candidate_depth),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=cfg.top_k,
            with_payload=True,
        )
        return [
            ScoredPassage.model_validate({**(hit.payload or {}), "score": hit.score})
            for hit in hits.points
        ]
