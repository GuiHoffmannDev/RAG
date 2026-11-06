import os
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.config import Settings
from src.ingestion.index import passages, publish
from src.retrieval.hybrid import HybridIndex


def test_chunk_overlap_and_end_boundary(tmp_path):
    (tmp_path / "runbook.md").write_text("abcdefghij")
    (tmp_path / "other.csv").write_text("skip")
    chunks = list(passages(tmp_path, size=6, overlap=2))
    assert [(p.text, p.chunk, p.source) for p in chunks] == [
        ("abcdef", 1, "runbook.md"),
        ("efghij", 2, "runbook.md"),
    ]


def test_partial_index_is_never_published(tmp_path):
    (tmp_path / "long.txt").write_text("x" * 1200)
    index = Mock()
    index.settings = Settings.load().model_copy(update={"chunk_size": 32, "chunk_overlap": 0})
    index.embeddings.embed_documents.side_effect = [[[1.0]] * 32, RuntimeError("provider")]
    with pytest.raises(RuntimeError, match="provider"):
        publish(tmp_path, index)
    index.client.upsert.assert_called_once()
    index.client.update_collection_aliases.assert_not_called()


@pytest.mark.skipif(not os.getenv("TEST_QDRANT_URL"), reason="Requires a disposable Qdrant server")
def test_server_side_bm25_fusion_and_alias_swap(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    cfg = Settings.load().model_copy(
        update={"collection": f"test-{uuid4().hex}", "qdrant_url": os.environ["TEST_QDRANT_URL"]}
    )
    index = HybridIndex(cfg)
    index.embeddings = Mock()
    index.embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    index.embeddings.embed_query.return_value = [1.0, 0.0]
    versions = []
    try:
        versions.append(publish(Path("assets/corpus"), index))
        hits = index.search("ARC-17")
        assert any("ARC-17" in p.text for p in hits)
        assert all(p.score > 0 for p in hits)
        (tmp_path / "replacement.md").write_text("Borealis is a test device.")
        versions.append(publish(tmp_path, index))
        assert [p.source for p in index.search("Borealis")] == ["replacement.md"]
    finally:
        for name in versions:
            index.client.delete_collection(name)
        index.client.close()
