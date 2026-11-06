import pytest

from src.config import Settings


@pytest.mark.parametrize(
    "provider,chat,embedding,alias",
    [
        ("ollama", "qwen3.5:4b", "qwen3-embedding:0.6b", "runbook_ollama"),
        ("openai", "gpt-4o-mini", "text-embedding-3-small", "runbook"),
    ],
)
def test_provider_defaults(monkeypatch, tmp_path, provider, chat, embedding, alias):
    monkeypatch.chdir(tmp_path)
    for key in ("CHAT_MODEL", "EMBEDDING_MODEL", "COLLECTION", "OLLAMA_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MODEL_PROVIDER", provider)
    cfg = Settings.load()
    assert (cfg.chat_model, cfg.embedding_model, cfg.collection) == (chat, embedding, alias)
    if provider == "ollama":
        assert cfg.client_options == {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}
        assert cfg.dense_name == "ollama-qwen3-embedding%3A0.6b"
    else:
        assert cfg.client_options == {}
        assert cfg.dense_name == f"dense-{embedding}"
