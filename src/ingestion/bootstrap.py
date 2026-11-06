"""Index the bundled corpus on first start, when the configured collection is missing or empty."""

import logging
import os
from pathlib import Path

from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import Settings
from src.ingestion.index import publish
from src.retrieval.hybrid import HybridIndex

log = logging.getLogger(__name__)


def needs_index(index: HybridIndex) -> bool:
    cfg = index.settings
    try:
        info = index.client.get_collection(cfg.collection)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return True
        raise
    # A different embedding model needs a new vector space, so the old index is unusable.
    return not info.points_count or cfg.dense_name not in info.config.params.vectors


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    index = HybridIndex(Settings.load())
    try:
        if not needs_index(index):
            log.info("Collection %s is populated. Skipping indexing.", index.settings.collection)
            return
        publish(Path(os.getenv("CORPUS_DIR", "assets/corpus")), index)
    finally:
        index.client.close()


if __name__ == "__main__":
    main()
