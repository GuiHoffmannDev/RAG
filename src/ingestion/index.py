"""Build a full snapshot and publish it only after every batch succeeds."""

import argparse
import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client import models

from src.config import Settings
from src.models import Passage
from src.retrieval.hybrid import HybridIndex, bm25

log = logging.getLogger(__name__)


def passages(root: Path, size: int, overlap: int) -> Iterator[Passage]:
    if not root.is_dir():
        raise ValueError(f"Corpus directory does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        for number, start in enumerate(range(0, len(text), size - overlap), start=1):
            section = text[start : start + size].strip()
            if section:
                yield Passage(source=path.relative_to(root).as_posix(), chunk=number, text=section)
            if start + size >= len(text):
                break


def publish(root: Path, index: HybridIndex) -> str:
    cfg, client = index.settings, index.client
    stream = passages(root, cfg.chunk_size, cfg.chunk_overlap)
    version = f"{cfg.collection}-{uuid4().hex}"
    count = 0
    while batch := list(islice(stream, 32)):
        vectors = index.embeddings.embed_documents([p.text for p in batch])
        if count == 0:
            client.create_collection(
                version,
                vectors_config={
                    cfg.dense_name: models.VectorParams(
                        size=len(vectors[0]), distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            log.info("Building collection %s", version)
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{p.source}#{p.chunk}")),
                vector={cfg.dense_name: vector, "bm25": bm25(p.text)},
                payload=p.model_dump(),
            )
            for p, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(version, points, wait=True)
        count += len(points)
    if not count:
        raise ValueError("Corpus contains no non-empty Markdown or text passages.")

    updates = []
    if any(alias.alias_name == cfg.collection for alias in client.get_aliases().aliases):
        updates.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=cfg.collection))
        )
    updates.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=version, alias_name=cfg.collection)
        )
    )
    client.update_collection_aliases(updates)
    log.info("Published %d passages to %s via %s", count, version, cfg.collection)
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    index = HybridIndex(Settings.load())
    try:
        publish(args.directory, index)
    finally:
        index.client.close()


if __name__ == "__main__":
    main()
