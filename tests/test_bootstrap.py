from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from src.ingestion.bootstrap import needs_index


def index_with(info=None, error=None):
    index = Mock()
    index.settings = SimpleNamespace(collection="docs", dense_name="dense-x")
    if error:
        index.client.get_collection.side_effect = error
    else:
        index.client.get_collection.return_value = info
    return index


def info(points, vectors):
    return SimpleNamespace(
        points_count=points, config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
    )


def missing(status):
    return UnexpectedResponse(status, "x", b"", {})


def test_missing_collection_needs_index():
    assert needs_index(index_with(error=missing(404)))


def test_empty_or_other_embedding_space_needs_index():
    assert needs_index(index_with(info(0, {"dense-x": 1})))
    assert needs_index(index_with(info(5, {"dense-other": 1})))


def test_populated_collection_is_kept():
    assert not needs_index(index_with(info(5, {"dense-x": 1})))


def test_other_server_errors_are_not_treated_as_missing():
    with pytest.raises(UnexpectedResponse):
        needs_index(index_with(error=missing(500)))
