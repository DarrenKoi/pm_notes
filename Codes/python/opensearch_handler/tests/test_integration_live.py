from __future__ import annotations

import uuid

import pytest

from client import create_client
from document import bulk_index, get_document, index_document, update_document
from index import create_index, delete_index, index_exists, refresh_index
from search import count_documents, match_search


@pytest.mark.integration
def test_live_cluster_smoke() -> None:
    client = create_client()
    assert client.ping() is True

    index_name = f"test-opensearch-handler-{uuid.uuid4().hex[:8]}"

    assert index_exists(client, index_name) is False
    create_index(
        client,
        index_name,
        mappings={
            "properties": {
                "title": {"type": "text"},
                "category": {"type": "keyword"},
                "content": {"type": "text"},
            }
        },
    )

    try:
        index_document(
            client,
            index_name,
            {"title": "first", "category": "note", "content": "hello opensearch"},
            doc_id="1",
            refresh="wait_for",
        )

        success, errors = bulk_index(
            client,
            index_name,
            docs=[
                {"id": "2", "title": "second", "category": "note", "content": "hello world"},
                {"id": "3", "title": "third", "category": "log", "content": "search me"},
            ],
            id_field="id",
            refresh=True,
        )
        assert success == 2
        assert errors == []

        update_document(client, index_name, "1", {"content": "hello updated"}, refresh="wait_for")
        refresh_index(client, index_name)

        got = get_document(client, index_name, "1")
        assert got["_source"]["content"] == "hello updated"

        result = match_search(client, index_name, "content", "hello", size=10)
        assert result["hits"]["total"]["value"] >= 2

        count = count_documents(client, index_name)
        assert count["count"] == 3
    finally:
        if index_exists(client, index_name):
            delete_index(client, index_name)
