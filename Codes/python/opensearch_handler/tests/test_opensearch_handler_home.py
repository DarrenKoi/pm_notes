"""Home-safe unit tests for the OpenSearch handler."""

from __future__ import annotations

from typing import Any

import pytest

from opensearch_handler.client import create_client
from opensearch_handler.connection_settings import ConnectionConfig, load_config
from opensearch_handler.document import bulk_actions
from opensearch_handler.lifecycle import format_rollover_index
from opensearch_handler.search import hybrid_search, knn_search


class OpenSearchClient:
    __module__ = "opensearchpy.client"

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append(kwargs)
        return {"hits": {"hits": []}}


def test_load_config_supports_opensearch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_HOST", "search.internal")
    monkeypatch.setenv("OPENSEARCH_PORT", "9243")
    monkeypatch.setenv("OPENSEARCH_USER", "admin")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "secret")
    monkeypatch.setenv("OPENSEARCH_VERIFY_CERTS", "true")

    config = load_config()

    assert config.host == "search.internal"
    assert config.port == 9243
    assert config.http_auth == ("admin", "secret")
    assert config.verify_certs is True


def test_load_config_allows_no_auth() -> None:
    config = load_config(user=None, password=None)
    assert config.http_auth is None


def test_create_client_builds_opensearch_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenSearch:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("opensearch_handler.client.OpenSearch", FakeOpenSearch)

    create_client(
        ConnectionConfig(
            host="opensearch.internal",
            port=9243,
            user="admin",
            password="secret",
            verify_certs=True,
        )
    )

    assert captured["hosts"] == [{"host": "opensearch.internal", "port": 9243, "scheme": "https"}]
    assert captured["http_auth"] == ("admin", "secret")
    assert captured["verify_certs"] is True


def test_knn_search_uses_opensearch_body_shape() -> None:
    client = OpenSearchClient()

    knn_search(
        client,
        "docs",
        "embedding",
        [0.1, 0.2, 0.3],
        k=2,
        filters=[{"term": {"tenant_id": "team-a"}}],
    )

    body = client.search_calls[0]["body"]
    assert body["query"]["bool"]["must"][0]["knn"]["embedding"]["vector"] == [0.1, 0.2, 0.3]
    assert body["query"]["bool"]["filter"] == [{"term": {"tenant_id": "team-a"}}]


def test_hybrid_search_uses_opensearch_query_shape() -> None:
    client = OpenSearchClient()

    hybrid_search(
        client,
        "docs",
        query="vector search",
        text_field="content",
        vector_field="embedding",
        vector=[0.1, 0.2, 0.3],
        filters=[{"term": {"tenant_id": "team-a"}}],
    )

    body = client.search_calls[0]["body"]
    assert body["query"]["bool"]["minimum_should_match"] == 1
    assert body["query"]["bool"]["filter"] == [{"term": {"tenant_id": "team-a"}}]


def test_bulk_actions_uses_opensearch_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_bulk(client: Any, actions: Any, **kwargs: Any) -> tuple[int, list]:
        captured["client"] = client
        captured["actions"] = list(actions)
        captured["kwargs"] = kwargs
        return (1, [])

    monkeypatch.setattr("opensearch_handler.document.helpers.bulk", fake_bulk)

    result = bulk_actions(object(), [{"_index": "docs", "_source": {"id": "1"}}])

    assert result == (1, [])
    assert captured["kwargs"]["chunk_size"] == 500


def test_format_rollover_index_uses_zero_padded_generation() -> None:
    assert format_rollover_index("logs", 1) == "logs-000001"
    with pytest.raises(ValueError):
        format_rollover_index("logs", 0)
