from __future__ import annotations

import document as document_module
from document import (
    bulk_index,
    delete_document,
    index_document,
    update_document,
    upsert_document,
)


class DummyClient:
    def index(self, **kwargs):
        return kwargs

    def update(self, **kwargs):
        return kwargs

    def delete(self, **kwargs):
        return kwargs


def test_document_crud_params():
    client = DummyClient()

    assert index_document(client, "idx", {"a": 1}, doc_id="1", refresh="wait_for") == {
        "index": "idx",
        "body": {"a": 1},
        "id": "1",
        "refresh": "wait_for",
    }

    assert update_document(client, "idx", "1", {"a": 2}, refresh="true") == {
        "index": "idx",
        "id": "1",
        "body": {"doc": {"a": 2}},
        "refresh": "true",
    }

    assert upsert_document(client, "idx", "2", {"a": 3}) == {
        "index": "idx",
        "id": "2",
        "body": {"doc": {"a": 3}, "doc_as_upsert": True},
    }

    assert delete_document(client, "idx", "2", refresh="wait_for") == {
        "index": "idx",
        "id": "2",
        "refresh": "wait_for",
    }


def test_bulk_index_uses_helpers_bulk(monkeypatch):
    captured = {}

    def fake_bulk(client, actions, **kwargs):
        captured["actions"] = list(actions)
        captured.update(kwargs)
        return (len(captured["actions"]), [])

    monkeypatch.setattr(document_module.helpers, "bulk", fake_bulk)

    docs = [{"id": "1", "title": "a"}, {"id": "2", "title": "b"}]
    success, errors = bulk_index(
        client=object(),
        index="idx",
        docs=docs,
        id_field="id",
        chunk_size=100,
        refresh=True,
        raise_on_error=False,
    )

    assert success == 2
    assert errors == []
    assert captured["chunk_size"] == 100
    assert captured["refresh"] is True
    assert captured["raise_on_error"] is False
    assert captured["actions"][0]["_id"] == "1"
