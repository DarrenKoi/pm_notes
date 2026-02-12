from __future__ import annotations

from index import (
    create_index,
    get_index_mapping,
    get_index_settings,
    index_exists,
    refresh_index,
    update_index_settings,
)


class DummyIndices:
    def __init__(self):
        self.calls = []

    def exists(self, **kwargs):
        self.calls.append(("exists", kwargs))
        return True

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return {"acknowledged": True, **kwargs}

    def get_settings(self, **kwargs):
        self.calls.append(("get_settings", kwargs))
        return kwargs

    def get_mapping(self, **kwargs):
        self.calls.append(("get_mapping", kwargs))
        return kwargs

    def put_settings(self, **kwargs):
        self.calls.append(("put_settings", kwargs))
        return kwargs

    def refresh(self, **kwargs):
        self.calls.append(("refresh", kwargs))
        return kwargs


class DummyClient:
    def __init__(self):
        self.indices = DummyIndices()


def test_index_helpers():
    client = DummyClient()

    assert index_exists(client, "notes") is True

    create_response = create_index(
        client,
        "notes",
        settings={"index.knn": True},
        mappings={"properties": {"title": {"type": "text"}}},
        shards=2,
        replicas=1,
    )
    assert create_response["index"] == "notes"
    body = create_response["body"]
    assert body["settings"]["number_of_shards"] == 2
    assert body["settings"]["number_of_replicas"] == 1
    assert body["settings"]["index.knn"] is True

    assert get_index_settings(client, "notes") == {"index": "notes"}
    assert get_index_mapping(client, "notes") == {"index": "notes"}
    assert update_index_settings(client, "notes", {"refresh_interval": "1s"}) == {
        "index": "notes",
        "body": {"index": {"refresh_interval": "1s"}},
    }
    assert refresh_index(client, "notes") == {"index": "notes"}
