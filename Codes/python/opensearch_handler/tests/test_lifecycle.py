from __future__ import annotations

import pytest

from lifecycle import (
    bootstrap_aliases,
    detect_cluster_flavor,
    put_index_template,
    put_lifecycle_policy,
    rollover_alias,
)


class DummyIndices:
    def __init__(self):
        self.calls = []

    def put_index_template(self, **kwargs):
        self.calls.append(("put_index_template", kwargs))
        return {"api": "put_index_template", **kwargs}

    def put_template(self, **kwargs):
        self.calls.append(("put_template", kwargs))
        return {"api": "put_template", **kwargs}

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return kwargs

    def rollover(self, **kwargs):
        self.calls.append(("rollover", kwargs))
        return kwargs


class DummyTransport:
    def __init__(self):
        self.calls = []

    def perform_request(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class DummyClient:
    def __init__(self, info_body):
        self._info_body = info_body
        self.indices = DummyIndices()
        self.transport = DummyTransport()

    def info(self):
        return self._info_body


def test_detect_cluster_flavor_opensearch_distribution():
    client = DummyClient({"version": {"distribution": "opensearch", "number": "2.11.0"}})
    assert detect_cluster_flavor(client) == "opensearch"


def test_detect_cluster_flavor_elasticsearch_tagline():
    client = DummyClient(
        {"version": {"number": "7.10.2"}, "tagline": "You Know, for Search"}
    )
    assert detect_cluster_flavor(client) == "elasticsearch"


def test_put_index_template_uses_composable_when_available():
    client = DummyClient({"version": {"distribution": "opensearch"}})
    response = put_index_template(client, "tpl", {"index_patterns": ["logs-*"]})
    assert response["api"] == "put_index_template"
    assert response["name"] == "tpl"


def test_put_index_template_legacy_fallback():
    client = DummyClient({"version": {"distribution": "opensearch"}})
    response = put_index_template(
        client,
        "tpl",
        {"index_patterns": ["logs-*"]},
        prefer_composable=False,
    )
    assert response["api"] == "put_template"


def test_bootstrap_aliases_adds_write_and_read_aliases():
    client = DummyClient({"version": {"distribution": "opensearch"}})
    response = bootstrap_aliases(client, "notes-000001", "notes-write", "notes-read")

    assert response["index"] == "notes-000001"
    aliases = response["body"]["aliases"]
    assert aliases["notes-write"]["is_write_index"] is True
    assert aliases["notes-read"] == {}


def test_rollover_alias_forwards_conditions():
    client = DummyClient({"version": {"distribution": "opensearch"}})
    response = rollover_alias(client, "notes-write", {"max_age": "7d"}, dry_run=True)

    assert response["alias"] == "notes-write"
    assert response["body"] == {"conditions": {"max_age": "7d"}}
    assert response["dry_run"] is True


def test_put_lifecycle_policy_routes_to_ism_when_opensearch():
    client = DummyClient({"version": {"distribution": "opensearch", "number": "2.11.0"}})
    body = {"policy": {"description": "retention"}}

    response = put_lifecycle_policy(client, "notes", body)

    assert response["url"] == "/_plugins/_ism/policies/notes"


def test_put_lifecycle_policy_routes_to_ilm_when_elasticsearch():
    client = DummyClient({"version": {"number": "7.10.2"}, "tagline": "You Know, for Search"})
    body = {"policy": {"phases": {}}}

    response = put_lifecycle_policy(client, "notes", body)

    assert response["url"] == "/_ilm/policy/notes"


def test_put_lifecycle_policy_raises_when_flavor_unknown():
    client = DummyClient({"version": {"number": "1.0.0"}})

    with pytest.raises(ValueError):
        put_lifecycle_policy(client, "notes", {"policy": {}})
