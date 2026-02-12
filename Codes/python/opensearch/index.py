"""Index management operations."""

from __future__ import annotations

from typing import Any, Optional

from opensearchpy import OpenSearch


def index_exists(client: OpenSearch, name: str) -> bool:
    """Check whether an index exists."""
    return client.indices.exists(index=name)


def create_index(
    client: OpenSearch,
    name: str,
    mappings: Optional[dict[str, Any]] = None,
    settings: Optional[dict[str, Any]] = None,
    shards: int = 1,
    replicas: int = 0,
    refresh_interval: str = "30s",
) -> dict:
    """Create an index with optional mappings and settings.

    ``shards``, ``replicas``, and ``refresh_interval`` are applied as
    defaults — they won't overwrite values already present in *settings*.
    """
    settings = dict(settings) if settings else {}
    settings.setdefault("number_of_shards", shards)
    settings.setdefault("number_of_replicas", replicas)
    settings.setdefault("refresh_interval", refresh_interval)

    body: dict[str, Any] = {"settings": settings}
    if mappings:
        body["mappings"] = mappings

    return client.indices.create(index=name, body=body)


def delete_index(client: OpenSearch, name: str) -> dict:
    """Delete an index."""
    return client.indices.delete(index=name)


def get_index_settings(client: OpenSearch, name: str) -> dict:
    """Return current settings for an index."""
    return client.indices.get_settings(index=name)


def update_index_settings(
    client: OpenSearch,
    name: str,
    settings: dict[str, Any],
) -> dict:
    """Update dynamic settings on an existing index."""
    return client.indices.put_settings(index=name, body={"index": settings})
