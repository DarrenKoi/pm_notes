"""Index management operations."""

from typing import Any, Optional


def index_exists(client: Any, name: str) -> bool:
    """Check whether an index exists."""
    return client.indices.exists(index=name)


def create_index(
    client: Any,
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


def delete_index(client: Any, name: str) -> dict:
    """Delete an index."""
    return client.indices.delete(index=name)


def get_index_settings(client: Any, name: str) -> dict:
    """Return current settings for an index."""
    return client.indices.get_settings(index=name)


def get_index_mapping(client: Any, name: str) -> dict:
    """Return mappings for an index."""
    return client.indices.get_mapping(index=name)


def update_index_settings(
    client: Any,
    name: str,
    settings: dict[str, Any],
) -> dict:
    """Update dynamic settings on an existing index."""
    return client.indices.put_settings(index=name, body={"index": settings})


def refresh_index(client: Any, name: str) -> dict:
    """Force-refresh an index so recent writes are searchable immediately."""
    return client.indices.refresh(index=name)
