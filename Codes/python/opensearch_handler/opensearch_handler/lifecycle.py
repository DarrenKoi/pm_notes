"""Template, alias, and rollover helpers for OpenSearch."""

from __future__ import annotations

from typing import Any


def format_rollover_index(prefix: str, generation: int) -> str:
    """Return a zero-padded rollover index name such as ``logs-000001``."""
    if generation < 1:
        raise ValueError("generation must be >= 1")
    return f"{prefix}-{generation:06d}"


def put_index_template(
    client: Any,
    name: str,
    body: dict[str, Any],
    prefer_composable: bool = True,
) -> dict[str, Any]:
    """Create or update an index template."""
    if prefer_composable and hasattr(client.indices, "put_index_template"):
        return client.indices.put_index_template(name=name, body=body)
    return client.indices.put_template(name=name, body=body)


def bootstrap_aliases(
    client: Any,
    first_index: str,
    write_alias: str,
    read_alias: str | None = None,
) -> dict[str, Any]:
    """Create a first-generation index and attach write/read aliases."""
    aliases: dict[str, Any] = {write_alias: {"is_write_index": True}}
    if read_alias:
        aliases[read_alias] = {}

    return client.indices.create(index=first_index, body={"aliases": aliases})


def rollover_alias(
    client: Any,
    write_alias: str,
    conditions: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Roll over a write alias to the next numbered index."""
    body = {"conditions": conditions}
    return client.indices.rollover(alias=write_alias, body=body, dry_run=dry_run)
