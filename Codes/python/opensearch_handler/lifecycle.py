"""Lifecycle, template, alias, and rollover helpers for OpenSearch."""

from typing import Any


def detect_cluster_flavor(client: Any) -> str:
    """Detect cluster flavor.

    Detection uses ``client.info()`` and falls back to ``unknown`` if the
    response does not clearly identify OpenSearch.
    """
    info = client.info()
    version = info.get("version", {})

    distribution = str(version.get("distribution", "")).lower()
    if distribution == "opensearch":
        return "opensearch"

    tagline = str(info.get("tagline", "")).lower()
    if "opensearch" in tagline:
        return "opensearch"

    return "unknown"


def put_index_template(
    client: Any,
    name: str,
    body: dict[str, Any],
    prefer_composable: bool = True,
) -> dict[str, Any]:
    """Create/update index template with compatibility fallback.

    - Composable template API: ``indices.put_index_template``
    - Legacy template API: ``indices.put_template``
    """
    if prefer_composable and hasattr(client.indices, "put_index_template"):
        return client.indices.put_index_template(name=name, body=body)
    return client.indices.put_template(name=name, body=body)


def bootstrap_aliases(
    client: Any,
    first_index: str,
    write_alias: str,
    read_alias: str | None = None,
) -> dict[str, Any]:
    """Create first generation index and attach write/read aliases."""
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
    """Rollover a write alias based on conditions."""
    body = {"conditions": conditions}
    return client.indices.rollover(alias=write_alias, body=body, dry_run=dry_run)


def put_lifecycle_policy(
    client: Any,
    policy_name: str,
    policy_body: dict[str, Any],
    flavor: str = "auto",
) -> dict[str, Any]:
    """Create/update lifecycle policy using OpenSearch ISM.

    Args:
        client: OpenSearch client.
        policy_name: Policy id.
        policy_body: Raw request body for target API.
            - OpenSearch ISM expects ``{"policy": {...states...}}``.
        flavor: ``auto`` or ``opensearch``.

    Returns:
        API response dict.
    """
    selected = detect_cluster_flavor(client) if flavor == "auto" else flavor

    if selected == "opensearch":
        path = f"/_plugins/_ism/policies/{policy_name}"
    else:
        raise ValueError(
            "Could not determine cluster flavor. "
            "Set flavor='opensearch'."
        )

    return client.transport.perform_request(method="PUT", url=path, body=policy_body)
