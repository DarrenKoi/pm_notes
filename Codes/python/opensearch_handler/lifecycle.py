"""Lifecycle, template, alias, and rollover helpers for OS/ES compatibility."""

from typing import Any


def detect_cluster_flavor(client: Any) -> str:
    """Detect cluster flavor: ``opensearch`` or ``elasticsearch``.

    Detection uses ``client.info()`` and falls back to ``unknown`` if the
    response does not clearly identify either distribution.
    """
    info = client.info()
    version = info.get("version", {})

    distribution = str(version.get("distribution", "")).lower()
    if distribution == "opensearch":
        return "opensearch"

    tagline = str(info.get("tagline", "")).lower()
    if "opensearch" in tagline:
        return "opensearch"

    number = str(version.get("number", ""))
    if number.startswith("7") or "you know, for search" in tagline:
        return "elasticsearch"

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
    """Create/update lifecycle policy using ISM (OS) or ILM (ES7).

    Args:
        client: OpenSearch/Elasticsearch client.
        policy_name: Policy id.
        policy_body: Raw request body for target API.
            - OpenSearch ISM expects ``{"policy": {...states...}}``.
            - Elasticsearch ILM expects ``{"policy": {...phases...}}``.
        flavor: ``auto``, ``opensearch``, or ``elasticsearch``.

    Returns:
        API response dict.
    """
    selected = detect_cluster_flavor(client) if flavor == "auto" else flavor

    if selected == "opensearch":
        path = f"/_plugins/_ism/policies/{policy_name}"
    elif selected == "elasticsearch":
        path = f"/_ilm/policy/{policy_name}"
    else:
        raise ValueError(
            "Could not determine cluster flavor. "
            "Set flavor='opensearch' or flavor='elasticsearch'."
        )

    return client.transport.perform_request(method="PUT", url=path, body=policy_body)
