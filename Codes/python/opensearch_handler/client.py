"""Client factory for OpenSearch / Elasticsearch connections."""

from typing import Any, Optional

from .connection_settings import ConnectionConfig, load_config

try:
    from opensearchpy import OpenSearch
except ModuleNotFoundError:  # pragma: no cover
    OpenSearch = None  # type: ignore[assignment]

try:
    from elasticsearch import Elasticsearch
except ModuleNotFoundError:  # pragma: no cover
    Elasticsearch = None  # type: ignore[assignment]


def _resolve_client_class() -> type[Any]:
    """Return the first available 7.x-compatible client class."""
    if OpenSearch is not None:
        return OpenSearch
    if Elasticsearch is not None:
        return Elasticsearch
    raise ModuleNotFoundError(
        "Either 'opensearch-py' or 'elasticsearch' must be installed to create a client."
    )


def create_client(
    config: Optional[ConnectionConfig] = None,
    **overrides,
) -> Any:
    """Create and return an OpenSearch client.

    Args:
        config: An explicit :class:`ConnectionConfig`.  When ``None``,
            one is built via :func:`load_config` (env vars + *overrides*).
        **overrides: Passed to :func:`load_config` when *config* is ``None``.

    Returns:
        A configured OpenSearch/Elasticsearch client instance.
    """
    if config is None:
        config = load_config(**overrides)

    kwargs: dict = {
        "hosts": config.hosts,
        "use_ssl": config.use_ssl,
        "verify_certs": config.verify_certs,
        "ssl_show_warn": config.ssl_show_warn,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "retry_on_timeout": config.retry_on_timeout,
        "http_compress": config.http_compress,
    }

    http_auth = config.http_auth
    if http_auth:
        kwargs["http_auth"] = http_auth

    if config.ca_certs:
        kwargs["ca_certs"] = config.ca_certs

    client_cls = _resolve_client_class()
    return client_cls(**kwargs)
