"""Client factory for OpenSearch / Elasticsearch connections."""

from __future__ import annotations

from typing import Optional

from opensearchpy import OpenSearch

from .config import ConnectionConfig, load_config


def create_client(
    config: Optional[ConnectionConfig] = None,
    **overrides,
) -> OpenSearch:
    """Create and return an OpenSearch client.

    Args:
        config: An explicit :class:`ConnectionConfig`.  When ``None``,
            one is built via :func:`load_config` (env vars + *overrides*).
        **overrides: Passed to :func:`load_config` when *config* is ``None``.

    Returns:
        A configured :class:`opensearchpy.OpenSearch` instance.
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

    return OpenSearch(**kwargs)
