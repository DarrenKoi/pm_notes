"""Client factory for OpenSearch connections."""

from __future__ import annotations

from typing import Optional

from .connection_settings import ConnectionConfig, load_config
from opensearchpy import OpenSearch


def create_client(
    config: Optional[ConnectionConfig] = None,
    **overrides,
) -> OpenSearch:
    """Create and return a configured OpenSearch client."""
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

    if config.http_auth:
        kwargs["http_auth"] = config.http_auth

    if config.ca_certs:
        kwargs["ca_certs"] = config.ca_certs

    return OpenSearch(**kwargs)
