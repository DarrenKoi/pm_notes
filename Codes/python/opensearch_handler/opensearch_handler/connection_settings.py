"""Connection settings for OpenSearch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


@dataclass
class ConnectionConfig:
    """Connection configuration for an OpenSearch cluster."""

    host: str = "localhost"
    port: int = 9200
    user: Optional[str] = "admin"
    password: Optional[str] = "admin"
    use_ssl: bool = True
    verify_certs: bool = False
    ssl_show_warn: bool = False
    ca_certs: Optional[str] = None
    bulk_chunk: int = 500
    timeout: int = 30
    max_retries: int = 3
    retry_on_timeout: bool = True
    http_compress: bool = True

    def __post_init__(self) -> None:
        self._validate_auth()

    def _validate_auth(self) -> None:
        if bool(self.user) != bool(self.password):
            raise ValueError(
                "ConnectionConfig requires both user and password when basic auth "
                "is configured."
            )

    @property
    def http_auth(self) -> Optional[tuple[str, str]]:
        if self.user and self.password:
            return (self.user, self.password)
        return None

    @property
    def hosts(self) -> list[dict]:
        """Return hosts in the format expected by opensearch-py."""
        scheme = "https" if self.use_ssl else "http"
        return [{"host": self.host, "port": self.port, "scheme": scheme}]


def load_config(**overrides) -> ConnectionConfig:
    """Build a ConnectionConfig with env-var and keyword overrides."""
    cfg = ConnectionConfig()

    host = os.getenv("OPENSEARCH_HOST")
    if host:
        cfg.host = host

    port = os.getenv("OPENSEARCH_PORT")
    if port:
        cfg.port = int(port)

    user = os.getenv("OPENSEARCH_USER")
    if user is not None:
        cfg.user = user or None

    password = os.getenv("OPENSEARCH_PASSWORD")
    if password is not None:
        cfg.password = password or None

    ssl_env = os.getenv("OPENSEARCH_USE_SSL")
    if ssl_env is not None:
        cfg.use_ssl = _parse_bool(ssl_env)

    verify_env = os.getenv("OPENSEARCH_VERIFY_CERTS")
    if verify_env is not None:
        cfg.verify_certs = _parse_bool(verify_env)

    ssl_show_warn = os.getenv("OPENSEARCH_SSL_SHOW_WARN")
    if ssl_show_warn is not None:
        cfg.ssl_show_warn = _parse_bool(ssl_show_warn)

    ca_certs = os.getenv("OPENSEARCH_CA_CERTS")
    if ca_certs:
        cfg.ca_certs = ca_certs

    bulk_chunk = os.getenv("OPENSEARCH_BULK_CHUNK")
    if bulk_chunk:
        cfg.bulk_chunk = int(bulk_chunk)

    timeout = os.getenv("OPENSEARCH_TIMEOUT")
    if timeout:
        cfg.timeout = int(timeout)

    max_retries = os.getenv("OPENSEARCH_MAX_RETRIES")
    if max_retries:
        cfg.max_retries = int(max_retries)

    retry_on_timeout = os.getenv("OPENSEARCH_RETRY_ON_TIMEOUT")
    if retry_on_timeout is not None:
        cfg.retry_on_timeout = _parse_bool(retry_on_timeout)

    http_compress = os.getenv("OPENSEARCH_HTTP_COMPRESS")
    if http_compress is not None:
        cfg.http_compress = _parse_bool(http_compress)

    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            continue
        raise TypeError(f"Unknown config key: {key!r}")

    cfg._validate_auth()
    return cfg
