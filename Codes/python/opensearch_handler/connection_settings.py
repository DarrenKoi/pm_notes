"""Connection and index settings for OpenSearch."""

Credential pattern follows the local OS settings convention:
  - Separate host / port (not combined URL)
  - opensearch-py is used as the client library for OpenSearch

All settings can be overridden via environment variables or by passing
values directly to ``ConnectionConfig``.
"""

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
    """Connection and index configuration for an OpenSearch cluster."""

    host: str = "localhost"
    port: int = 9200
    user: str = "admin"
    password: str = "admin"
    use_ssl: bool = True
    verify_certs: bool = False
    ssl_show_warn: bool = False
    ca_certs: Optional[str] = None
    bulk_chunk: int = 500
    timeout: int = 30
    max_retries: int = 3
    retry_on_timeout: bool = True
    http_compress: bool = True

    @property
    def http_auth(self) -> Optional[tuple[str, str]]:
        if self.user and self.password:
            return (self.user, self.password)
        return None

    @property
    def hosts(self) -> list[dict]:
        """Return hosts list in the format expected by opensearch-py."""
        scheme = "https" if self.use_ssl else "http"
        return [{"host": self.host, "port": self.port, "scheme": scheme}]


def load_config(**overrides) -> ConnectionConfig:
    """Build a ConnectionConfig with env-var and keyword overrides.

    Resolution order (later wins):
      1. Dataclass defaults
      2. Environment variables (``OPENSEARCH_HOST``, etc.)
      3. Explicit keyword arguments

    Supported env vars:
      - OPENSEARCH_HOST / OPENSEARCH_PORT
      - OPENSEARCH_USER / OPENSEARCH_PASSWORD
      - OPENSEARCH_USE_SSL  ("true"/"false")
      - OPENSEARCH_VERIFY_CERTS  ("true"/"false")
      - OPENSEARCH_CA_CERTS
      - OPENSEARCH_BULK_CHUNK
      - OPENSEARCH_TIMEOUT
      - OPENSEARCH_MAX_RETRIES
      - OPENSEARCH_RETRY_ON_TIMEOUT ("true"/"false")
      - OPENSEARCH_HTTP_COMPRESS ("true"/"false")
    """
    cfg = ConnectionConfig()

    # Env-var layer
    host = os.getenv("OPENSEARCH_HOST")
    if host:
        cfg.host = host

    port = os.getenv("OPENSEARCH_PORT")
    if port:
        cfg.port = int(port)

    user = os.getenv("OPENSEARCH_USER")
    if user:
        cfg.user = user

    password = os.getenv("OPENSEARCH_PASSWORD")
    if password:
        cfg.password = password

    ssl_env = os.getenv("OPENSEARCH_USE_SSL")
    if ssl_env is not None:
        cfg.use_ssl = _parse_bool(ssl_env)

    verify_env = os.getenv("OPENSEARCH_VERIFY_CERTS")
    if verify_env is not None:
        cfg.verify_certs = _parse_bool(verify_env)

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

    # Explicit overrides layer
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise TypeError(f"Unknown config key: {key!r}")

    return cfg
