"""Connection and index settings for OpenSearch/Elasticsearch.

Credential pattern follows knowhow-elasticsearch/os_settings.py:
  - Separate host / port (not combined URL)
  - opensearch-py is compatible with ES 7.x (forked from ES 7.10)

All settings can be overridden via environment variables or by passing
values directly to ``ConnectionConfig``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConnectionConfig:
    """Connection and index configuration for an OpenSearch/ES cluster."""

    host: str = "localhost"
    port: int = 9200
    user: str = "admin"
    password: str = "admin"
    use_ssl: bool = True
    verify_certs: bool = False
    ssl_show_warn: bool = False
    ca_certs: Optional[str] = None
    bulk_chunk: int = 500

    @property
    def http_auth(self) -> Optional[tuple[str, str]]:
        if self.user and self.password:
            return (self.user, self.password)
        return None

    @property
    def hosts(self) -> list[dict]:
        """Return hosts list in the format opensearch-py expects."""
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
        cfg.use_ssl = ssl_env.lower() == "true"

    verify_env = os.getenv("OPENSEARCH_VERIFY_CERTS")
    if verify_env is not None:
        cfg.verify_certs = verify_env.lower() == "true"

    ca_certs = os.getenv("OPENSEARCH_CA_CERTS")
    if ca_certs:
        cfg.ca_certs = ca_certs

    bulk_chunk = os.getenv("OPENSEARCH_BULK_CHUNK")
    if bulk_chunk:
        cfg.bulk_chunk = int(bulk_chunk)

    # Explicit overrides layer
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise TypeError(f"Unknown config key: {key!r}")

    return cfg
