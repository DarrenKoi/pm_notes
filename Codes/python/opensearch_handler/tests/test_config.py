from __future__ import annotations

import pytest

from config import ConnectionConfig, load_config


def test_load_config_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_HOST", "example.com")
    monkeypatch.setenv("OPENSEARCH_PORT", "443")
    monkeypatch.setenv("OPENSEARCH_USER", "u")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "p")
    monkeypatch.setenv("OPENSEARCH_USE_SSL", "yes")
    monkeypatch.setenv("OPENSEARCH_VERIFY_CERTS", "no")
    monkeypatch.setenv("OPENSEARCH_RETRY_ON_TIMEOUT", "on")
    monkeypatch.setenv("OPENSEARCH_HTTP_COMPRESS", "off")
    monkeypatch.setenv("OPENSEARCH_TIMEOUT", "99")
    monkeypatch.setenv("OPENSEARCH_MAX_RETRIES", "5")
    monkeypatch.setenv("OPENSEARCH_BULK_CHUNK", "1000")

    cfg = load_config()

    assert cfg.host == "example.com"
    assert cfg.port == 443
    assert cfg.http_auth == ("u", "p")
    assert cfg.use_ssl is True
    assert cfg.verify_certs is False
    assert cfg.retry_on_timeout is True
    assert cfg.http_compress is False
    assert cfg.timeout == 99
    assert cfg.max_retries == 5
    assert cfg.bulk_chunk == 1000


def test_invalid_bool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_USE_SSL", "maybe")
    with pytest.raises(ValueError):
        load_config()


def test_unknown_override_key_raises() -> None:
    with pytest.raises(TypeError):
        load_config(not_a_real_key=True)


def test_hosts_property_uses_scheme() -> None:
    cfg = ConnectionConfig(host="localhost", port=9200, use_ssl=False)
    assert cfg.hosts == [{"host": "localhost", "port": 9200, "scheme": "http"}]
