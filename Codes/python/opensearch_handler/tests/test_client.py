from __future__ import annotations

from dataclasses import asdict

import client as client_module
from config import ConnectionConfig


class DummyOpenSearch:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_create_client_passes_expected_kwargs(monkeypatch):
    captured = {}

    def fake_opensearch(**kwargs):
        captured.update(kwargs)
        return DummyOpenSearch(**kwargs)

    monkeypatch.setattr(client_module, "OpenSearch", fake_opensearch)

    cfg = ConnectionConfig(
        host="os.local",
        port=9201,
        user="admin",
        password="admin",
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        ca_certs="/tmp/ca.pem",
        timeout=10,
        max_retries=7,
        retry_on_timeout=True,
        http_compress=True,
    )

    result = client_module.create_client(config=cfg)

    assert isinstance(result, DummyOpenSearch)
    assert captured["hosts"] == [{"host": "os.local", "port": 9201, "scheme": "https"}]
    assert captured["http_auth"] == ("admin", "admin")
    assert captured["ca_certs"] == "/tmp/ca.pem"
    assert captured["timeout"] == 10
    assert captured["max_retries"] == 7
    assert captured["retry_on_timeout"] is True
    assert captured["http_compress"] is True


def test_create_client_uses_overrides_when_no_config(monkeypatch):
    class Dummy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(client_module, "OpenSearch", lambda **kwargs: Dummy(**kwargs))

    client = client_module.create_client(host="127.0.0.1", port=9200, use_ssl=False)
    assert client.kwargs["hosts"][0]["scheme"] == "http"
