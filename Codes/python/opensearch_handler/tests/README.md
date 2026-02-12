# Tests for `opensearch_handler`

This test suite supports two workflows:

- Home/offline: run fast unit tests only (no OpenSearch cluster required).
- Workplace/cluster: run integration tests against your running OpenSearch.

## 1) Install test deps

```bash
cd Codes/python/opensearch_handler
pip install -r requirements-dev.txt
```

## 2) Run unit tests (default)

```bash
cd Codes/python/opensearch_handler
pytest
```

## 3) Run integration test (only when cluster is running)

Set your connection env vars first (example):

```bash
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASSWORD=admin
export OPENSEARCH_USE_SSL=false
```

Then enable integration tests:

```bash
cd Codes/python/opensearch_handler
OPENSEARCH_RUN_INTEGRATION=1 pytest -m integration -v
```

By default, integration tests are skipped unless `OPENSEARCH_RUN_INTEGRATION=1`.
