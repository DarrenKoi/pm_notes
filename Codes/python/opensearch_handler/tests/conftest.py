from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make package importable when running tests from module root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that require a running OpenSearch/Elasticsearch cluster",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("OPENSEARCH_RUN_INTEGRATION") == "1":
        return

    skip_integration = pytest.mark.skip(
        reason="Set OPENSEARCH_RUN_INTEGRATION=1 to run integration tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
