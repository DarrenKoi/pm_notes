# opensearch_handler

`opensearch_handler` is a lightweight Python helper package for OpenSearch.

It focuses on the OpenSearch code paths most often used in production:

- connection/config loading
- index creation and settings
- aliases and rollover-style index naming
- document CRUD and bulk indexing
- text search, aggregation, vector search, and hybrid search
- raw query escape hatch for OpenSearch-specific features

Lifecycle retention policy management is intentionally out of scope here. The package handles index naming, templates, aliases, rollover conventions, and application-side queries and writes.

## Installation

```bash
cd Codes/python/opensearch_handler
python3 -m pip install -r requirements.txt
```

If you want to install it as a reusable package from this folder:

```bash
python3 -m pip install -e .
```

## Configuration

Use the OpenSearch environment convention:

```bash
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASSWORD=admin
export OPENSEARCH_USE_SSL=true
export OPENSEARCH_VERIFY_CERTS=false
```

## Common Patterns

### 1. Standard client

```python
from opensearch_handler import create_client, load_config

client = create_client(load_config())
```

### 2. Explicit config object

```python
from opensearch_handler import ConnectionConfig, create_client

client = create_client(ConnectionConfig(
    host="opensearch.internal",
    port=9200,
    user="admin",
    password="admin",
))
```

### 3. Unsupported features

When OpenSearch needs a request not covered by the helper API:

- use `search_raw()` for request-body control
- add a small project-specific helper next to this package
- keep this package focused on common OpenSearch workflows

## Team Rollout

For cloud or shared-team use, the practical options are:

1. Keep this folder in a shared repo and install with `pip install -e .` during development.
2. Install from your Git repository URL in worker environments.
3. Publish it to an internal package registry once the API stabilizes.

The example scripts under [`example/`](./example/README.md) cover the common operational flows.
