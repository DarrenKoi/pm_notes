# opensearch_handler Examples

These examples are written around OpenSearch usage patterns teams actually use:

- connect cleanly
- keep index naming consistent
- use aliases for writes and reads
- bulk ingest documents
- run text and aggregation queries
- add vector or hybrid search where k-NN is enabled

## Setup

```bash
cd Codes/python/opensearch_handler
python3 -m pip install -r requirements.txt
```

OpenSearch environment variables:

```bash
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASSWORD=admin
export OPENSEARCH_USE_SSL=true
export OPENSEARCH_VERIFY_CERTS=false
```

## Topic Scripts

```bash
python3 -m example.topic_01_connection
python3 -m example.topic_02_index_management
python3 -m example.topic_03_document_crud
python3 -m example.topic_04_search_text_and_agg
python3 -m example.topic_05_vector_and_hybrid
python3 -m example.topic_06_template_alias_rollover
python3 -m example.topic_07_search_raw
```

Run them from the `Codes/python/opensearch_handler` directory so imports resolve through the package layout directly, without path rewrites.

## What Each Topic Covers

- `topic_01_connection.py`: backend-aware config and connection check.
- `topic_02_index_management.py`: create index, inspect settings/mapping, refresh, alias inspection.
- `topic_03_document_crud.py`: single-document CRUD plus bulk indexing.
- `topic_04_search_text_and_agg.py`: the text-query helpers teams use most often.
- `topic_05_vector_and_hybrid.py`: OpenSearch vector and hybrid examples.
- `topic_06_template_alias_rollover.py`: numbered index naming, aliases, and rollover dry-run.
- `topic_07_search_raw.py`: OpenSearch request bodies when the helper layer is not enough.

## Notes

- Topic 05 depends on OpenSearch k-NN support and field mappings.
- Topic 06 does not manage lifecycle retention policy in code.
- When you need something not covered by a helper, use `search_raw()` or add a local wrapper function.
