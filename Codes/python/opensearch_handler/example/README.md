# opensearch_handler Examples by Topic

This folder contains topic-based scripts for learning how to use `opensearch_handler`.

## 0) Setup

```bash
cd Codes/python/opensearch_handler
python3 -m pip install -r requirements.txt
```

Set connection environment variables if needed:

```bash
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASSWORD=admin
export OPENSEARCH_USE_SSL=true
export OPENSEARCH_VERIFY_CERTS=false
```

## 1) Topic scripts

```bash
python3 example/topic_01_connection.py
python3 example/topic_02_index_management.py
python3 example/topic_03_document_crud.py
python3 example/topic_04_search_text_and_agg.py
python3 example/topic_05_vector_and_hybrid.py
python3 example/topic_06_template_alias_lifecycle.py
python3 example/topic_07_search_raw.py
```

## 2) What each topic covers

- `topic_01_connection.py`: Build config and client, ping cluster, inspect version.
- `topic_02_index_management.py`: Create, inspect, update settings, refresh, delete index.
- `topic_03_document_crud.py`: Index/get/update/upsert/delete documents and bulk indexing.
- `topic_04_search_text_and_agg.py`: Match/term/multi-match/bool search and aggregation.
- `topic_05_vector_and_hybrid.py`: k-NN vector search and hybrid text+vector search.
- `topic_06_template_alias_lifecycle.py`: Template upsert, aliases, rollover dry-run, lifecycle policy.
- `topic_07_search_raw.py`: Advanced queries with `search_raw` — pagination, sorting, `_source` filtering, highlighting, date range.

## 3) Notes

- Most scripts create temporary indexes and clean them up at the end.
- `topic_05_vector_and_hybrid.py` requires OpenSearch k-NN support.
- `topic_06_template_alias_lifecycle.py` leaves the template and policy in place by design.
