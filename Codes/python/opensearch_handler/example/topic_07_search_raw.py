"""Topic 07: search_raw for advanced queries beyond convenience functions.

Covers pagination, sorting, _source filtering, highlighting, and
combined patterns that don't fit the single-purpose helpers.
"""

import _path_setup  # noqa: F401

from opensearch_handler import (
    bulk_index,
    create_client,
    create_index,
    delete_index,
    index_exists,
    load_config,
    search_raw,
)

INDEX_NAME = "example-topic-raw"


def main() -> None:
    client = create_client(config=load_config())

    if index_exists(client, INDEX_NAME):
        delete_index(client, INDEX_NAME)

    create_index(
        client,
        INDEX_NAME,
        mappings={
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "category": {"type": "keyword"},
                "priority": {"type": "integer"},
                "created_at": {"type": "date"},
            }
        },
    )

    bulk_index(
        client,
        INDEX_NAME,
        docs=[
            {"id": "1", "title": "First Post", "content": "Introduction to OpenSearch.", "category": "tutorial", "priority": 3, "created_at": "2025-01-01"},
            {"id": "2", "title": "Search Deep Dive", "content": "Advanced search patterns and tuning.", "category": "guide", "priority": 5, "created_at": "2025-01-15"},
            {"id": "3", "title": "Indexing Tips", "content": "Bulk indexing and refresh strategies.", "category": "operations", "priority": 4, "created_at": "2025-02-01"},
            {"id": "4", "title": "Query DSL Guide", "content": "Bool, match, term, and nested queries.", "category": "guide", "priority": 2, "created_at": "2025-02-10"},
            {"id": "5", "title": "Monitoring Setup", "content": "Dashboard and alerting configuration.", "category": "operations", "priority": 1, "created_at": "2025-03-01"},
        ],
        id_field="id",
        refresh=True,
    )

    # ── 1. Pagination with from/size ──────────────────────────────────
    print("=== Pagination (page 2, size 2) ===")
    results = search_raw(client, INDEX_NAME, {
        "query": {"match_all": {}},
        "from": 2,
        "size": 2,
        "sort": [{"created_at": "asc"}],
    })
    for hit in results["hits"]["hits"]:
        print(f"  {hit['_id']}: {hit['_source']['title']}")

    # ── 2. Sorting ────────────────────────────────────────────────────
    print("\n=== Sort by priority DESC ===")
    results = search_raw(client, INDEX_NAME, {
        "query": {"match_all": {}},
        "size": 3,
        "sort": [{"priority": {"order": "desc"}}],
    })
    for hit in results["hits"]["hits"]:
        print(f"  priority={hit['_source']['priority']}: {hit['_source']['title']}")

    # ── 3. _source filtering ─────────────────────────────────────────
    print("\n=== _source filtering (title only) ===")
    results = search_raw(client, INDEX_NAME, {
        "query": {"match_all": {}},
        "size": 5,
        "_source": ["title"],
    })
    for hit in results["hits"]["hits"]:
        print(f"  fields returned: {list(hit['_source'].keys())}")

    # ── 4. Highlighting ──────────────────────────────────────────────
    print("\n=== Highlighting ===")
    results = search_raw(client, INDEX_NAME, {
        "query": {"match": {"content": "search"}},
        "highlight": {
            "fields": {"content": {}},
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"],
        },
    })
    for hit in results["hits"]["hits"]:
        highlights = hit.get("highlight", {}).get("content", [])
        print(f"  {hit['_source']['title']}: {highlights}")

    # ── 5. Date range + sort combined ────────────────────────────────
    print("\n=== Date range filter + sort ===")
    results = search_raw(client, INDEX_NAME, {
        "query": {
            "bool": {
                "filter": [
                    {"range": {"created_at": {"gte": "2025-01-15", "lte": "2025-03-01"}}},
                ]
            }
        },
        "sort": [{"created_at": "desc"}],
        "size": 10,
    })
    for hit in results["hits"]["hits"]:
        print(f"  {hit['_source']['created_at']}: {hit['_source']['title']}")

    # ── Cleanup ──────────────────────────────────────────────────────
    delete_index(client, INDEX_NAME)
    print("\nCleanup complete.")


if __name__ == "__main__":
    main()
