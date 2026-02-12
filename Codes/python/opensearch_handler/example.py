"""Usage examples for the OpenSearch handler.

Run with an active OpenSearch/ES cluster to test.
Configure via env vars (OPENSEARCH_HOST, etc.) or pass overrides directly.
"""

from opensearch_handler import (
    aggregate,
    bool_search,
    bulk_index,
    count_documents,
    create_client,
    create_index,
    delete_document,
    delete_index,
    get_document,
    hybrid_search,
    index_document,
    index_exists,
    knn_search,
    load_config,
    match_search,
    multi_match_search,
    put_index_template,
    refresh_index,
    term_search,
    upsert_document,
    update_document,
)

# ── 1. Client ────────────────────────────────────────────────────────
# Default: localhost:9200, admin/admin, SSL on
config = load_config()
client = create_client(config=config)

# Or override directly:
#   client = create_client(host="my-es-host", port=9200, use_ssl=False)

# ── 2. Template / lifecycle (optional) ─────────────────────────────
# Composable template works on OpenSearch and Elasticsearch 7.x.
put_index_template(
    client,
    "example-articles-template",
    {
        "index_patterns": ["example-articles*"],
        "template": {
            "settings": {
                "index.number_of_shards": 1,
                "index.number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "fields": {"raw": {"type": "keyword"}},
                    },
                    "category": {"type": "keyword"},
                    "content": {"type": "text"},
                }
            },
        },
        "priority": 100,
    },
)

# Lifecycle API differs by cluster flavor:
# - OpenSearch -> ISM endpoint
# - Elasticsearch 7.x -> ILM endpoint
# Pass the proper policy body for your target cluster.
# put_lifecycle_policy(client, "example-retention", {"policy": {...}}, flavor="auto")

# ── 3. Index management ─────────────────────────────────────────────
INDEX_NAME = "example-articles"

if index_exists(client, INDEX_NAME):
    delete_index(client, INDEX_NAME)

# You decide shards/replicas per index
create_index(
    client,
    INDEX_NAME,
    shards=1,
    replicas=0,
    mappings={
        "properties": {
            "title": {"type": "text"},
            "category": {"type": "keyword"},
            "content": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 3,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
        }
    },
    settings={"index.knn": True},
)

# Optional: rollover alias call
# rollover_alias(client, "example-articles-write", {"max_age": "7d", "max_docs": 1000000})

# ── 4. Document CRUD ────────────────────────────────────────────────
index_document(
    client,
    INDEX_NAME,
    doc={
        "title": "OpenSearch 시작하기",
        "category": "tutorial",
        "content": "OpenSearch는 오픈소스 검색 엔진입니다.",
        "embedding": [0.1, 0.2, 0.3],
    },
    doc_id="1",
)

bulk_index(
    client,
    INDEX_NAME,
    docs=[
        {
            "id": "2",
            "title": "벡터 검색 가이드",
            "category": "guide",
            "content": "k-NN 플러그인을 활용한 벡터 유사도 검색",
            "embedding": [0.4, 0.5, 0.6],
        },
        {
            "id": "3",
            "title": "하이브리드 검색 구현",
            "category": "tutorial",
            "content": "텍스트 검색과 벡터 검색을 결합하는 방법",
            "embedding": [0.7, 0.8, 0.9],
        },
    ],
    id_field="id",
    chunk_size=config.bulk_chunk,
)

doc = get_document(client, INDEX_NAME, "1")
print("Get document:", doc["_source"]["title"])

update_document(client, INDEX_NAME, "1", {"category": "beginner-tutorial"})
upsert_document(
    client,
    INDEX_NAME,
    "4",
    {
        "title": "검색 엔진 운영 팁",
        "category": "operations",
        "content": "샤드, 리프레시, 벌크 튜닝 기본 가이드",
        "embedding": [0.2, 0.2, 0.2],
    },
)

# ── 5. Search ───────────────────────────────────────────────────────
# Full-text
results = match_search(client, INDEX_NAME, "content", "벡터 검색")
print("Match search hits:", results["hits"]["total"]["value"])

# Exact match
results = term_search(client, INDEX_NAME, "category", "tutorial")
print("Term search hits:", results["hits"]["total"]["value"])

results = multi_match_search(client, INDEX_NAME, "검색", ["title", "content"])
print("Multi-match hits:", results["hits"]["total"]["value"])

# Bool compound query
results = bool_search(
    client,
    INDEX_NAME,
    must=[{"match": {"content": "검색"}}],
    filter=[{"term": {"category": "tutorial"}}],
)
print("Bool search hits:", results["hits"]["total"]["value"])

# k-NN vector search
results = knn_search(client, INDEX_NAME, "embedding", [0.3, 0.4, 0.5], k=2)
print("kNN search hits:", results["hits"]["total"]["value"])

# Hybrid (text + vector)
results = hybrid_search(
    client,
    INDEX_NAME,
    query="벡터 검색",
    text_field="content",
    vector_field="embedding",
    vector=[0.3, 0.4, 0.5],
    k=2,
)
print("Hybrid search hits:", results["hits"]["total"]["value"])

# Aggregation
results = aggregate(
    client,
    INDEX_NAME,
    agg_body={"categories": {"terms": {"field": "category"}}},
)
print("Aggregation buckets:", results["aggregations"]["categories"]["buckets"])

count = count_documents(client, INDEX_NAME)
print("Total documents:", count["count"])

# Force refresh when you need read-after-write guarantees.
refresh_index(client, INDEX_NAME)

# ── 6. Cleanup ──────────────────────────────────────────────────────
delete_document(client, INDEX_NAME, "1")
delete_index(client, INDEX_NAME)
print("Cleanup complete.")
