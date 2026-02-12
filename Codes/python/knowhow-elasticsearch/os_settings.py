"""
Cluster settings for OpenSearch / Elasticsearch environments.

Select the active cluster by setting ACTIVE_CLUSTER to one of the keys
in CLUSTERS. All other modules read from the module-level exports:
    OS_HOST, OS_INDEX, OS_BULK_CHUNK, INDEX_SETTINGS
"""

import os

# ========================================================================
# Cluster definitions
# ========================================================================

CLUSTERS = {
    # ------------------------------------------------------------------
    # OpenSearch standalone (dev)
    #   Cluster Manager: 3 nodes | 2 cores | 4 GiB RAM  | 10 GiB storage
    #   Data:            1 node  | 4 cores | 8 GiB RAM  | 100 GiB storage
    # ------------------------------------------------------------------
    "opensearch-dev": {
        "host": "https://localhost:9200",
        "index": "knowhow",
        "bulk_chunk": 500,
        "shards": 1,
        "replicas": 0,
        "refresh_interval": "30s",
    },
    # ------------------------------------------------------------------
    # Elasticsearch 7.14 cluster (production)
    #   Master: 3 nodes (dedicated)
    #   Data:   3 nodes
    #
    #   opensearch-py is compatible with ES 7.x (forked from ES 7.10).
    # ------------------------------------------------------------------
    "es-prod": {
        "host": "https://localhost:9200",  # TODO: replace with actual ES host
        "index": "knowhow",
        "bulk_chunk": 500,
        "shards": 2,
        "replicas": 1,
        "refresh_interval": "30s",
    },
}

# ========================================================================
# Active cluster selection
# ========================================================================
# Override via environment variable: KNOWHOW_CLUSTER=es-prod python index.py
ACTIVE_CLUSTER = os.environ.get("KNOWHOW_CLUSTER", "opensearch-dev")

_cfg = CLUSTERS[ACTIVE_CLUSTER]

# ========================================================================
# Module-level exports (used by os_client, retrieval, etc.)
# ========================================================================

OS_HOST = _cfg["host"]
OS_INDEX = _cfg["index"]
OS_BULK_CHUNK = _cfg["bulk_chunk"]

# ========================================================================
# Index settings
# ========================================================================

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": _cfg["shards"],
        "number_of_replicas": _cfg["replicas"],
        "refresh_interval": _cfg["refresh_interval"],
        "analysis": {
            "analyzer": {
                "korean": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["nori_readingform", "lowercase"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "KNOWHOW_ID": {"type": "keyword"},
            "knowhow": {
                "type": "text",
                "analyzer": "korean",
                "fields": {"standard": {"type": "text", "analyzer": "standard"}},
            },
            "user_id": {"type": "keyword"},
            "user_name": {"type": "keyword"},
            "user_department": {"type": "keyword"},
            "summary": {
                "type": "text",
                "analyzer": "korean",
                "fields": {"standard": {"type": "text", "analyzer": "standard"}},
            },
            "category": {"type": "keyword"},
            "keywords": {"type": "keyword"},
        },
    },
}

# ========================================================================
# Scaling guide
# ========================================================================
#
# opensearch-dev (1 data node):
#   shards=1, replicas=0 → Green health. No HA.
#
# es-prod (3 data nodes):
#   | Shards | Replicas | Total copies | Notes                          |
#   |--------|----------|--------------|--------------------------------|
#   | 1      | 1        | 2            | Simple HA. 2 of 3 nodes used.  |
#   | 1      | 2        | 3            | Max read throughput. All nodes. |
#   | 2      | 1        | 4            | Balance write + read across 3. |  ← default
#   | 3      | 1        | 6            | Max parallelism. 2 per node.   |
#
# To change replicas on a live index:
#   client.indices.put_settings(
#       index="knowhow",
#       body={"index": {"number_of_replicas": 2}},
#   )
