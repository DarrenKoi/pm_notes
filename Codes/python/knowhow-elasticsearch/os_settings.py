"""
OpenSearch index settings optimized for the current cluster topology.

Cluster topology (as of 2026-02):
  - Cluster Manager: 3 nodes  | 2 cores | 4 GiB RAM  | 10 GiB storage
  - Data:           1 node    | 4 cores | 8 GiB RAM  | 100 GiB storage

Key decisions:
  - replicas=0: Only 1 data node, so replicas would remain unassigned (Yellow health).
                Add replicas when scaling to 2+ data nodes.
  - shards=1:   Small dataset on a single data node. More shards add overhead with no benefit.
  - refresh=30s: Reduces I/O during bulk indexing. Default 1s is too aggressive for batch loads.
  - nori analyzer: Handles Korean morphological analysis for better full-text search quality.
                   Requires analysis-nori plugin installed on all OpenSearch nodes.
"""

# -- Connection ----------------------------------------------------------

OS_HOST = "https://localhost:9200"
OS_INDEX = "knowhow"
OS_BULK_CHUNK = 500

# -- Index-level settings ------------------------------------------------

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "30s",
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

# -- Scaling guide -------------------------------------------------------
#
# When you add more DATA nodes, update these settings:
#
# | Data Nodes | Shards | Replicas | Notes                              |
# |------------|--------|----------|------------------------------------|
# | 1          | 1      | 0        | Current setup. Green health.       |
# | 2          | 1      | 1        | 1 replica for HA. Green health.    |
# | 3+         | 2      | 1        | Distribute load across nodes.      |
#
# To apply new replica count on a live index:
#   client.indices.put_settings(
#       index="knowhow",
#       body={"index": {"number_of_replicas": 1}},
#   )
