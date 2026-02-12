"""General-purpose OpenSearch/Elasticsearch handler.

Supports both OpenSearch and Elasticsearch 7.x via opensearch-py.
"""

from .client import create_client
from .config import ConnectionConfig, load_config
from .document import (
    bulk_index,
    delete_document,
    get_document,
    index_document,
    upsert_document,
    update_document,
)
from .index import (
    create_index,
    delete_index,
    get_index_mapping,
    get_index_settings,
    index_exists,
    refresh_index,
    update_index_settings,
)
from .search import (
    aggregate,
    bool_search,
    count_documents,
    hybrid_search,
    knn_search,
    match_search,
    multi_match_search,
    search_raw,
    term_search,
)

__all__ = [
    # client
    "create_client",
    # config
    "ConnectionConfig",
    "load_config",
    # index
    "create_index",
    "delete_index",
    "get_index_settings",
    "get_index_mapping",
    "index_exists",
    "refresh_index",
    "update_index_settings",
    # document
    "index_document",
    "get_document",
    "update_document",
    "upsert_document",
    "delete_document",
    "bulk_index",
    # search
    "match_search",
    "term_search",
    "bool_search",
    "knn_search",
    "hybrid_search",
    "aggregate",
    "search_raw",
    "multi_match_search",
    "count_documents",
]
