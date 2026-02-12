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
    update_document,
)
from .index import (
    create_index,
    delete_index,
    get_index_settings,
    index_exists,
    update_index_settings,
)
from .search import (
    aggregate,
    bool_search,
    hybrid_search,
    knn_search,
    match_search,
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
    "index_exists",
    "update_index_settings",
    # document
    "index_document",
    "get_document",
    "update_document",
    "delete_document",
    "bulk_index",
    # search
    "match_search",
    "term_search",
    "bool_search",
    "knn_search",
    "hybrid_search",
    "aggregate",
]
