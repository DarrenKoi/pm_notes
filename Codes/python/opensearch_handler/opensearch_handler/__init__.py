"""General-purpose handler for OpenSearch."""

from .client import create_client
from .connection_settings import ConnectionConfig, load_config
from .document import (
    bulk_actions,
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
    get_aliases,
    get_index_mapping,
    get_index_settings,
    index_exists,
    refresh_index,
    update_aliases,
    update_index_settings,
)
from .lifecycle import (
    bootstrap_aliases,
    format_rollover_index,
    put_index_template,
    rollover_alias,
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
    "ConnectionConfig",
    "load_config",
    "create_client",
    "create_index",
    "delete_index",
    "get_aliases",
    "get_index_settings",
    "get_index_mapping",
    "index_exists",
    "refresh_index",
    "update_aliases",
    "update_index_settings",
    "bootstrap_aliases",
    "format_rollover_index",
    "put_index_template",
    "rollover_alias",
    "index_document",
    "get_document",
    "update_document",
    "upsert_document",
    "delete_document",
    "bulk_actions",
    "bulk_index",
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
