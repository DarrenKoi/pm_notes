"""Search operations for OpenSearch."""

from __future__ import annotations

from typing import Any, Optional, Sequence


def _search(client: Any, index: str, body: dict[str, Any]) -> dict:
    return client.search(index=index, body=body)


def _count(client: Any, index: str, body: dict[str, Any]) -> dict:
    return client.count(index=index, body=body)


def match_search(
    client: Any,
    index: str,
    field: str,
    query: str,
    size: int = 10,
) -> dict:
    """Full-text match search on a single field."""
    body = {"query": {"match": {field: query}}, "size": size}
    return _search(client, index, body)


def term_search(
    client: Any,
    index: str,
    field: str,
    value: Any,
    size: int = 10,
) -> dict:
    """Exact-match term search."""
    body = {"query": {"term": {field: value}}, "size": size}
    return _search(client, index, body)


def bool_search(
    client: Any,
    index: str,
    must: Optional[list[dict]] = None,
    should: Optional[list[dict]] = None,
    filter: Optional[list[dict]] = None,
    must_not: Optional[list[dict]] = None,
    size: int = 10,
) -> dict:
    """Compound bool query."""
    bool_clause: dict[str, Any] = {}
    if must:
        bool_clause["must"] = must
    if should:
        bool_clause["should"] = should
    if filter:
        bool_clause["filter"] = filter
    if must_not:
        bool_clause["must_not"] = must_not

    body = {"query": {"bool": bool_clause}, "size": size}
    return _search(client, index, body)


def knn_search(
    client: Any,
    index: str,
    field: str,
    vector: Sequence[float],
    k: int = 5,
    size: int = 10,
    filters: Optional[Sequence[dict[str, Any]]] = None,
) -> dict:
    """Run OpenSearch k-NN vector similarity search."""
    filter_list = list(filters or [])
    knn_query = {
        "knn": {
            field: {
                "vector": list(vector),
                "k": k,
            }
        }
    }
    if filter_list:
        body = {
            "query": {
                "bool": {
                    "must": [knn_query],
                    "filter": filter_list,
                }
            },
            "size": size,
        }
    else:
        body = {"query": knn_query, "size": size}
    return _search(client, index, body)


def hybrid_search(
    client: Any,
    index: str,
    query: str,
    text_field: str,
    vector_field: str,
    vector: Sequence[float],
    k: int = 5,
    size: int = 10,
    filters: Optional[Sequence[dict[str, Any]]] = None,
) -> dict:
    """Combine lexical and vector retrieval using OpenSearch syntax."""
    filter_list = list(filters or [])
    bool_clause: dict[str, Any] = {
        "should": [
            {"match": {text_field: query}},
            {"knn": {vector_field: {"vector": list(vector), "k": k}}},
        ],
        "minimum_should_match": 1,
    }
    if filter_list:
        bool_clause["filter"] = filter_list

    body = {"query": {"bool": bool_clause}, "size": size}
    return _search(client, index, body)


def aggregate(
    client: Any,
    index: str,
    agg_body: dict[str, Any],
    query: Optional[dict[str, Any]] = None,
    size: int = 0,
) -> dict:
    """Run an aggregation query."""
    body: dict[str, Any] = {"aggs": agg_body, "size": size}
    if query:
        body["query"] = query
    return _search(client, index, body)


def search_raw(
    client: Any,
    index: str,
    body: dict[str, Any],
) -> dict:
    """Run an arbitrary query body."""
    return _search(client, index, body)


def multi_match_search(
    client: Any,
    index: str,
    query: str,
    fields: Sequence[str],
    size: int = 10,
    match_type: str = "best_fields",
) -> dict:
    """Search across multiple text fields using multi_match."""
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": list(fields),
                "type": match_type,
            }
        },
        "size": size,
    }
    return _search(client, index, body)


def count_documents(
    client: Any,
    index: str,
    query: Optional[dict[str, Any]] = None,
) -> dict:
    """Return count for all docs or docs matching a query."""
    body = {"query": query} if query else {}
    return _count(client, index, body)
