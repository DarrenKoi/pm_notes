"""Search operations: full-text, term, bool, knn, hybrid, aggregations."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from opensearchpy import OpenSearch


def match_search(
    client: OpenSearch,
    index: str,
    field: str,
    query: str,
    size: int = 10,
) -> dict:
    """Full-text match search on a single field."""
    body = {"query": {"match": {field: query}}, "size": size}
    return client.search(index=index, body=body)


def term_search(
    client: OpenSearch,
    index: str,
    field: str,
    value: Any,
    size: int = 10,
) -> dict:
    """Exact-match term search."""
    body = {"query": {"term": {field: value}}, "size": size}
    return client.search(index=index, body=body)


def bool_search(
    client: OpenSearch,
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
    return client.search(index=index, body=body)


def knn_search(
    client: OpenSearch,
    index: str,
    field: str,
    vector: Sequence[float],
    k: int = 5,
    size: int = 10,
) -> dict:
    """k-NN vector similarity search (OpenSearch k-NN plugin)."""
    body = {
        "query": {
            "knn": {
                field: {
                    "vector": list(vector),
                    "k": k,
                }
            }
        },
        "size": size,
    }
    return client.search(index=index, body=body)


def hybrid_search(
    client: OpenSearch,
    index: str,
    query: str,
    text_field: str,
    vector_field: str,
    vector: Sequence[float],
    k: int = 5,
    size: int = 10,
) -> dict:
    """Hybrid search combining full-text match and k-NN vector similarity.

    Uses a bool query with ``should`` to blend lexical and semantic scores.
    """
    body = {
        "query": {
            "bool": {
                "should": [
                    {"match": {text_field: query}},
                    {"knn": {vector_field: {"vector": list(vector), "k": k}}},
                ],
            }
        },
        "size": size,
    }
    return client.search(index=index, body=body)


def aggregate(
    client: OpenSearch,
    index: str,
    agg_body: dict[str, Any],
    query: Optional[dict[str, Any]] = None,
    size: int = 0,
) -> dict:
    """Run an aggregation query.

    Args:
        client: OpenSearch client.
        index: Target index.
        agg_body: Aggregation definition (the value of ``"aggs"``).
        query: Optional query to scope the aggregation.
        size: Number of hits to return alongside aggregations (default 0).
    """
    body: dict[str, Any] = {"aggs": agg_body, "size": size}
    if query:
        body["query"] = query
    return client.search(index=index, body=body)
