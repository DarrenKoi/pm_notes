"""Search module for retrieving knowhow from OpenSearch.

Provides keyword-based and full-text search functions
intended to be called from an external LangGraph pipeline.
"""

import logging

from opensearchpy import OpenSearch

from os_client import get_client
from os_settings import OS_INDEX

logger = logging.getLogger(__name__)


def search_by_keywords(
    keywords: list[str],
    *,
    client: OpenSearch | None = None,
    size: int = 5,
    category: str | None = None,
) -> list[dict]:
    """Exact match on the `keywords` field. Best when the LLM extracts
    keywords that closely match the stored keyword terms."""
    client = client or get_client()

    must = [{"terms": {"keywords": keywords}}]
    if category:
        must.append({"term": {"category": category}})

    body = {"query": {"bool": {"must": must}}, "size": size}
    resp = client.search(index=OS_INDEX, body=body)
    return _hits_to_results(resp)


def search_by_text(
    query: str,
    *,
    client: OpenSearch | None = None,
    size: int = 5,
    category: str | None = None,
) -> list[dict]:
    """Full-text search across `knowhow`, `summary`, and `keywords` fields.
    Good for broader matching when exact keyword alignment isn't guaranteed."""
    client = client or get_client()

    should = [
        {"match": {"knowhow": {"query": query, "boost": 2.0}}},
        {"match": {"summary": {"query": query, "boost": 1.5}}},
        {"match": {"keywords": {"query": query, "boost": 1.0}}},
    ]
    filter_ = []
    if category:
        filter_.append({"term": {"category": category}})

    body = {
        "query": {"bool": {"should": should, "filter": filter_, "minimum_should_match": 1}},
        "size": size,
    }
    resp = client.search(index=OS_INDEX, body=body)
    return _hits_to_results(resp)



def hybrid_search(
    query: str,
    keywords: list[str],
    *,
    client: OpenSearch | None = None,
    size: int = 5,
    category: str | None = None,
) -> list[dict]:
    """Combines keyword exact match + full-text search in a single query.
    Use this when the LangGraph LLM provides both extracted keywords and
    the original query text."""
    client = client or get_client()

    should = [
        {"terms": {"keywords": [k.lower() for k in keywords], "boost": 3.0}},
        {"match": {"knowhow": {"query": query, "boost": 2.0}}},
        {"match": {"summary": {"query": query, "boost": 1.5}}},
    ]
    filter_ = []
    if category:
        filter_.append({"term": {"category": category}})

    body = {
        "query": {"bool": {"should": should, "filter": filter_, "minimum_should_match": 1}},
        "size": size,
    }
    resp = client.search(index=OS_INDEX, body=body)
    return _hits_to_results(resp)


def _hits_to_results(resp: dict) -> list[dict]:
    results = []
    for hit in resp["hits"]["hits"]:
        doc = hit["_source"]
        results.append({"score": hit["_score"], **doc})
    return results
