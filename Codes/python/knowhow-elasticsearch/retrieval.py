"""Retrieval module for knowhow documents from OpenSearch.

Primary strategy: exact match on the `keywords` field (keyword type).
Fallback: full-text search on `summary` when no keyword matches are found.
"""

import logging
from typing import Any

from opensearchpy import OpenSearch

from os_client import get_client
from os_settings import OS_INDEX

logger = logging.getLogger(__name__)


# -- Primary retrieval ----------------------------------------------------


def retrieve(
    keywords: list[str],
    *,
    client: OpenSearch | None = None,
    size: int = 10,
    min_match: int = 1,
) -> list[dict]:
    """Retrieve documents matching the given keywords list.

    1. Exact match on the `keywords` field.
    2. If no results, fall back to full-text search on `summary`.

    Args:
        keywords:  List of keyword strings to search for.
        size:      Max number of results to return.
        min_match: Minimum number of keywords that must match (default 1).

    Returns:
        List of document dicts with an added `_score` and `_match_type` field.
    """
    client = client or get_client()

    results = search_keywords_exact(keywords, client=client, size=size, min_match=min_match)
    if results:
        return results

    logger.info("No exact keyword match. Falling back to summary search.")
    return search_summary_fallback(keywords, client=client, size=size)


def search_keywords_exact(
    keywords: list[str],
    *,
    client: OpenSearch | None = None,
    size: int = 10,
    min_match: int = 1,
) -> list[dict]:
    """Exact match on the `keywords` field (keyword type).

    Uses a bool/should query so that documents matching more keywords
    are scored higher. `min_match` controls how many keywords must match.
    """
    client = client or get_client()
    normalized = [k.strip().lower() for k in keywords if k.strip()]

    should = [{"term": {"keywords": kw}} for kw in normalized]

    body = {
        "query": {
            "bool": {
                "should": should,
                "minimum_should_match": min_match,
            }
        },
        "size": size,
    }
    resp = client.search(index=OS_INDEX, body=body)
    return _format_hits(resp, match_type="keyword_exact")


def search_summary_fallback(
    keywords: list[str],
    *,
    client: OpenSearch | None = None,
    size: int = 10,
) -> list[dict]:
    """Full-text search on `summary` using the keyword list as query text."""
    client = client or get_client()
    query_text = " ".join(kw.strip() for kw in keywords if kw.strip())

    body = {
        "query": {
            "match": {
                "summary": {
                    "query": query_text,
                    "analyzer": "korean",
                }
            }
        },
        "size": size,
    }
    resp = client.search(index=OS_INDEX, body=body)
    return _format_hits(resp, match_type="summary_fallback")


# -- Utility methods ------------------------------------------------------


def retrieve_by_category(
    category: str,
    *,
    client: OpenSearch | None = None,
    size: int = 50,
) -> list[dict]:
    """Get all documents in a specific category."""
    client = client or get_client()

    body = {"query": {"term": {"category": category}}, "size": size}
    resp = client.search(index=OS_INDEX, body=body)
    return _format_hits(resp)


def retrieve_by_user(
    user_id: str,
    *,
    client: OpenSearch | None = None,
    size: int = 50,
) -> list[dict]:
    """Get all documents authored by a specific user."""
    client = client or get_client()

    body = {"query": {"term": {"user_id": user_id}}, "size": size}
    resp = client.search(index=OS_INDEX, body=body)
    return _format_hits(resp)


def retrieve_by_id(
    knowhow_id: str,
    *,
    client: OpenSearch | None = None,
) -> dict | None:
    """Get a single document by KNOWHOW_ID."""
    client = client or get_client()

    body = {"query": {"term": {"KNOWHOW_ID": knowhow_id}}, "size": 1}
    resp = client.search(index=OS_INDEX, body=body)
    hits = _format_hits(resp)
    return hits[0] if hits else None


def list_all_keywords(
    *,
    client: OpenSearch | None = None,
    size: int = 500,
) -> list[dict[str, Any]]:
    """Return all unique keywords in the index with their document counts.

    Useful for understanding what keywords exist and building autocomplete.

    Returns:
        List of {"keyword": str, "doc_count": int} sorted by count desc.
    """
    client = client or get_client()

    body = {
        "size": 0,
        "aggs": {
            "unique_keywords": {
                "terms": {"field": "keywords", "size": size}
            }
        },
    }
    resp = client.search(index=OS_INDEX, body=body)
    buckets = resp["aggregations"]["unique_keywords"]["buckets"]
    return [{"keyword": b["key"], "doc_count": b["doc_count"]} for b in buckets]


def list_categories(
    *,
    client: OpenSearch | None = None,
) -> list[dict[str, Any]]:
    """Return all unique categories with their document counts."""
    client = client or get_client()

    body = {
        "size": 0,
        "aggs": {
            "unique_categories": {
                "terms": {"field": "category", "size": 100}
            }
        },
    }
    resp = client.search(index=OS_INDEX, body=body)
    buckets = resp["aggregations"]["unique_categories"]["buckets"]
    return [{"category": b["key"], "doc_count": b["doc_count"]} for b in buckets]


# -- Internal helpers -----------------------------------------------------


def _format_hits(resp: dict, match_type: str = "") -> list[dict]:
    results = []
    for hit in resp["hits"]["hits"]:
        doc = hit["_source"]
        entry = {"_score": hit["_score"], **doc}
        if match_type:
            entry["_match_type"] = match_type
        results.append(entry)
    return results


# -- CLI usage ------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python retrieval.py keyword1 keyword2 ...")
        sys.exit(1)

    input_keywords = sys.argv[1:]
    print(f"Searching for keywords: {input_keywords}\n")

    docs = retrieve(input_keywords)
    if not docs:
        print("No results found.")
    else:
        for i, doc in enumerate(docs, 1):
            print(f"[{i}] score={doc['_score']:.2f}  match={doc.get('_match_type', 'n/a')}")
            print(f"    keywords: {doc.get('keywords', [])}")
            print(f"    category: {doc.get('category', '')}")
            print(f"    summary:  {doc.get('summary', '')[:120]}")
            print()
