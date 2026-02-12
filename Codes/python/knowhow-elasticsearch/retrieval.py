"""Retrieval module for knowhow documents from OpenSearch.

Primary strategy: hybrid search combining keyword exact match + full-text
search on knowhow/summary in a single query for better relevance ranking.
"""

import logging
from typing import Any

from opensearchpy import OpenSearch

from os_settings import OS_INDEX, get_connection_config  # triggers _path_setup
import opensearch_handler as osh

logger = logging.getLogger(__name__)

_client: OpenSearch | None = None


def _get_client() -> OpenSearch:
    """Lazy-initialise a shared client from the active cluster config."""
    global _client
    if _client is None:
        _client = osh.create_client(config=get_connection_config())
    return _client


# -- Primary retrieval ----------------------------------------------------


def retrieve(
    keywords: list[str],
    *,
    query: str | None = None,
    client: OpenSearch | None = None,
    size: int = 10,
    category: str | None = None,
) -> list[dict]:
    """Hybrid retrieval combining keyword exact match + full-text search.

    Runs keyword exact match, knowhow full-text, and summary full-text
    in a single query. Documents matching multiple signals score higher.

    Boost strategy:
        keywords (exact):  3.0  — highest priority, exact term match
        knowhow (text):    2.0  — original knowhow content
        summary (text):    1.5  — LLM-generated summary

    Args:
        keywords: List of keyword strings to search for.
        query:    Optional free-text query. If not provided, keywords are
                  joined as query text for the full-text fields.
        size:     Max number of results to return.
        category: Optional category filter (exact match).

    Returns:
        List of document dicts with `_score` field, ranked by relevance.
    """
    client = client or _get_client()
    normalized = [k.strip().lower() for k in keywords if k.strip()]
    query_text = query or " ".join(normalized)

    should = [
        {"terms": {"keywords": normalized, "boost": 3.0}},
        {"match": {"knowhow": {"query": query_text, "boost": 2.0}}},
        {"match": {"summary": {"query": query_text, "boost": 1.5}}},
    ]

    filter_ = []
    if category:
        filter_.append({"term": {"category": category}})

    body = {
        "query": {
            "bool": {
                "should": should,
                "filter": filter_,
                "minimum_should_match": 1,
            }
        },
        "size": size,
    }
    resp = client.search(index=OS_INDEX, body=body)
    return _format_hits(resp)


# -- Utility methods ------------------------------------------------------


def retrieve_by_category(
    category: str,
    *,
    client: OpenSearch | None = None,
    size: int = 50,
) -> list[dict]:
    """Get all documents in a specific category."""
    client = client or _get_client()
    resp = osh.term_search(client, OS_INDEX, "category", category, size=size)
    return _format_hits(resp)


def retrieve_by_user(
    user_id: str,
    *,
    client: OpenSearch | None = None,
    size: int = 50,
) -> list[dict]:
    """Get all documents authored by a specific user."""
    client = client or _get_client()
    resp = osh.term_search(client, OS_INDEX, "user_id", user_id, size=size)
    return _format_hits(resp)


def retrieve_by_id(
    knowhow_id: str,
    *,
    client: OpenSearch | None = None,
) -> dict | None:
    """Get a single document by KNOWHOW_ID."""
    client = client or _get_client()
    resp = osh.term_search(client, OS_INDEX, "KNOWHOW_ID", knowhow_id, size=1)
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
    client = client or _get_client()
    agg_body = {"unique_keywords": {"terms": {"field": "keywords", "size": size}}}
    resp = osh.aggregate(client, OS_INDEX, agg_body)
    buckets = resp["aggregations"]["unique_keywords"]["buckets"]
    return [{"keyword": b["key"], "doc_count": b["doc_count"]} for b in buckets]


def list_categories(
    *,
    client: OpenSearch | None = None,
) -> list[dict[str, Any]]:
    """Return all unique categories with their document counts."""
    client = client or _get_client()
    agg_body = {"unique_categories": {"terms": {"field": "category", "size": 100}}}
    resp = osh.aggregate(client, OS_INDEX, agg_body)
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
            print(f"[{i}] score={doc['_score']:.2f}")
            print(f"    keywords: {doc.get('keywords', [])}")
            print(f"    category: {doc.get('category', '')}")
            print(f"    summary:  {doc.get('summary', '')[:120]}")
            print()
