"""Search module for retrieving knowhow from Elasticsearch.

Provides keyword-based, full-text, and vector search functions
intended to be called from an external LangGraph pipeline.
"""

import logging

from elasticsearch import Elasticsearch

import config
from es_client import get_client
from llm_processor import create_embedding_client, generate_embeddings

logger = logging.getLogger(__name__)


def search_by_keywords(
    keywords: list[str],
    *,
    es: Elasticsearch | None = None,
    size: int = 5,
    category: str | None = None,
) -> list[dict]:
    """Exact match on the `keywords` field. Best when the LLM extracts
    keywords that closely match the stored keyword terms."""
    es = es or get_client()

    must = [{"terms": {"keywords": keywords}}]
    if category:
        must.append({"term": {"category": category}})

    body = {"query": {"bool": {"must": must}}, "size": size}
    resp = es.search(index=config.ES_INDEX, body=body)
    return _hits_to_results(resp)


def search_by_text(
    query: str,
    *,
    es: Elasticsearch | None = None,
    size: int = 5,
    category: str | None = None,
) -> list[dict]:
    """Full-text search across `knowhow`, `summary`, and `keywords` fields.
    Good for broader matching when exact keyword alignment isn't guaranteed."""
    es = es or get_client()

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
    resp = es.search(index=config.ES_INDEX, body=body)
    return _hits_to_results(resp)


def search_by_vector(
    query_text: str,
    *,
    es: Elasticsearch | None = None,
    size: int = 5,
) -> list[dict]:
    """Vector similarity search using cosine similarity (ES 7.x script_score).
    Embeds the query text via the embedding API, then scores all documents."""
    es = es or get_client()

    emb_client = create_embedding_client()
    query_vector = generate_embeddings(emb_client, [query_text])[0]

    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector},
                },
            }
        },
        "size": size,
    }
    resp = es.search(index=config.ES_INDEX, body=body)
    return _hits_to_results(resp)


def hybrid_search(
    query: str,
    keywords: list[str],
    *,
    es: Elasticsearch | None = None,
    size: int = 5,
    category: str | None = None,
) -> list[dict]:
    """Combines keyword exact match + full-text search in a single query.
    Use this when the LangGraph LLM provides both extracted keywords and
    the original query text."""
    es = es or get_client()

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
    resp = es.search(index=config.ES_INDEX, body=body)
    return _hits_to_results(resp)


def _hits_to_results(resp: dict) -> list[dict]:
    results = []
    for hit in resp["hits"]["hits"]:
        doc = hit["_source"]
        doc.pop("embedding", None)
        results.append({"score": hit["_score"], **doc})
    return results
