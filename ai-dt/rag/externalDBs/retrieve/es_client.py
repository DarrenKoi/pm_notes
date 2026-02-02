"""Elasticsearch 7.14 glossary search (fuzzy + exact)."""

from dataclasses import dataclass

from elasticsearch import Elasticsearch

from config import ES_HOST, ES_INDEX


@dataclass
class GlossaryMatch:
    term: str
    aliases: list[str]
    definition: str
    category: str
    score: float


def get_es_client() -> Elasticsearch:
    return Elasticsearch(ES_HOST)


def lookup_terms(
    query: str,
    top_k: int = 5,
    fuzziness: str = "AUTO",
    min_score: float = 0.0,
    es: Elasticsearch | None = None,
) -> list[GlossaryMatch]:
    """Search glossary by combining exact match on term/aliases and fuzzy match on term_text."""
    es = es or get_es_client()

    body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"terms": {"term": query.split(), "boost": 3.0}},
                    {"terms": {"aliases": query.split(), "boost": 3.0}},
                    {
                        "match": {
                            "term_text": {
                                "query": query,
                                "fuzziness": fuzziness,
                                "boost": 1.5,
                            }
                        }
                    },
                    {"match": {"definition": {"query": query, "boost": 0.5}}},
                ],
                "minimum_should_match": 1,
            }
        },
    }

    resp = es.search(index=ES_INDEX, body=body)

    results = []
    for hit in resp["hits"]["hits"]:
        if hit["_score"] < min_score:
            continue
        src = hit["_source"]
        results.append(GlossaryMatch(
            term=src["term"],
            aliases=src.get("aliases", []),
            definition=src.get("definition", ""),
            category=src.get("category", ""),
            score=hit["_score"],
        ))
    return results
