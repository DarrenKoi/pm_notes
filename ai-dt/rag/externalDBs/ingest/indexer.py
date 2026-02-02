"""Elasticsearch 7.14 index creation and bulk ingestion."""

from datetime import datetime, timezone

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from config import ES_HOST, ES_INDEX
from models import GlossaryEntry

INDEX_BODY = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "term_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "trim"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "term":       {"type": "keyword"},
            "aliases":    {"type": "keyword"},
            "term_text":  {"type": "text", "analyzer": "term_analyzer"},
            "definition": {"type": "text", "analyzer": "standard"},
            "category":   {"type": "keyword"},
            "source_ids": {"type": "keyword"},
            "updated_at": {"type": "date"},
        }
    },
}


def get_es_client() -> Elasticsearch:
    return Elasticsearch(ES_HOST)


def create_index(es: Elasticsearch | None = None, recreate: bool = False):
    """Create the glossary index. If recreate=True, delete existing first."""
    es = es or get_es_client()
    if es.indices.exists(index=ES_INDEX):
        if recreate:
            es.indices.delete(index=ES_INDEX)
            print(f"Deleted existing index '{ES_INDEX}'")
        else:
            print(f"Index '{ES_INDEX}' already exists, skipping creation")
            return
    es.indices.create(index=ES_INDEX, body=INDEX_BODY)
    print(f"Index '{ES_INDEX}' created.")


def _build_bulk_actions(entries: list[GlossaryEntry]):
    now = datetime.now(timezone.utc).isoformat()
    for entry in entries:
        yield {
            "_index": ES_INDEX,
            "_id": entry.term.lower().replace(" ", "_"),
            "_source": {
                "term": entry.term,
                "aliases": entry.aliases,
                "term_text": entry.term,
                "definition": entry.definition,
                "category": entry.category,
                "source_ids": entry.source_ids,
                "updated_at": now,
            },
        }


def ingest_glossary(entries: list[GlossaryEntry], es: Elasticsearch | None = None):
    """Bulk index glossary entries into ES."""
    es = es or get_es_client()
    success, errors = bulk(es, _build_bulk_actions(entries), raise_on_error=False)
    print(f"Indexed {success} entries, {len(errors)} errors.")
    if errors:
        for err in errors[:5]:
            print(f"  Error: {err}")
    return success, errors
