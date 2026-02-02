import logging

from elasticsearch import Elasticsearch, helpers

import config
from models import EnrichedKnowhow

logger = logging.getLogger(__name__)

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "knowhow_no": {"type": "integer"},
            "KNOWHOW_ID": {"type": "keyword"},
            "knowhow": {"type": "text", "analyzer": "standard"},
            "user_id": {"type": "keyword"},
            "user_name": {"type": "keyword"},
            "user_department": {"type": "keyword"},
            "summary": {"type": "text", "analyzer": "standard"},
            "category": {"type": "keyword"},
            "keywords": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": config.EMBEDDING_DIMENSIONS,
            },
        }
    }
}


def get_client() -> Elasticsearch:
    return Elasticsearch(config.ES_HOST)


def ensure_index(es: Elasticsearch) -> None:
    if not es.indices.exists(index=config.ES_INDEX):
        es.indices.create(index=config.ES_INDEX, **INDEX_MAPPING)
        logger.info("Created index: %s", config.ES_INDEX)
    else:
        logger.info("Index already exists: %s", config.ES_INDEX)


def bulk_index(es: Elasticsearch, items: list[EnrichedKnowhow]) -> int:
    """Bulk index items in chunks to avoid memory/payload issues with large batches."""
    total_success = 0
    chunk_size = config.ES_BULK_CHUNK

    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        actions = []
        for item in chunk:
            doc = item.model_dump()
            if doc.get("embedding") is None:
                doc.pop("embedding", None)
            actions.append(
                {
                    "_index": config.ES_INDEX,
                    "_source": doc,
                }
            )

        success, errors = helpers.bulk(es, actions, raise_on_error=False)
        if errors:
            logger.error("Bulk index errors in chunk %d-%d: %s", start, start + len(chunk), errors)
        total_success += success
        logger.info("Indexed %d/%d documents", start + success, len(items))

    return total_success
