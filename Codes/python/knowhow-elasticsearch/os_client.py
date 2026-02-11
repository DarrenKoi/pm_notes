import logging

from opensearchpy import OpenSearch, helpers

import config
from models import EnrichedKnowhow

logger = logging.getLogger(__name__)

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "KNOWHOW_ID": {"type": "keyword"},
            "knowhow": {"type": "text", "analyzer": "standard"},
            "user_id": {"type": "keyword"},
            "user_name": {"type": "keyword"},
            "user_department": {"type": "keyword"},
            "summary": {"type": "text", "analyzer": "standard"},
            "category": {"type": "keyword"},
            "keywords": {"type": "keyword"},
        }
    }
}


def get_client() -> OpenSearch:
    return OpenSearch(
        config.OS_HOST,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


def ensure_index(client: OpenSearch) -> None:
    if not client.indices.exists(index=config.OS_INDEX):
        client.indices.create(index=config.OS_INDEX, body=INDEX_MAPPING)
        logger.info("Created index: %s", config.OS_INDEX)
    else:
        logger.info("Index already exists: %s", config.OS_INDEX)


def bulk_index(client: OpenSearch, items: list[EnrichedKnowhow]) -> int:
    """Bulk index items in chunks to avoid memory/payload issues with large batches."""
    total_success = 0
    total_skipped = 0
    chunk_size = config.OS_BULK_CHUNK

    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        actions = []
        for item in chunk:
            if not item.keywords:
                total_skipped += 1
                continue
            doc = item.model_dump(exclude={"knowhow_no"})
            actions.append(
                {
                    "_index": config.OS_INDEX,
                    "_source": doc,
                }
            )

        if not actions:
            continue

        success, errors = helpers.bulk(client, actions, raise_on_error=False)
        if errors:
            logger.error("Bulk index errors in chunk %d-%d: %s", start, start + len(chunk), errors)
        total_success += success
        logger.info("Indexed %d/%d documents", start + success, len(items))

    if total_skipped:
        logger.info("Skipped %d items with empty keywords", total_skipped)

    return total_success
