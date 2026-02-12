import logging

from opensearchpy import OpenSearch, helpers

from models import EnrichedKnowhow
from os_settings import (
    OS_HOST, OS_PORT, OS_USER, OS_PASSWORD, OS_USE_SSL,
    OS_INDEX, OS_BULK_CHUNK, INDEX_SETTINGS, ACTIVE_CLUSTER,
)

logger = logging.getLogger(__name__)


def get_client() -> OpenSearch:
    hosts = [{"host": OS_HOST, "port": OS_PORT}]
    http_auth = (OS_USER, OS_PASSWORD)

    if ACTIVE_CLUSTER.startswith("es"):
        return OpenSearch(
            hosts=hosts,
            http_auth=http_auth,
        )

    return OpenSearch(
        hosts=hosts,
        http_auth=http_auth,
        use_ssl=OS_USE_SSL,
        verify_certs=False,
        ssl_show_warn=False,
    )


def ensure_index(client: OpenSearch) -> None:
    if not client.indices.exists(index=OS_INDEX):
        client.indices.create(index=OS_INDEX, body=INDEX_SETTINGS)
        logger.info("Created index: %s", OS_INDEX)
    else:
        logger.info("Index already exists: %s", OS_INDEX)


def bulk_index(client: OpenSearch, items: list[EnrichedKnowhow]) -> int:
    """Bulk index items in chunks to avoid memory/payload issues with large batches."""
    total_success = 0
    total_skipped = 0
    chunk_size = OS_BULK_CHUNK

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
                    "_index": OS_INDEX,
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
