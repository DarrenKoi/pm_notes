import json
import logging
import sys
from pathlib import Path

from models import EnrichedKnowhow
from os_settings import OS_INDEX, INDEX_SETTINGS, get_connection_config  # triggers _path_setup
import opensearch_handler as osh

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("processed_data")
PROGRESS_JSONL = PROCESSED_DIR / "progress.jsonl"
PROGRESS_JSON = PROCESSED_DIR / "progress.json"


def load_enriched() -> list[EnrichedKnowhow]:
    # JSONL (extract_v2)
    if PROGRESS_JSONL.exists():
        logger.info("Loading: %s", PROGRESS_JSONL.name)
        items = []
        with open(PROGRESS_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(EnrichedKnowhow.model_validate(json.loads(line)))
        return items

    # JSON (extract v1)
    if PROGRESS_JSON.exists():
        source = PROGRESS_JSON
    else:
        json_files = sorted(PROCESSED_DIR.glob("enriched_*.json"))
        if not json_files:
            logger.error("No processed files found in %s", PROCESSED_DIR)
            sys.exit(1)
        source = json_files[-1]

    logger.info("Loading: %s", source.name)
    with open(source, encoding="utf-8") as f:
        raw_list = json.load(f)
    return [EnrichedKnowhow.model_validate(item) for item in raw_list]


def main():
    enriched = load_enriched()
    logger.info("Loaded %d enriched items", len(enriched))

    config = get_connection_config()
    client = osh.create_client(config=config)

    # Ensure index exists
    if not osh.index_exists(client, OS_INDEX):
        osh.create_index(
            client,
            OS_INDEX,
            mappings=INDEX_SETTINGS["mappings"],
            settings=INDEX_SETTINGS["settings"],
        )
        logger.info("Created index: %s", OS_INDEX)
    else:
        logger.info("Index already exists: %s", OS_INDEX)

    # Filter empty-keyword items and prepare docs
    docs = []
    skipped = 0
    for item in enriched:
        if not item.keywords:
            skipped += 1
            continue
        docs.append(item.model_dump(exclude={"knowhow_no"}))

    if skipped:
        logger.info("Skipped %d items with empty keywords", skipped)

    if docs:
        success, errors = osh.bulk_index(
            client, OS_INDEX, docs, chunk_size=config.bulk_chunk
        )
        logger.info("Index complete. %d/%d documents indexed.", success, len(enriched))
    else:
        logger.info("No documents to index.")


if __name__ == "__main__":
    main()
