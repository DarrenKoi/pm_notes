import json
import logging
import sys
from pathlib import Path

from models import EnrichedKnowhow
from es_client import get_client, ensure_index, bulk_index

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

    es = get_client()
    ensure_index(es)
    total_indexed = bulk_index(es, enriched)
    logger.info("Index complete. %d/%d documents indexed.", total_indexed, len(enriched))


if __name__ == "__main__":
    main()
