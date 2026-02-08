import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import config
from models import EnrichedKnowhow, KnowhowFile, KnowhowItem
from llm_processor import process_batch
from es_client import get_client, ensure_index, bulk_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("processed_data")


def load_json_files(input_dir: Path) -> list[KnowhowItem]:
    items = []
    for path in sorted(input_dir.glob("*.json")):
        logger.info("Loading %s", path.name)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        parsed = KnowhowFile.model_validate(raw)
        items.extend(parsed.data)
        logger.info("  → %d items (total: %d)", len(parsed.data), len(items))
    return items


def extract():
    """Step 1: Load raw data, run LLM enrichment, save processed results."""
    input_dir = Path("sample_data")
    batch_size = config.BATCH_SIZE

    items = load_json_files(input_dir)
    total = len(items)
    logger.info("Total items loaded: %d", total)

    all_enriched = []
    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        logger.info("=== Batch %d-%d / %d ===", start + 1, start + len(batch), total)
        enriched = process_batch(batch, offset=start, total=total)
        all_enriched.extend(enriched)

    PROCESSED_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PROCESSED_DIR / f"enriched_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in all_enriched], f, ensure_ascii=False, indent=2)

    logger.info("Extract complete. %d items saved to %s", len(all_enriched), output_path)


def index_to_es():
    """Step 2: Load processed data and index to Elasticsearch."""
    json_files = sorted(PROCESSED_DIR.glob("enriched_*.json"))
    if not json_files:
        logger.error("No processed files found in %s", PROCESSED_DIR)
        sys.exit(1)

    latest = json_files[-1]
    logger.info("Loading processed file: %s", latest.name)
    with open(latest, encoding="utf-8") as f:
        raw_list = json.load(f)

    enriched = [EnrichedKnowhow.model_validate(item) for item in raw_list]
    logger.info("Loaded %d enriched items", len(enriched))

    es = get_client()
    ensure_index(es)
    total_indexed = bulk_index(es, enriched)
    logger.info("Index complete. %d/%d documents indexed.", total_indexed, len(enriched))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("extract", "index"):
        print("Usage: python pipeline.py <extract|index>")
        sys.exit(1)

    command = sys.argv[1]
    if command == "extract":
        extract()
    elif command == "index":
        index_to_es()


if __name__ == "__main__":
    main()
