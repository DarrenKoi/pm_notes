import json
import logging
from pathlib import Path

import config
from models import KnowhowFile, KnowhowItem
from llm_processor import process_batch
from es_client import get_client, ensure_index, bulk_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def main():
    input_dir = Path("sample_data")
    dry_run = False
    batch_size = config.BATCH_SIZE

    items = load_json_files(input_dir)
    total = len(items)
    logger.info("Total items loaded: %d", total)

    es = None
    if not dry_run:
        es = get_client()
        ensure_index(es)

    total_indexed = 0
    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        logger.info("=== Batch %d-%d / %d ===", start + 1, start + len(batch), total)

        enriched = process_batch(batch, offset=start, total=total)

        if dry_run:
            for e in enriched:
                logger.info("[%s] category=%s, keywords=%s, summary=%.60s", e.KNOWHOW_ID, e.category, e.keywords, e.summary)
        else:
            indexed = bulk_index(es, enriched)
            total_indexed += indexed

    if dry_run:
        logger.info("Dry run complete. %d items processed.", total)
    else:
        logger.info("Pipeline complete. %d/%d documents indexed.", total_indexed, total)


if __name__ == "__main__":
    main()
