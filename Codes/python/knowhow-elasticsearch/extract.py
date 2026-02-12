import json
import logging
import time
from pathlib import Path

from models import EnrichedKnowhow, KnowhowFile, KnowhowItem
from llm_processor import create_llm_client, enrich_item

REQUEST_DELAY = 1.5  # seconds between each LLM request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = Path("sample_data")
PROCESSED_DIR = Path("processed_data")
PROGRESS_FILE = PROCESSED_DIR / "progress.json"


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


def load_progress() -> list[EnrichedKnowhow]:
    if not PROGRESS_FILE.exists():
        return []
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        return [EnrichedKnowhow.model_validate(item) for item in json.load(f)]


def save_progress(enriched: list[EnrichedKnowhow]):
    PROCESSED_DIR.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in enriched], f, ensure_ascii=False, indent=2)


def main():
    items = load_json_files(INPUT_DIR)
    total = len(items)
    logger.info("Total items loaded: %d", total)

    enriched = load_progress()
    processed_ids = {e.KNOWHOW_ID for e in enriched}
    logger.info("Already processed: %d/%d", len(processed_ids), total)

    remaining = [item for item in items if item.KNOWHOW_ID not in processed_ids]
    if not remaining:
        logger.info("All items already processed.")
        return

    logger.info("Remaining items: %d", len(remaining))
    llm_client = create_llm_client()

    for i, item in enumerate(remaining):
        logger.info("Processing [%d/%d] %s", len(enriched) + 1, total, item.KNOWHOW_ID)

        result = enrich_item(llm_client, item)
        enriched.append(result)
        save_progress(enriched)

        if i < len(remaining) - 1:
            time.sleep(REQUEST_DELAY)

    logger.info("Done. %d/%d items processed.", len(enriched), total)


if __name__ == "__main__":
    main()
