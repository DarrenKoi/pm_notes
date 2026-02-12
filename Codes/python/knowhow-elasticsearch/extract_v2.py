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
PROGRESS_FILE = PROCESSED_DIR / "progress.jsonl"


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


def load_progress() -> tuple[list[EnrichedKnowhow], set[str]]:
    enriched = []
    processed_ids = set()
    if not PROGRESS_FILE.exists():
        return enriched, processed_ids
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = EnrichedKnowhow.model_validate(json.loads(line))
            enriched.append(item)
            processed_ids.add(item.KNOWHOW_ID)
    return enriched, processed_ids


def append_result(result: EnrichedKnowhow):
    PROCESSED_DIR.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")


def main():
    items = load_json_files(INPUT_DIR)
    total = len(items)
    logger.info("Total items loaded: %d", total)

    enriched, processed_ids = load_progress()
    logger.info("Already processed: %d/%d", len(processed_ids), total)

    remaining = [item for item in items if item.KNOWHOW_ID not in processed_ids]
    if not remaining:
        logger.info("All items already processed.")
        return

    logger.info("Remaining items: %d", len(remaining))
    llm_client = create_llm_client()
    done = len(enriched)

    for i, item in enumerate(remaining):
        done += 1
        logger.info("Processing [%d/%d] %s", done, total, item.KNOWHOW_ID)

        result = enrich_item(llm_client, item)
        append_result(result)

        if i < len(remaining) - 1:
            time.sleep(REQUEST_DELAY)

    logger.info("Done. %d/%d items processed.", done, total)


if __name__ == "__main__":
    main()
