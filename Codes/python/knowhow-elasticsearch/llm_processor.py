import json
import logging
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import config
from models import EnrichedKnowhow, KnowhowItem

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowhow document analyzer. Given a knowhow text, extract structured information.
Return ONLY a JSON object with these fields:
- summary: A concise one-sentence summary in the same language as the input
- category: A single category label (e.g., "매뉴얼", "부품정보", "프로세스", "트러블슈팅", "약어/용어")
- keywords: A list of 3-5 relevant keywords

Example output:
{"summary": "웹사이트 검색 기능 사용 방법 안내", "category": "매뉴얼", "keywords": ["검색", "필터", "웹사이트"]}"""


def create_llm_client() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.LLM_URL,
        api_key="not-needed",
        model=config.LLM_MODEL,
        temperature=0.1,
    )



def enrich_item(client: ChatOpenAI, item: KnowhowItem) -> EnrichedKnowhow:
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=item.knowhow),
            ]
            resp = client.invoke(messages)
            raw = resp.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            return EnrichedKnowhow(
                **item.model_dump(),
                summary=parsed.get("summary", ""),
                category=parsed.get("category", ""),
                keywords=parsed.get("keywords", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Parse error on attempt %d for %s: %s", attempt + 1, item.KNOWHOW_ID, e)
        except Exception as e:
            logger.warning("LLM error on attempt %d for %s: %s", attempt + 1, item.KNOWHOW_ID, e)
            if attempt < config.LLM_MAX_RETRIES - 1:
                time.sleep(config.LLM_RETRY_DELAY)

    logger.error("Failed to enrich %s after %d attempts", item.KNOWHOW_ID, config.LLM_MAX_RETRIES)
    return EnrichedKnowhow(**item.model_dump())


def process_batch(
    items: list[KnowhowItem],
    offset: int = 0,
    total: int = 0,
) -> list[EnrichedKnowhow]:
    """Process a batch of items with LLM enrichment."""
    llm_client = create_llm_client()
    enriched = []

    for i, item in enumerate(items):
        logger.info("Processing %d/%d: %s", offset + i + 1, total or len(items), item.KNOWHOW_ID)
        enriched.append(enrich_item(llm_client, item))

    return enriched
