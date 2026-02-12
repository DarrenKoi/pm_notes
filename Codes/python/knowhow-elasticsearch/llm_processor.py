import json
import logging
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from models import EnrichedKnowhow, KnowhowItem

LLM_URL = "http://common.llm.skhynix.com/v1"
LLM_MODEL = "gpt-oss-20b"
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2  # seconds

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a technical document analyzer specializing in extracting structured metadata from knowhow texts.

## Task
Analyze the given text and extract structured information. The input can be:
- A short definition explaining a term, abbreviation, or concept
- A procedural snippet describing work steps or processes
- A troubleshooting guide or solution description
- General technical documentation

## Output Format
Return ONLY a valid JSON object with exactly these fields:

{
  "category": "<single category from the list below>",
  "keywords": ["<keyword1>", "<keyword2>", ...],
  "summary": "<2-3 sentence summary>"
}

## Category Options (choose exactly one)
- "용어/약어": Definitions of terms, abbreviations, or technical concepts
- "프로세스": Step-by-step procedures, workflows, or operational instructions
- "트러블슈팅": Problem-solving guides, error resolutions, or debugging steps
- "매뉴얼": User guides, system manuals, or reference documentation
- "부품정보": Component specifications, part details, or hardware information
- "설정/구성": Configuration guides, setup instructions, or system settings
- "기타": Content that doesn't fit other categories

## Field Requirements

### category
- Select the single most appropriate category from the list above
- If the text defines a term or explains what something means → "용어/약어"
- If the text describes how to do something step-by-step → "프로세스"
- If the text explains how to fix or resolve an issue → "트러블슈팅"

### keywords
- Extract 3-5 domain-specific keywords
- Include technical terms, system names, and key concepts
- Use the original language of the keywords (do not translate)
- Prioritize nouns and noun phrases

### summary
- Write 2-3 sentences that capture the essential information
- Use the same language as the input text
- For term definitions: explain what the term means and its significance
- For procedures: summarize the goal and key steps
- For troubleshooting: state the problem and solution approach

## Examples

Input: "PVD(Physical Vapor Deposition)는 진공 상태에서 금속을 기화시켜 기판 위에 박막을 형성하는 기술이다."
Output:
{"category": "용어/약어", "keywords": ["PVD", "Physical Vapor Deposition", "진공", "박막", "기판"], "summary": "PVD(Physical Vapor Deposition)는 물리적 기상 증착 기술이다. 진공 환경에서 금속을 기화시켜 기판 표면에 얇은 막을 코팅하는 방식으로 작동한다."}

Input: "장비 PM 진행 시 1) 전원 차단 2) 잔류 가스 배출 3) 챔버 오픈 4) 파츠 교체 5) 리크 테스트 6) 시운전 순서로 진행한다."
Output:
{"category": "프로세스", "keywords": ["PM", "장비", "챔버", "파츠 교체", "리크 테스트"], "summary": "장비 예방정비(PM) 절차를 설명한다. 전원 차단부터 시작하여 가스 배출, 챔버 작업, 파츠 교체를 거쳐 리크 테스트와 시운전으로 마무리하는 6단계 프로세스이다."}

Input: "Interlock 발생 시 해당 센서 상태 확인 후 수동 리셋이 필요하며, 반복 발생 시 센서 교체를 검토한다."
Output:
{"category": "트러블슈팅", "keywords": ["Interlock", "센서", "수동 리셋", "센서 교체"], "summary": "Interlock 오류 발생 시 대응 방법을 설명한다. 먼저 센서 상태를 확인하고 수동 리셋을 수행하며, 문제가 반복될 경우 센서 교체를 고려해야 한다."}

## Rules
- Output ONLY the JSON object, no additional text or explanation
- Do not wrap the JSON in markdown code blocks
- Ensure the JSON is valid and parseable
- Never leave any field empty - provide reasonable values based on context"""


def create_llm_client() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LLM_URL,
        api_key="not-needed",
        model=LLM_MODEL,
        temperature=0.1,
    )



def enrich_item(client: ChatOpenAI, item: KnowhowItem) -> EnrichedKnowhow:
    for attempt in range(LLM_MAX_RETRIES):
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"다음 텍스트를 분석하세요:\n\n{item.knowhow}"),
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
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(LLM_RETRY_DELAY)

    logger.error("Failed to enrich %s after %d attempts", item.KNOWHOW_ID, LLM_MAX_RETRIES)
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
