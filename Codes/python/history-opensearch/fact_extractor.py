"""사용자 팩트 추출 — 대화에서 장기 메모리용 구조화된 팩트를 추출."""

import json
import logging

import httpx

import config

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM = """당신은 사용자 정보 추출 전문가입니다.
대화 내용에서 사용자에 대한 중요한 팩트를 추출합니다.

카테고리:
- preference: 선호/비선호 (예: "Python을 Java보다 선호")
- goal: 목표/계획 (예: "RAG 시스템 구축 중")
- skill: 기술/역량 (예: "FastAPI 사용 경험 있음")
- pattern: 행동 패턴 (예: "매주 월요일에 코드 리뷰")

규칙:
1. 명시적으로 언급되거나 강하게 암시된 것만 추출
2. 일시적/세션 한정 정보는 제외
3. 각 팩트에 중요도(1-10) 부여
4. JSON 배열로 반환"""


def extract_facts(messages: list[dict], existing_facts: list[dict] | None = None) -> list[dict]:
    """대화에서 사용자 팩트 추출.

    Args:
        messages: [{"role": ..., "content": ...}, ...]
        existing_facts: 기존 팩트 목록 (충돌 시 새 것 우선)

    Returns:
        [{"fact": ..., "category": ..., "importance": ...}, ...]
    """
    formatted = "\n".join(
        f"[{m['role']}] {m['content']}" for m in messages
    )
    existing_str = ""
    if existing_facts:
        existing_str = f"\n기존 팩트:\n{json.dumps(existing_facts, ensure_ascii=False, indent=2)}\n"

    prompt = (
        f"{existing_str}\n"
        f"대화 내용:\n{formatted}\n\n"
        f"위 대화에서 사용자 팩트를 추출하세요.\n"
        f"기존 팩트와 충돌하면 새 것으로 교체하세요.\n"
        f'JSON 배열로 반환: [{{"fact": "...", "category": "...", "importance": N}}, ...]'
    )

    resp = httpx.post(
        f"{config.LLM_URL}/chat/completions",
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        },
        timeout=60.0,
    )
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]

    try:
        facts = json.loads(raw)
        if not isinstance(facts, list):
            logger.warning("Expected list, got %s", type(facts))
            return []
        return facts
    except json.JSONDecodeError:
        logger.warning("Failed to parse facts JSON: %s", raw)
        return []
