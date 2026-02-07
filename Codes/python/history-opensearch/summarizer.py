"""대화 요약 — Qwen3/Kimi2 via OpenAI-compatible local API.

재귀적 요약과 계층적 요약을 지원한다.
"""

import json
import logging

import httpx

import config

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM = """당신은 대화 요약 전문가입니다. 아래 규칙을 지키세요:
1. 사용자 선호와 결정사항을 반드시 보존
2. 미해결 질문/작업 유지
3. 잡담 및 중복 교환 제거
4. 구체적 팩트(이름, 숫자, 날짜) 유지
5. 한국어로 작성"""


def _chat(system: str, user: str) -> str:
    """LLM /v1/chat/completions 호출."""
    resp = httpx.post(
        f"{config.LLM_URL}/chat/completions",
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def summarize_messages(messages: list[dict], existing_summary: str = "") -> str:
    """재귀적 요약 — 기존 요약 + 새 메시지 → 갱신된 요약.

    Args:
        messages: [{"role": ..., "content": ...}, ...]
        existing_summary: 이전 요약 (없으면 빈 문자열)

    Returns:
        갱신된 요약 텍스트
    """
    formatted = "\n".join(
        f"[{m['role']}] {m['content']}" for m in messages
    )
    prompt = f"기존 요약:\n{existing_summary}\n\n새 대화:\n{formatted}\n\n위 내용을 종합하여 핵심 정보를 보존한 요약을 작성하세요."
    return _chat(SUMMARIZE_SYSTEM, prompt)


def extract_topics(summary: str) -> list[str]:
    """요약에서 주요 토픽을 추출."""
    prompt = (
        f"다음 대화 요약에서 주요 토픽 키워드를 3~5개 추출하세요.\n"
        f"JSON 배열로만 응답하세요. 예: [\"RAG\", \"벡터검색\"]\n\n{summary}"
    )
    raw = _chat("토픽을 추출하세요.", prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse topics: %s", raw)
        return []


def hierarchical_summarize(session_summaries: list[str]) -> str:
    """계층적 요약 — 여러 세션 요약을 하나로 통합.

    세션 1 요약 ─┐
    세션 2 요약 ─┼─→ 통합 요약
    세션 3 요약 ─┘
    """
    combined = "\n---\n".join(
        f"세션 {i + 1}: {s}" for i, s in enumerate(session_summaries)
    )
    prompt = (
        f"아래 여러 세션의 요약을 하나의 통합 요약으로 합쳐주세요.\n"
        f"사용자의 핵심 목표, 결정사항, 진행 상태를 보존하세요.\n\n{combined}"
    )
    return _chat(SUMMARIZE_SYSTEM, prompt)
