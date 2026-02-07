"""데모 시나리오 — 3계층 메모리 시스템 동작 확인.

Usage:
    python main.py
"""

import logging

from memory_manager import MemoryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

USER_ID = "user-001"
SESSION_ID = "session-demo-001"


def demo_conversation(mm: MemoryManager) -> None:
    """시뮬레이션 대화 — 메시지 저장 → 세션 종료 → 컨텍스트 조립."""

    # 1) 대화 메시지 저장 (단기 메모리)
    turns = [
        ("user", "안녕하세요. 저는 FastAPI로 RAG 시스템을 만들고 있어요."),
        ("assistant", "반갑습니다! FastAPI + RAG 조합이군요. 어떤 벡터 DB를 사용하시나요?"),
        ("user", "OpenSearch를 벡터 DB로 쓰고 있고, 임베딩은 BGE-M3를 사용합니다."),
        ("assistant", "좋은 선택이네요. k-NN 플러그인으로 HNSW 인덱스를 설정하면 됩니다."),
        ("user", "Python을 주로 쓰고 TypeScript도 약간 합니다. 비동기 처리가 좀 어려워요."),
        ("assistant", "FastAPI는 async/await을 기본 지원하니 좋은 출발점입니다."),
        ("user", "맞아요. 그리고 사용자 대화 이력도 저장하고 싶어요."),
        ("assistant", "대화 메모리 시스템을 구축하면 됩니다. 단기/중기/장기 3계층으로 설계하는 것을 추천합니다."),
        ("user", "로컬 LLM으로 Qwen3를 쓰고 있는데, 요약 품질이 괜찮을까요?"),
        ("assistant", "Qwen3는 요약 태스크에서 충분한 성능을 보입니다. 프롬프트 엔지니어링이 중요합니다."),
    ]

    logger.info("=== Step 1: Storing messages (short-term) ===")
    for role, content in turns:
        mm.add_message(USER_ID, SESSION_ID, role, content)

    # 2) 세션 종료 → 중기 요약 + 장기 팩트 추출
    logger.info("=== Step 2: Finalizing session (mid-term + long-term) ===")
    session = mm.finalize_session(USER_ID, SESSION_ID)
    logger.info("Session summary: %s", session.summary[:200])
    logger.info("Topics: %s", session.topics)

    # 3) 새 세션에서 컨텍스트 조립
    logger.info("=== Step 3: Building context for new query ===")
    new_query = "OpenSearch에서 하이브리드 검색은 어떻게 구현하나요?"
    profile = mm.build_context(USER_ID, "session-demo-002", new_query)

    system_prompt = mm.format_system_prompt(profile)
    logger.info("--- System prompt ---\n%s", system_prompt)
    logger.info("--- Recent messages: %d ---", len(profile.recent_messages))
    logger.info("--- Session summaries: %d ---", len(profile.session_summaries))
    logger.info("--- Relevant facts: %d ---", len(profile.relevant_facts))


def main() -> None:
    logger.info("Initializing MemoryManager...")
    mm = MemoryManager()
    demo_conversation(mm)
    logger.info("Demo complete.")


if __name__ == "__main__":
    main()
