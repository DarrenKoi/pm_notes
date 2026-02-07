"""3계층 메모리 오케스트레이션.

흐름:
  1. 메시지 저장 (단기)
  2. 세션 종료 시 요약 생성 (중기)
  3. 팩트 추출 (장기)
  4. 컨텍스트 조립 — 사용자 쿼리에 메모리 주입
"""

import logging
from datetime import datetime

from opensearchpy import OpenSearch

import config
from models import Message, Session, UserFact, UserProfile
from embedding import embed_text
from summarizer import summarize_messages, extract_topics
from fact_extractor import extract_facts
import os_client

logger = logging.getLogger(__name__)


class MemoryManager:
    """3계층 메모리를 관리하는 핵심 클래스."""

    def __init__(self, client: OpenSearch | None = None):
        self.client = client or os_client.get_client()
        os_client.ensure_indices(self.client)

    # ── 단기 메모리 ──────────────────────────────

    def add_message(
        self, user_id: str, session_id: str, role: str, content: str
    ) -> str:
        """메시지를 단기 메모리에 저장."""
        vector = embed_text(content)
        msg = Message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            embedding=vector,
        )
        doc_id = os_client.index_message(self.client, msg)
        logger.info("Stored message %s [%s/%s]", doc_id, user_id, session_id)
        return doc_id

    def get_short_term(
        self, user_id: str, session_id: str
    ) -> list[dict]:
        """현재 세션의 최근 메시지 조회."""
        return os_client.get_recent_messages(
            self.client, user_id, session_id, limit=config.SHORT_TERM_LIMIT
        )

    # ── 중기 메모리 ──────────────────────────────

    def finalize_session(self, user_id: str, session_id: str) -> Session:
        """세션 종료 — 대화 요약 생성 및 저장."""
        messages = os_client.get_recent_messages(
            self.client, user_id, session_id, limit=200
        )
        if not messages:
            logger.warning("No messages for session %s", session_id)
            return Session(user_id=user_id, session_id=session_id, summary="")

        # Recursive summarization
        summary = summarize_messages(messages)
        topics = extract_topics(summary)
        summary_vector = embed_text(summary)

        session = Session(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            topics=topics,
            message_count=len(messages),
            embedding=summary_vector,
            start_time=datetime.fromisoformat(messages[0]["timestamp"]),
            end_time=datetime.fromisoformat(messages[-1]["timestamp"]),
        )
        os_client.index_session(self.client, session)
        logger.info(
            "Session %s finalized: %d msgs → summary (%d chars)",
            session_id,
            len(messages),
            len(summary),
        )

        # Extract long-term facts from this session
        self._extract_and_store_facts(user_id, messages)

        return session

    # ── 장기 메모리 ──────────────────────────────

    def _extract_and_store_facts(
        self, user_id: str, messages: list[dict]
    ) -> list[UserFact]:
        """대화에서 사용자 팩트를 추출하고 장기 메모리에 저장."""
        existing = os_client.get_all_facts(self.client, user_id)
        new_facts_raw = extract_facts(messages, existing)

        stored = []
        for f in new_facts_raw:
            vector = embed_text(f["fact"])
            fact = UserFact(
                user_id=user_id,
                fact=f["fact"],
                category=f.get("category", "pattern"),
                importance=f.get("importance", 5.0),
                embedding=vector,
            )
            os_client.index_fact(self.client, fact)
            stored.append(fact)

        logger.info("Extracted %d facts for user %s", len(stored), user_id)
        return stored

    def search_relevant_facts(
        self, user_id: str, query: str, top_k: int | None = None
    ) -> list[dict]:
        """쿼리와 관련된 장기 메모리 팩트를 벡터 검색."""
        query_vector = embed_text(query)
        return os_client.search_facts_by_vector(
            self.client,
            user_id,
            query_vector,
            top_k=top_k or config.LONG_MEMORY_TOP_K,
        )

    # ── 컨텍스트 조립 ─────────────────────────────

    def build_context(
        self, user_id: str, session_id: str, current_query: str
    ) -> UserProfile:
        """사용자 쿼리에 주입할 메모리 컨텍스트를 조립.

        Returns:
            UserProfile with recent messages, session summaries, relevant facts
        """
        recent_msgs = self.get_short_term(user_id, session_id)
        session_summaries = os_client.get_recent_sessions(
            self.client, user_id, limit=config.SESSION_SUMMARY_TOP_K
        )
        relevant_facts = self.search_relevant_facts(user_id, current_query)

        return UserProfile(
            user_id=user_id,
            recent_messages=[Message(**m) for m in recent_msgs],
            session_summaries=[Session(**s) for s in session_summaries],
            relevant_facts=[UserFact(**f) for f in relevant_facts],
        )

    def format_system_prompt(self, profile: UserProfile) -> str:
        """UserProfile을 시스템 프롬프트 문자열로 포맷."""
        parts = ["당신은 도움이 되는 AI 어시스턴트입니다."]

        if profile.relevant_facts:
            parts.append("\n[장기 메모리 - 사용자 정보]")
            for f in profile.relevant_facts:
                parts.append(f"- [{f.category}] {f.fact}")

        if profile.session_summaries:
            parts.append("\n[이전 세션 요약]")
            for s in profile.session_summaries:
                parts.append(f"- {s.summary}")

        return "\n".join(parts)
