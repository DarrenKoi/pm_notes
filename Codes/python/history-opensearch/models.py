from datetime import datetime

from pydantic import BaseModel, Field


class Message(BaseModel):
    """단기 메모리 - 개별 메시지"""

    user_id: str
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    embedding: list[float] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class Session(BaseModel):
    """중기 메모리 - 세션 요약"""

    user_id: str
    session_id: str
    summary: str
    topics: list[str] = Field(default_factory=list)
    message_count: int = 0
    embedding: list[float] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None


class UserFact(BaseModel):
    """장기 메모리 - 사용자 팩트"""

    user_id: str
    fact: str
    category: str  # preference | goal | skill | pattern
    importance: float = 5.0  # 1-10
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)


class UserProfile(BaseModel):
    """컨텍스트 조립용 - 사용자 프로필 스냅샷"""

    user_id: str
    recent_messages: list[Message] = Field(default_factory=list)
    session_summaries: list[Session] = Field(default_factory=list)
    relevant_facts: list[UserFact] = Field(default_factory=list)
