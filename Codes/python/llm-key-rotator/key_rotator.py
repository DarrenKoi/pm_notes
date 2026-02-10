import os
import threading
from pathlib import Path

from dotenv import load_dotenv


class KeyRotator:
    """API key를 교대로 사용하는 로테이터.

    연속 실패 횟수가 threshold에 도달하면 자동으로 다음 key로 전환한다.
    모든 key가 소진되면 None을 반환한다 (에러를 발생시키지 않음).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        max_consecutive_failures: int = 5,
        env_path: str | None = None,
    ):
        self.base_url = base_url
        self.model = model
        self.max_consecutive_failures = max_consecutive_failures
        self._lock = threading.Lock()

        # .env 로드
        dotenv_path = env_path or Path(__file__).parent / ".env"
        load_dotenv(dotenv_path)

        # API_KEY1 ~ API_KEY5 로드
        self._keys: list[str] = []
        for i in range(1, 6):
            key = os.getenv(f"API_KEY{i}")
            if key and key != f"your-api-key-{i}":
                self._keys.append(key)

        if not self._keys:
            print("[KeyRotator] WARNING: No API keys loaded from .env")

        self._failure_counts: list[int] = [0] * len(self._keys)
        self._exhausted: list[bool] = [False] * len(self._keys)
        self._current_index: int = 0

        # SDK 클라이언트 캐시
        self._openai_client = None
        self._openai_client_key_index: int | None = None
        self._langchain_client = None
        self._langchain_client_key_index: int | None = None

    # ── Public API ──────────────────────────────────────────────

    def get_key(self) -> str | None:
        """현재 활성 API key를 반환한다. 모든 key 소진 시 None."""
        with self._lock:
            return self._get_active_key()

    def report_success(self) -> None:
        """현재 key의 실패 카운터를 리셋한다."""
        with self._lock:
            if self._get_active_key() is not None:
                self._failure_counts[self._current_index] = 0

    def report_failure(self) -> None:
        """현재 key의 실패 카운터를 증가시킨다.

        threshold 도달 시 해당 key를 exhausted 처리하고 다음 key로 전환한다.
        """
        with self._lock:
            if self._get_active_key() is None:
                return

            self._failure_counts[self._current_index] += 1
            idx = self._current_index
            count = self._failure_counts[idx]

            if count >= self.max_consecutive_failures:
                self._exhausted[idx] = True
                print(
                    f"[KeyRotator] Key #{idx + 1} exhausted "
                    f"({count} consecutive failures). Switching..."
                )
                self._advance_to_next_key()

    def get_openai_client(self):
        """OpenAI SDK 클라이언트를 반환한다. key 전환 시 자동 재생성."""
        with self._lock:
            key = self._get_active_key()
            if key is None:
                return None

            if (
                self._openai_client is not None
                and self._openai_client_key_index == self._current_index
            ):
                return self._openai_client

            from openai import OpenAI

            self._openai_client = OpenAI(
                api_key=key,
                base_url=self.base_url,
            )
            self._openai_client_key_index = self._current_index
            return self._openai_client

    def get_langchain_client(self, **kwargs):
        """ChatOpenAI 클라이언트를 반환한다. key 전환 시 자동 재생성."""
        with self._lock:
            key = self._get_active_key()
            if key is None:
                return None

            if (
                self._langchain_client is not None
                and self._langchain_client_key_index == self._current_index
            ):
                return self._langchain_client

            from langchain_openai import ChatOpenAI

            self._langchain_client = ChatOpenAI(
                api_key=key,
                base_url=self.base_url,
                model=self.model,
                **kwargs,
            )
            self._langchain_client_key_index = self._current_index
            return self._langchain_client

    def chat_completion(self, messages: list[dict], **kwargs) -> str | None:
        """OpenAI SDK chat completion 래퍼.

        자동으로 retry + key rotation을 수행한다.
        모든 key 소진 시 None을 반환한다.
        """
        while True:
            client = self.get_openai_client()
            if client is None:
                print("[KeyRotator] All keys exhausted. Returning None.")
                return None

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                self.report_success()
                return response.choices[0].message.content
            except Exception as e:
                print(f"[KeyRotator] API error: {e}")
                self.report_failure()

    def langchain_invoke(self, messages: list, **kwargs) -> str | None:
        """LangChain invoke 래퍼.

        자동으로 retry + key rotation을 수행한다.
        모든 key 소진 시 None을 반환한다.
        """
        while True:
            llm = self.get_langchain_client()
            if llm is None:
                print("[KeyRotator] All keys exhausted. Returning None.")
                return None

            try:
                response = llm.invoke(messages, **kwargs)
                self.report_success()
                return response.content
            except Exception as e:
                print(f"[KeyRotator] API error: {e}")
                self.report_failure()

    def status(self) -> dict:
        """현재 상태를 dict로 반환한다 (디버깅용)."""
        with self._lock:
            return {
                "total_keys": len(self._keys),
                "current_index": self._current_index,
                "current_key_preview": self._key_preview(self._current_index),
                "failure_counts": list(self._failure_counts),
                "exhausted": list(self._exhausted),
                "all_exhausted": all(self._exhausted) if self._keys else True,
            }

    def reset_all_keys(self) -> None:
        """모든 key의 상태를 초기화한다 (quota 갱신 후 사용)."""
        with self._lock:
            self._failure_counts = [0] * len(self._keys)
            self._exhausted = [False] * len(self._keys)
            self._current_index = 0
            self._openai_client = None
            self._openai_client_key_index = None
            self._langchain_client = None
            self._langchain_client_key_index = None
            print("[KeyRotator] All keys reset.")

    # ── Internal ────────────────────────────────────────────────

    def _get_active_key(self) -> str | None:
        """현재 활성 key를 반환한다 (lock 없이 내부 호출용)."""
        if not self._keys or all(self._exhausted):
            return None
        if self._exhausted[self._current_index]:
            self._advance_to_next_key()
            if all(self._exhausted):
                return None
        return self._keys[self._current_index]

    def _advance_to_next_key(self) -> None:
        """다음 사용 가능한 key로 이동한다."""
        start = self._current_index
        for _ in range(len(self._keys)):
            self._current_index = (self._current_index + 1) % len(self._keys)
            if not self._exhausted[self._current_index]:
                print(
                    f"[KeyRotator] Switched to Key #{self._current_index + 1} "
                    f"({self._key_preview(self._current_index)})"
                )
                # 클라이언트 캐시 무효화
                self._openai_client = None
                self._langchain_client = None
                return
        # 모든 key 소진
        self._current_index = start

    def _key_preview(self, index: int) -> str:
        """key의 앞 8자만 보여주는 미리보기."""
        if index < 0 or index >= len(self._keys):
            return "N/A"
        key = self._keys[index]
        return f"{key[:8]}..." if len(key) > 8 else key
