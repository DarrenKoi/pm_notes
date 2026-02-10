"""LLM Key Rotator 사용 예제.

3가지 패턴을 시연한다:
1. OpenAI SDK 래퍼 (chat_completion) — 가장 간단
2. LangChain 래퍼 (langchain_invoke) — LangChain 프로젝트용
3. Manual mode (get_key + report) — httpx 등 직접 호출 시
"""

from config import rotator


def example_openai_sdk():
    """패턴 1: OpenAI SDK 래퍼 — 가장 간단한 방법."""
    print("=" * 60)
    print("[패턴 1] OpenAI SDK — chat_completion()")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ]

    # 한 줄로 끝. 실패 시 자동으로 다음 key로 전환, 모든 key 소진 시 None 반환.
    result = rotator.chat_completion(messages, temperature=0.7)

    if result:
        print(f"Response: {result}")
    else:
        print("All keys exhausted — no response.")

    print()


def example_langchain():
    """패턴 2: LangChain 래퍼."""
    print("=" * 60)
    print("[패턴 2] LangChain — langchain_invoke()")
    print("=" * 60)

    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "What is 2 + 2?"),
    ]

    result = rotator.langchain_invoke(messages)

    if result:
        print(f"Response: {result}")
    else:
        print("All keys exhausted — no response.")

    print()


def example_manual():
    """패턴 3: Manual mode — 직접 HTTP 호출 시."""
    print("=" * 60)
    print("[패턴 3] Manual mode — get_key() + report")
    print("=" * 60)

    import httpx

    while True:
        key = rotator.get_key()
        if key is None:
            print("All keys exhausted.")
            break

        try:
            response = httpx.post(
                f"{rotator.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": rotator.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            print(f"Response: {data['choices'][0]['message']['content']}")
            rotator.report_success()
            break
        except Exception as e:
            print(f"Error: {e}")
            rotator.report_failure()

    print()


if __name__ == "__main__":
    # 현재 상태 확인
    print("Initial status:", rotator.status())
    print()

    # 원하는 패턴 하나만 주석 해제하여 실행
    example_openai_sdk()
    # example_langchain()
    # example_manual()

    # 최종 상태 확인
    print("Final status:", rotator.status())
