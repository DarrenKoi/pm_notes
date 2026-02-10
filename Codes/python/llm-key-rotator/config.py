from key_rotator import KeyRotator

LLM_BASE_URL = "http://common.llm.skhynix.com/v1"
LLM_MODEL = "gpt-oss-20b"
MAX_CONSECUTIVE_FAILURES = 5

rotator = KeyRotator(
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    max_consecutive_failures=MAX_CONSECUTIVE_FAILURES,
)
