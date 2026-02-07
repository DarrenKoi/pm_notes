"""BGE-M3 임베딩 — OpenAI-compatible 로컬 API 호출."""

import logging

import httpx

import config

logger = logging.getLogger(__name__)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트를 BGE-M3 임베딩 벡터로 변환.

    POST /v1/embeddings (OpenAI-compatible)
    """
    resp = httpx.post(
        f"{config.EMBEDDING_URL}/embeddings",
        json={"model": config.EMBEDDING_MODEL, "input": texts},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # API returns objects sorted by index
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]


def embed_text(text: str) -> list[float]:
    """단일 텍스트 임베딩."""
    return embed_texts([text])[0]
