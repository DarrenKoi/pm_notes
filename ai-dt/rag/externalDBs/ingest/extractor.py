"""LLM-based glossary term extraction with batch processing."""

import json

from openai import OpenAI

from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, BATCH_SIZE
from models import GlossaryEntry

EXTRACTION_PROMPT = """\
아래 텍스트 스니펫들에서 전문 용어를 추출하세요.

각 용어에 대해 JSON 객체로 반환:
{{"terms": [
  {{
    "term": "정규 용어명 (영문 full name)",
    "aliases": ["약어", "한국어명", "변형"],
    "definition": "한 문장 정의",
    "category": "분류 키워드"
  }}
]}}

중복 용어는 aliases를 병합하여 하나로 합치세요.

--- 텍스트 스니펫 ---
{snippets}
"""


def _get_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def extract_glossary_batch(
    snippets: list[dict],
    batch_size: int = BATCH_SIZE,
) -> list[GlossaryEntry]:
    """Extract glossary terms from snippets using LLM in batches."""
    client = _get_client()
    all_entries: list[dict] = []

    total = len(snippets)
    for i in range(0, total, batch_size):
        batch = snippets[i : i + batch_size]
        combined = "\n\n".join(
            f"[snippet_id: {s['id']}]\n{s['text']}" for s in batch
        )
        source_ids = [s["id"] for s in batch]

        print(f"  Extracting batch {i // batch_size + 1} ({i+1}-{min(i+batch_size, total)}/{total})")

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "전문 용어 추출기입니다. 반드시 JSON으로 응답하세요."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(snippets=combined)},
            ],
            temperature=0.0,
        )

        raw = resp.choices[0].message.content
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
            else:
                print(f"  WARNING: Could not parse LLM response, skipping batch")
                continue

        entries = parsed if isinstance(parsed, list) else parsed.get("terms", parsed.get("results", []))

        for entry in entries:
            entry["source_ids"] = source_ids
        all_entries.extend(entries)

    deduped = _deduplicate(all_entries)
    return [GlossaryEntry(**e) for e in deduped]


def _deduplicate(entries: list[dict]) -> list[dict]:
    """Merge entries with the same normalized term."""
    merged: dict[str, dict] = {}
    for e in entries:
        key = e["term"].lower().strip()
        if key in merged:
            existing = merged[key]
            existing["aliases"] = list(set(existing["aliases"] + e.get("aliases", [])))
            existing["source_ids"] = list(set(existing["source_ids"] + e.get("source_ids", [])))
            if len(e.get("definition", "")) > len(existing.get("definition", "")):
                existing["definition"] = e["definition"]
        else:
            merged[key] = e
    return list(merged.values())
