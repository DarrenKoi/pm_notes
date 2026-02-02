"""Custom glossary retriever that returns JSON dicts for LangChain context injection."""

from retrieve.es_client import lookup_terms, GlossaryMatch


def retrieve_glossary(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """Retrieve glossary entries as JSON-serializable dicts.

    Returns:
        List of dicts with keys: term, aliases, definition, category, score
    """
    matches = lookup_terms(query, top_k=top_k, min_score=min_score)
    return [
        {
            "term": m.term,
            "aliases": m.aliases,
            "definition": m.definition,
            "category": m.category,
            "score": m.score,
        }
        for m in matches
    ]


def expand_query(original_query: str, matches: list[GlossaryMatch]) -> str:
    """Append canonical terms not already present in the query."""
    extra = [
        m.term for m in matches
        if m.term.lower() not in original_query.lower()
    ]
    if extra:
        return f"{original_query} {' '.join(extra)}"
    return original_query


def build_glossary_context(matches: list[GlossaryMatch]) -> str:
    """Build a glossary context block for LLM system prompt injection."""
    if not matches:
        return ""
    lines = ["[용어 사전 - 아래 정의를 참고하여 답변하세요]"]
    for m in matches:
        alias_str = f" ({', '.join(m.aliases)})" if m.aliases else ""
        lines.append(f"- {m.term}{alias_str}: {m.definition}")
    return "\n".join(lines)
