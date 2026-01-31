# 보조 용어 사전 DB (Auxiliary Glossary DB) - Elasticsearch 기반

> Elasticsearch BM25 검색으로 사내 전문 용어를 관리하고, Milvus Dense Search를 보완하는 하이브리드 RAG 구조를 만든다.

---
tags: [rag, elasticsearch, glossary, hybrid-search, langgraph]
level: intermediate
last_updated: 2026-01-31
status: in-progress
---

## 왜 필요한가? (Why)

### Milvus Dense Search만으로는 부족한 이유

Dense 임베딩 기반 검색은 의미적 유사성에 강하지만, **도메인 전문 용어**에서 약점을 보인다:

1. **약어/줄임말 처리 취약**: "ETCH", "CMP", "CVD" 같은 반도체 공정 약어가 임베딩에 제대로 반영되지 않음
2. **동의어/변형 인식 불가**: 같은 개념의 다양한 표현(alias)을 연결하지 못함
3. **정의 부재**: 검색된 문서에 용어 정의가 없으면 LLM이 hallucination 생성

### 왜 Elasticsearch인가?

| 특성 | Elasticsearch | Redis | MongoDB |
|------|---------------|-------|---------|
| BM25 텍스트 검색 | ✅ 네이티브 | ❌ 없음 | △ 약함 |
| Fuzzy 매칭 | ✅ 내장 | ❌ | ❌ |
| 분석기/토크나이저 | ✅ 다양 | ❌ | ❌ |
| 용어 정규화 | ✅ 자동 | 수동 구현 | 수동 구현 |

Elasticsearch의 BM25 검색은 Milvus의 Dense Search와 자연스럽게 결합되어 **의사 하이브리드 검색(Pseudo-hybrid Search)**을 구현할 수 있다.

### 전체 아키텍처

```
User Query
    │
    ├─→ [1] ES Glossary Lookup (fuzzy + exact)
    │       → matched terms + definitions + canonical forms
    │
    ├─→ [2] Query Expansion: canonical terms 추가
    │       → expanded query → Milvus dense search
    │
    └─→ [3] Prompt Assembly:
            - Milvus retrieved docs
            - Glossary definitions (system context로 주입)
            - Original user query
            → LLM generates answer
```

## 핵심 개념 (What)

### 1. 용어 사전 엔트리 구조

비정형 텍스트에서 LLM을 활용해 용어를 추출하고, 다음 구조로 정규화한다:

```python
from dataclasses import dataclass, field

@dataclass
class GlossaryEntry:
    term: str                    # 정규 용어 (예: "Chemical Vapor Deposition")
    aliases: list[str]           # 변형/약어 (예: ["CVD", "화학기상증착"])
    definition: str              # 정의
    category: str                # 분류 (예: "deposition", "etch", "metrology")
    source_ids: list[str]        # 출처 스니펫 ID
```

### 2. 쿼리 확장 (Query Expansion)

사용자 쿼리에서 전문 용어를 감지하면, 정규 형태(canonical form)를 쿼리에 추가한다:

- 입력: `"CVD 공정 온도 설정 방법"`
- 용어 감지: `CVD` → canonical: `Chemical Vapor Deposition`
- 확장 쿼리: `"CVD Chemical Vapor Deposition 공정 온도 설정 방법"`

### 3. 컨텍스트 주입 (Context Injection)

감지된 용어의 정의를 LLM 시스템 프롬프트에 주입:

```
[용어 사전]
- CVD (Chemical Vapor Deposition): 기체 상태의 원료를 화학 반응시켜 ...
```

## 어떻게 사용하는가? (How)

### Step 1: 비정형 텍스트에서 용어 추출 (LLM-assisted)

```python
from openai import OpenAI

client = OpenAI()

EXTRACTION_PROMPT = """\
아래 텍스트 스니펫들에서 전문 용어를 추출하세요.

각 용어에 대해 JSON 배열로 반환:
[{
  "term": "정규 용어명 (영문 full name)",
  "aliases": ["약어", "한국어명", "변형"],
  "definition": "한 문장 정의",
  "category": "분류 키워드"
}]

중복 용어는 aliases를 병합하여 하나로 합치세요.

--- 텍스트 스니펫 ---
{snippets}
"""

def extract_glossary_batch(
    snippets: list[dict],  # [{"id": "...", "text": "..."}]
    batch_size: int = 10,
) -> list[dict]:
    """비정형 스니펫에서 용어를 배치 추출한다."""
    all_entries = []

    for i in range(0, len(snippets), batch_size):
        batch = snippets[i : i + batch_size]
        combined = "\n\n".join(
            f"[snippet_id: {s['id']}]\n{s['text']}" for s in batch
        )
        source_ids = [s["id"] for s in batch]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "전문 용어 추출기입니다."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(snippets=combined)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        import json
        entries = json.loads(resp.choices[0].message.content)
        # entries가 dict일 수 있음 ({"terms": [...]})
        if isinstance(entries, dict):
            entries = entries.get("terms", entries.get("results", []))

        for entry in entries:
            entry["source_ids"] = source_ids
        all_entries.extend(entries)

    return deduplicate_entries(all_entries)


def deduplicate_entries(entries: list[dict]) -> list[dict]:
    """동일 term을 가진 엔트리를 병합한다."""
    merged: dict[str, dict] = {}
    for e in entries:
        key = e["term"].lower().strip()
        if key in merged:
            existing = merged[key]
            existing["aliases"] = list(set(existing["aliases"] + e.get("aliases", [])))
            existing["source_ids"] = list(set(existing["source_ids"] + e.get("source_ids", [])))
            # definition은 더 긴 것을 유지
            if len(e.get("definition", "")) > len(existing.get("definition", "")):
                existing["definition"] = e["definition"]
        else:
            merged[key] = e
    return list(merged.values())
```

**추출 결과 예시:**

| 원본 스니펫 | 추출 결과 |
|-------------|-----------|
| "CVD 챔버에서 SiO2 박막을 증착할 때 온도는..." | `{ "term": "Chemical Vapor Deposition", "aliases": ["CVD"], "definition": "기체 원료를 화학 반응시켜 기판 위에 박막을 형성하는 공정", "category": "deposition" }` |

### Step 2: Elasticsearch 인덱스 설계 및 생성

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "glossary"

INDEX_BODY = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "term_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "trim"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "term":       {"type": "keyword"},                          # 정확 매칭
            "aliases":    {"type": "keyword"},                          # 약어/변형 정확 매칭
            "term_text":  {"type": "text", "analyzer": "term_analyzer"},# 퍼지 매칭
            "definition": {"type": "text", "analyzer": "standard"},     # 정의 전문 검색
            "category":   {"type": "keyword"},                          # 필터링
            "source_ids": {"type": "keyword"},                          # 출처 추적
            "updated_at": {"type": "date"}
        }
    }
}


def create_index():
    """인덱스를 생성한다. 이미 존재하면 삭제 후 재생성."""
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(index=INDEX_NAME, body=INDEX_BODY)
    print(f"Index '{INDEX_NAME}' created.")
```

### Step 3: 벌크 색인 (Ingestion Pipeline)

```python
from datetime import datetime, timezone
from elasticsearch.helpers import bulk


def build_bulk_actions(entries: list[dict]):
    """추출된 용어 리스트를 ES bulk action 형태로 변환한다."""
    for entry in entries:
        term = entry["term"]
        doc = {
            "_index": INDEX_NAME,
            "_id": term.lower().replace(" ", "_"),  # 중복 방지용 deterministic ID
            "_source": {
                "term": term,
                "aliases": entry.get("aliases", []),
                "term_text": term,  # text 필드에도 저장 (fuzzy 매칭용)
                "definition": entry.get("definition", ""),
                "category": entry.get("category", ""),
                "source_ids": entry.get("source_ids", []),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        # upsert: 같은 _id면 덮어쓰기
        yield doc


def ingest_glossary(entries: list[dict]):
    """용어 리스트를 ES에 벌크 색인한다."""
    success, errors = bulk(es, build_bulk_actions(entries), raise_on_error=False)
    print(f"Indexed {success} entries, {len(errors)} errors.")
    if errors:
        for err in errors[:5]:
            print(f"  Error: {err}")
    return success, errors
```

### Step 4: 쿼리 타임 용어 검색 및 확장

```python
from dataclasses import dataclass


@dataclass
class GlossaryMatch:
    term: str
    aliases: list[str]
    definition: str
    score: float


def lookup_terms(query: str, top_k: int = 5, fuzziness: str = "AUTO") -> list[GlossaryMatch]:
    """사용자 쿼리에서 전문 용어를 감지한다.

    exact match (term, aliases) + fuzzy match (term_text)를 조합한다.
    """
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    # 1) 정확 매칭 (높은 boost)
                    {"terms": {"term": query.split(), "boost": 3.0}},
                    {"terms": {"aliases": query.split(), "boost": 3.0}},
                    # 2) 퍼지 매칭
                    {
                        "match": {
                            "term_text": {
                                "query": query,
                                "fuzziness": fuzziness,
                                "boost": 1.5,
                            }
                        }
                    },
                    # 3) 정의 내 매칭 (낮은 boost)
                    {"match": {"definition": {"query": query, "boost": 0.5}}},
                ],
                "minimum_should_match": 1,
            }
        },
    }

    resp = es.search(index=INDEX_NAME, body=body)

    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append(GlossaryMatch(
            term=src["term"],
            aliases=src.get("aliases", []),
            definition=src.get("definition", ""),
            score=hit["_score"],
        ))
    return results


def expand_query(original_query: str, matches: list[GlossaryMatch]) -> str:
    """감지된 용어의 canonical form을 쿼리에 추가한다."""
    extra_terms = []
    for m in matches:
        # 이미 쿼리에 canonical term이 있으면 스킵
        if m.term.lower() not in original_query.lower():
            extra_terms.append(m.term)
    if extra_terms:
        return f"{original_query} {' '.join(extra_terms)}"
    return original_query


def build_glossary_context(matches: list[GlossaryMatch]) -> str:
    """LLM 시스템 프롬프트에 주입할 용어 정의 블록을 생성한다."""
    if not matches:
        return ""
    lines = ["[용어 사전 - 아래 정의를 참고하여 답변하세요]"]
    for m in matches:
        alias_str = f" ({', '.join(m.aliases)})" if m.aliases else ""
        lines.append(f"- {m.term}{alias_str}: {m.definition}")
    return "\n".join(lines)
```

**사용 예시:**

```python
query = "CVD 공정에서 온도 파라미터 설정"

# 1) 용어 감지
matches = lookup_terms(query, top_k=3)
# → [GlossaryMatch(term="Chemical Vapor Deposition", aliases=["CVD", ...], ...)]

# 2) 쿼리 확장 → Milvus 검색에 사용
expanded = expand_query(query, matches)
# → "CVD 공정에서 온도 파라미터 설정 Chemical Vapor Deposition"

# 3) 컨텍스트 생성 → LLM 프롬프트에 주입
glossary_ctx = build_glossary_context(matches)
# → "[용어 사전]\n- Chemical Vapor Deposition (CVD): 기체 원료를..."
```

### Step 5: LangGraph 노드 통합

기존 Corrective RAG 그래프에 두 개의 노드를 추가한다:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END


class RAGState(TypedDict):
    question: str
    glossary_matches: list[GlossaryMatch]
    expanded_query: str
    glossary_context: str
    documents: list[str]
    generation: str


def glossary_lookup(state: RAGState) -> dict:
    """[신규 노드] 쿼리에서 전문 용어를 감지하고 확장한다."""
    question = state["question"]

    matches = lookup_terms(question, top_k=5)
    expanded = expand_query(question, matches)
    context = build_glossary_context(matches)

    return {
        "glossary_matches": matches,
        "expanded_query": expanded,
        "glossary_context": context,
    }


def retrieve(state: RAGState) -> dict:
    """Milvus에서 문서를 검색한다. 확장된 쿼리를 사용."""
    query = state.get("expanded_query", state["question"])
    # Milvus 검색 (기존 retriever 활용)
    docs = milvus_retriever.search(query, top_k=5)
    return {"documents": docs}


def generate(state: RAGState) -> dict:
    """LLM으로 답변을 생성한다. 용어 정의를 시스템 프롬프트에 주입."""
    glossary_ctx = state.get("glossary_context", "")
    docs = state["documents"]
    question = state["question"]

    system_msg = "당신은 도움이 되는 AI 어시스턴트입니다."
    if glossary_ctx:
        system_msg += f"\n\n{glossary_ctx}"

    context_text = "\n\n".join(docs)
    user_msg = f"참고 문서:\n{context_text}\n\n질문: {question}"

    response = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])
    return {"generation": response.content}


# 그래프 구성
graph = StateGraph(RAGState)

graph.add_node("glossary_lookup", glossary_lookup)   # 신규
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("glossary_lookup")             # 진입점 변경
graph.add_edge("glossary_lookup", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()
```

**그래프 흐름:**

```
glossary_lookup → retrieve → generate → END
     │                │           │
     │ 용어 감지       │ 확장 쿼리  │ 용어 정의 +
     │ + 확장          │ → Milvus   │ 문서 → LLM
```

> **참고**: 기존 Corrective RAG 그래프와 결합할 때는 `glossary_lookup`을 진입점에,
> `prompt_enrichment` 로직을 `generate` 노드 내부에 통합하면 된다.
> 문서 평가(Grade) → 재검색 루프는 기존 그대로 유지.

## 운영 고려사항

### 용어 사전 업데이트

```python
def update_glossary_entry(term: str, updates: dict):
    """기존 용어를 부분 업데이트한다."""
    doc_id = term.lower().replace(" ", "_")
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    es.update(index=INDEX_NAME, id=doc_id, body={"doc": updates})
```

### 성능 팁

- **캐싱**: 자주 등장하는 용어 매칭 결과를 Redis/인메모리 캐시에 저장
- **비동기**: Milvus 검색과 ES 용어 검색을 동시에 실행 (`asyncio.gather`)
- **임계값**: ES score가 낮은 매칭은 필터링 (예: `score < 1.0`이면 제외)

### 품질 관리

- 추출된 용어는 주기적으로 사람이 검토 (sampling)
- 카테고리별 커버리지 모니터링
- 사용자 피드백 루프: 잘못된 용어 매칭 리포트

## 참고 자료 (References)

- [Elasticsearch 7.14 공식 문서 - Text Analysis](https://www.elastic.co/guide/en/elasticsearch/reference/7.14/analysis.html)
- [elasticsearch-py Bulk Helpers](https://elasticsearch-py.readthedocs.io/en/latest/helpers.html)
- [LangGraph 공식 문서](https://python.langchain.com/docs/langgraph)

## 관련 문서

- [LangGraph 기반 RAG](./langgraph/langgraph-rag.md)
- [Milvus 기초](./milvus/milvus-basics.md)
- [Milvus RAG 통합](./milvus/milvus-rag-integration.md)
