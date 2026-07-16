# 멀티에이전트 RAG 통합

> Supervisor 패턴과 SKILL.md 기반 SubAgent로 엑셀·SQL·RAG Agent를 오케스트레이션하는 통합 시스템을 구축한다

---
tags: [multi-agent, supervisor, subagent, skill-md, orchestration, langgraph]
level: advanced
last_updated: 2026-07-16
---

## 왜 필요한가? (Why)

실무에서는 **하나의 데이터 소스**로는 질문에 충분히 답변할 수 없다. "PRJ-001의 예산(엑셀), 태스크 현황(DB), 관련 절차(문서)를 종합 분석해줘" 같은 복합 질문은 **여러 전문 Agent를 조합**해야 한다.

| 단일 Agent 한계 | 멀티에이전트 해결 |
|----------------|-----------------|
| 하나의 데이터 소스만 접근 | 엑셀/SQL/문서 각각 전문 Agent |
| 컨텍스트 블로트 (모든 도구 로드) | SubAgent별 독립 컨텍스트 |
| 라우팅 로직이 코드에 하드코딩 | Supervisor LLM이 자율 판단 |
| 확장 시 코드 전체 수정 | SKILL.md 추가만으로 Agent 확장 |

이 문서에서는 두 가지 통합 패턴을 다룬다:

1. **서브에이전트 래핑 패턴** — 기존 Agent를 `@tool`로 감싸 Supervisor가 도구처럼 호출
2. **SKILL.md 기반 SubAgent 패턴** — 선언적 파일로 Agent 행동을 정의하고 오케스트레이션

---

## 패턴 1: 서브에이전트 래핑 (Tool Wrapping)

### 핵심 개념

앞서 구축한 `rag_agent`(Agentic RAG StateGraph)를 **`@tool` 데코레이터**로 감싸면, Supervisor가 일반 도구처럼 호출할 수 있다.

```
사용자 질문
  → [Supervisor] 질문 유형 판별
    → [search_pm_docs]    → rag_agent 호출 → 문서 검색 답변
    → [analyze_methodology] → LLM 직접 호출 → 분석 답변
  → [Supervisor] 결과 통합 → 최종 응답
```

### 구현

#### 1단계: RAG Agent를 도구로 래핑

```python
from langchain.tools import tool

@tool
def search_pm_docs(query: str) -> str:
    """PM 문서를 검색하고 답변을 생성합니다.
    리스크 관리, 스프린트, 품질 검수 등 구체적 PM 절차 질문에 사용하세요."""
    result = rag_agent.invoke({
        "messages": [],
        "question": query,
        "documents": [],
        "generation": "",
    })
    answer = result.get("generation", "(검색 실패)")
    sources = [
        os.path.basename(d.metadata.get("source", ""))
        for d in result.get("documents", [])
    ]
    return f"[검색 문서: {sources}]\n{answer}"
```

**핵심 포인트:**

| 항목 | 설명 |
|------|------|
| `@tool` 데코레이터 | 함수를 LangChain Tool로 변환 |
| docstring | Supervisor가 도구 선택 시 참조하는 설명 — **라우팅 정확도의 핵심** |
| 반환값 | 문자열로 반환해야 Supervisor가 해석 가능 |
| 출처 정보 포함 | `[검색 문서: [...]]`를 앞에 붙여 근거 추적 가능 |

**docstring 작성 원칙:** Supervisor LLM이 이 설명을 읽고 도구 선택을 판단하므로, **어떤 유형의 질문에 사용해야 하는지** 명확히 기술한다.

```python
# 좋은 예: 구체적 사용 시나리오 명시
"""PM 문서를 검색하고 답변을 생성합니다.
리스크 관리, 스프린트, 품질 검수 등 구체적 PM 절차 질문에 사용하세요."""

# 나쁜 예: 너무 일반적
"""문서를 검색합니다."""
```

#### 2단계: 분석 Agent 도구 정의

벡터스토어를 사용하지 않는 **LLM 추론 기반 분석 도구**를 별도로 정의한다.

```python
@tool
def analyze_pm_methodology(query: str) -> str:
    """PM 방법론을 비교·분석합니다.
    Agile vs Waterfall 비교, 프로세스 장단점, 방법론 추천 등 분석적 질문에 사용하세요."""
    analysis_prompt = (
        "당신은 프로젝트 관리(PM) 방법론 전문 분석가입니다.\n"
        "아래 질문에 대해 근거 기반의 분석을 제공하세요.\n"
        "가능하면 표, 비교 항목, 장단점을 포함하세요.\n\n"
        f"질문: {query}"
    )
    response = llm.invoke(analysis_prompt)
    return response.content
```

RAG Agent는 **문서에 있는 내용**을 검색하고, 분석 Agent는 **LLM의 도메인 지식**으로 추론한다. 이 역할 분리가 멀티에이전트의 핵심이다.

#### 3단계: Supervisor 생성

```python
from langchain.agents import create_agent

SUPERVISOR_PROMPT = """당신은 프로젝트 관리(PM) 팀의 감독 에이전트입니다.
사용자의 질문을 분석하여 적절한 전문 에이전트에게 위임하세요:

- PM 문서에 있는 구체적 절차·기준·체크리스트 질문 → search_pm_docs 사용
- PM 방법론 비교, 장단점 분석, 추천 요청 → analyze_pm_methodology 사용
- 복합 질문은 두 도구를 모두 활용하세요

모든 결과를 종합하여 한국어로 명확한 최종 답변을 제공하세요.
"""

supervisor = create_agent(
    model=llm,
    tools=[search_pm_docs, analyze_pm_methodology],
    system_prompt=SUPERVISOR_PROMPT,
)
```

**Supervisor 프롬프트 설계 원칙:**

| 요소 | 역할 | 예시 |
|------|------|------|
| 역할 정의 | LLM에게 감독자 역할 부여 | "감독 에이전트입니다" |
| 라우팅 규칙 | 질문 유형 → 도구 매핑 | "절차 질문 → search_pm_docs" |
| 복합 질문 처리 | 여러 도구 순차 호출 지시 | "복합 질문은 두 도구를 모두" |
| 출력 형식 | 결과 통합 방법 | "한국어로 명확한 최종 답변" |

#### 4단계: 테스트

```python
# 테스트 1: 문서 검색 질문 → RAG Agent 라우팅
result1 = supervisor.invoke(
    {"messages": [{"role": "user", "content": "리스크 관리 절차서에서 리스크 식별 단계의 구체적 절차를 알려주세요."}]},
)

# 테스트 2: 분석 질문 → 분석 Agent 라우팅
result2 = supervisor.invoke(
    {"messages": [{"role": "user", "content": "Agile과 Waterfall 방법론의 장단점을 비교해주세요."}]},
)

# 테스트 3: 복합 질문 → 두 Agent 순차 호출
result3 = supervisor.invoke(
    {"messages": [{"role": "user", "content": "품질 검수 체크리스트를 검색하고, CI/CD에 자동화할 항목을 분석해주세요."}]},
)
```

| 테스트 | 유형 | 기대 라우팅 |
|--------|------|------------|
| 1 | 문서 검색 | `search_pm_docs` → RAG Agent |
| 2 | 방법론 분석 | `analyze_pm_methodology` → 분석 Agent |
| 3 | 복합 질문 | 두 Agent 순차 호출 |

### 반도체 공정 도메인 적용

동일 패턴을 반도체 공정 도메인에 적용하면:

```python
@tool
def search_process_docs(query: str) -> str:
    """반도체 공정 문서를 검색하고 답변을 생성합니다.
    Particle 불량, CMP 장비, Etch 공정 등 구체적 공정 질문에 사용하세요."""
    result = rag_agent.invoke({
        "messages": [], "question": query, "documents": [], "generation": "",
    })
    answer = result.get("generation", "(검색 실패)")
    sources = [os.path.basename(d.metadata.get("source", "")) for d in result.get("documents", [])]
    return f"[검색 문서: {sources}]\n{answer}"

@tool
def analyze_process_data(query: str) -> str:
    """반도체 공정 데이터를 분석합니다.
    수율 트렌드, 공정 파라미터 비교, 불량 원인 분석, 개선 방안 도출 등 분석적 질문에 사용하세요."""
    analysis_prompt = (
        "당신은 반도체 공정 데이터 분석 전문가입니다.\n"
        "아래 질문에 대해 반도체 제조 도메인 지식을 활용하여 분석을 제공하세요.\n"
        "가능하면 수치 기반 트렌드, 원인-결과 관계, 개선 방안을 포함하세요.\n\n"
        f"질문: {query}"
    )
    response = llm.invoke(analysis_prompt)
    return response.content

SUPERVISOR_PROMPT = """당신은 반도체 공정 팀의 감독 에이전트입니다.
- 공정 매뉴얼·장비 스펙·불량 조치 가이드 등 문서 검색 → search_process_docs 사용
- 수율 트렌드·공정 파라미터 분석·불량 원인 추론 → analyze_process_data 사용
- 복합 질문은 두 도구를 모두 활용하세요

모든 결과를 종합하여 한국어로 명확한 최종 답변을 제공하세요.
"""

supervisor = create_agent(
    model=llm,
    tools=[search_process_docs, analyze_process_data],
    system_prompt=SUPERVISOR_PROMPT,
)
```

---

## 패턴 2: SKILL.md 기반 SubAgent 오케스트레이션

### 핵심 개념

**SKILL.md**는 Agent의 행동을 **선언적으로 정의하는 파일**이다. SubAgent의 `system_prompt`로 로드되어 Agent의 역할, 도구, 행동 규칙을 명시한다.

| 구성 요소 | 역할 |
|-----------|------|
| **SKILL.md** | 각 Agent의 행동·도구·역할을 선언적으로 정의 |
| **SubAgent** | SKILL.md를 system_prompt로 로드하여 독립 컨텍스트에서 실행 |
| **Supervisor** | SubAgent들을 오케스트레이션하는 상위 Agent |

```
사용자 질문
  → Supervisor
    → SubAgent Harness
      → {엑셀 SubAgent, SQL SubAgent, RAG SubAgent}
    → 통합 응답
          ↑
     SKILL.md가 각 SubAgent의 행동을 선언적으로 정의
```

### SubAgent vs create_agent 비교

| 항목 | create_agent (기존) | SubAgent + SKILL.md |
|------|-------------------|---------------------|
| 행동 정의 | system_prompt 문자열 직접 작성 | SKILL.md 파일로 선언적 관리 |
| 컨텍스트 | 메인 Agent와 공유 (블로트 위험) | 독립된 컨텍스트 (격리) |
| 결과 전달 | 전체 메시지 히스토리 공유 | 요약만 메인 Agent에 반환 |
| 재사용성 | 코드에 종속 | SKILL.md 파일로 이식 가능 |
| 확장 | 코드 수정 필요 | SKILL.md 추가만으로 Agent 확장 |

### SKILL.md 구조

```yaml
---
name: agent-name           # 스킬 식별자 (소문자+하이픈)
description: >             # 설명 — SubAgent 매칭에 사용
  이 Agent가 수행하는 역할을 기술합니다.
allowed-tools: tool1 tool2 # 사용 가능한 도구 목록
---

# Agent 행동 지침 (본문)

## 역할
이 Agent가 무엇을 하는지 정의합니다.

## 행동 규칙
- 구체적인 행동 지침을 나열합니다
- 제약 조건도 명시합니다
```

### Progressive Disclosure (점진적 공개)

SKILL.md의 핵심 설계 원칙:

```
1단계: 프론트매터(name, description)만 먼저 로드 → Supervisor가 라우팅 판단
2단계: 실제 호출 시에만 전체 SKILL.md 본문을 로드 → 토큰 절약
```

| 단계 | 로드 범위 | 목적 |
|------|----------|------|
| 1단계 | name + description (2~3줄) | 라우팅 판단 |
| 2단계 | 전체 본문 (행동 지침) | SubAgent 실행 |

대규모 멀티에이전트 시스템에서 10개 이상의 Agent가 있을 때, 모든 Agent의 전체 지침을 미리 로드하면 컨텍스트가 폭발한다. Progressive Disclosure로 **필요할 때만** 로드하여 토큰 효율을 유지한다.

### 구현

#### 1단계: SKILL.md 정의

```python
EXCEL_SKILL = """\
---
name: excel-analysis-agent
description: 엑셀 데이터를 분석하고 시각화하는 Agent
allowed-tools: get_project_summary analyze_monthly_performance get_resource_allocation
---

# 엑셀 분석 Agent

## 역할
엑셀 파일을 로드하여 데이터 분석을 수행합니다.
사용자의 자연어 질문을 pandas 코드로 변환하여 실행합니다.

## 행동 규칙
- 예산, 진행률, 월별 실적 관련 질문을 처리
- 결과는 숫자와 근거를 함께 제시
- 차트가 필요한 경우 create_chart Tool 활용
"""

SQL_SKILL = """\
---
name: sql-query-agent
description: SQL 데이터베이스를 조회하여 태스크·리소스 현황을 분석하는 Agent
allowed-tools: run_sql_query get_db_schema
---

# SQL 조회 Agent

## 역할
SQLite 데이터베이스에서 프로젝트 관리 데이터를 조회합니다.

## 행동 규칙
- SQL 작성 전 반드시 get_db_schema로 스키마 확인
- SELECT 전용 — INSERT/UPDATE/DELETE 금지
- 태스크 목록, 리소스 현황, 상태별 조회를 처리
"""

RAG_SKILL = """\
---
name: rag-document-agent
description: 벡터 스토어에서 문서를 검색하여 절차·정책·가이드라인 질문에 답변하는 Agent
allowed-tools: search_documents
---

# RAG 문서검색 Agent

## 역할
ChromaDB 벡터 스토어에서 관련 문서를 검색하고 답변합니다.

## 행동 규칙
- 절차, 정책, 리스크 관리, 가이드라인 질문을 처리
- 검색 결과의 출처를 반드시 명시
- 문서에 없는 내용은 "문서에서 확인되지 않음"으로 응답
"""
```

#### 2단계: SubAgent 등록

```python
excel_subagent = {
    "name": "excel-agent",
    "description": "엑셀 데이터 분석 전문 Agent — 예산, 진행률, 월별 실적 분석",
    "system_prompt": EXCEL_SKILL,
    "tools": excel_tools,  # [get_project_summary, analyze_monthly_performance, get_resource_allocation]
}

sql_subagent = {
    "name": "sql-agent",
    "description": "SQL 데이터 조회 전문 Agent — 태스크 목록, 리소스 현황, DB 질의",
    "system_prompt": SQL_SKILL,
    "tools": sql_tools,  # [run_sql_query, get_db_schema]
}

rag_subagent = {
    "name": "rag-agent",
    "description": "문서 검색 및 분석 전문 Agent — 절차, 정책, 가이드라인 검색",
    "system_prompt": RAG_SKILL,
    "tools": rag_tools,  # [search_documents]
}

subagents = [excel_subagent, sql_subagent, rag_subagent]
```

**핵심:** `system_prompt` 필드에 SKILL.md 전체를 주입한다. SubAgent는 이 프롬프트를 기반으로 독립된 컨텍스트에서 실행된다.

#### 3단계: Supervisor 구성

```python
from deepagents import create_deep_agent

SUPERVISOR_PROMPT = """당신은 통합 어시스턴트입니다.
사용자의 질문을 분석하여 적절한 서브에이전트에게 위임하세요.

라우팅 기준:
- 엑셀 데이터 분석 (예산, 진행률, 월별 실적) → excel-agent
- DB 조회 (태스크 목록, 리소스 현황, SQL 질의) → sql-agent
- 문서 검색 (절차, 정책, 리스크 관리, 가이드라인) → rag-agent
- 복합 질문 → 여러 서브에이전트를 순차 호출하여 종합 답변 생성

서브에이전트의 결과를 종합하여 최종 답변을 한국어로 작성하세요."""

app = create_deep_agent(
    model=llm,
    system_prompt=SUPERVISOR_PROMPT,
    subagents=subagents,
)
```

#### 4단계: 실행 헬퍼

```python
from langgraph.types import Overwrite

def _normalize_messages(raw_messages):
    if isinstance(raw_messages, Overwrite):
        raw_messages = raw_messages.value
    if raw_messages is None:
        return []
    if isinstance(raw_messages, (list, tuple)):
        return list(raw_messages)
    return [raw_messages]

def run_and_print(question: str, verbose: bool = True):
    """질문을 실행하고 스트리밍으로 결과를 출력한다."""
    print(f"\n{'=' * 60}")
    print(f"질문: {question}")
    print('=' * 60)

    final = None
    for chunk in app.stream(
        {"messages": [{"role": "user", "content": question}]},
    ):
        if verbose:
            for agent_name, state in chunk.items():
                if not hasattr(state, 'get'):
                    continue
                msgs = _normalize_messages(state.get("messages"))
                for m in msgs:
                    content = getattr(m, "content", None)
                    if content:
                        role = type(m).__name__
                        print(f"  [{agent_name}/{role}] {str(content)[:200]}")
        final = chunk

    if final:
        for state in final.values():
            if not hasattr(state, 'get'):
                continue
            msgs = _normalize_messages(state.get("messages"))
            if msgs:
                content = getattr(msgs[-1], "content", msgs[-1])
                print(f"\n최종 답변:\n{content}")
    return final
```

### Supervisor 프롬프트 튜닝

라우팅 오류가 발생하면 프롬프트를 개선한다:

```python
# 기본 버전: 서술형 라우팅 기준
"엑셀 데이터 분석 (예산, 진행률, 월별 실적) → excel-agent"

# 개선 버전: 키워드 매칭 규칙
IMPROVED_PROMPT = """당신은 통합 어시스턴트입니다.

[서브에이전트 선택 규칙]
1. 질문에 '예산', '진행률', '월별 실적', '집행률' → excel-agent
2. 질문에 '태스크', '목록', '조회', 'Blocked', 'DB' → sql-agent
3. 질문에 '절차', '정책', '가이드', '리스크 관리' → rag-agent
4. 여러 도메인이 혼합된 경우 → 각 서브에이전트를 순차 호출하여 결과를 종합

서브에이전트의 결과를 종합하여 최종 답변을 한국어로 작성하세요."""
```

키워드 기반 규칙이 서술형보다 라우팅 정확도가 높다. LLM이 모호하게 판단할 여지를 줄이기 때문이다.

---

## 도구(Tool) 정의 가이드

### 엑셀 도구

```python
@tool
def get_project_summary(status: str = None) -> str:
    """프로젝트 현황을 요약합니다. status 필터 가능 (예: '진행중', '지연', '완료')."""
    df = df_projects.copy()
    if status:
        df = df[df["status"] == status]
    summary = df.groupby("status").agg(
        count=("project_id", "count"),
        total_budget=("budget_million", "sum"),
        avg_progress=("progress_pct", "mean"),
    ).reset_index()
    return summary.to_string(index=False)

@tool
def analyze_monthly_performance(month: int = None) -> str:
    """월별 실적(완료율, 이슈 건수)을 분석합니다. month 필터 가능 (1~12)."""
    df = df_monthly.copy()
    if month:
        df = df[df["month"] == month]
    result = df.groupby("month").agg(
        avg_completion=("completed_tasks", "sum"),
        total_issues=("issues_count", "sum"),
    ).reset_index()
    return result.to_string(index=False)

@tool
def get_resource_allocation(project_id: str = None, role: str = None) -> str:
    """리소스 배분 현황을 조회합니다. project_id, role 필터 가능."""
    df = df_resources.copy()
    if project_id:
        df = df[df["project_id"] == project_id]
    if role:
        df = df[df["role"] == role]
    if df.empty:
        return "조건에 맞는 리소스 배분 데이터가 없습니다."
    return df.to_string(index=False)
```

### SQL 도구

```python
@tool
def run_sql_query(query: str) -> str:
    """SQLite DB에서 SQL 쿼리를 실행하고 결과를 반환합니다."""
    try:
        df = pd.read_sql_query(query, conn)
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL 오류: {e}"

@tool
def get_db_schema() -> str:
    """DB 스키마(테이블명·컬럼명)를 반환합니다. SQL 작성 전에 먼저 호출하세요."""
    schema_info = []
    for table_name, in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall():
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        col_names = [c[1] for c in cols]
        schema_info.append(f"{table_name}: {', '.join(col_names)}")
    return "\n".join(schema_info)
```

### RAG 도구

```python
@tool
def search_documents(query: str) -> str:
    """벡터 스토어에서 프로젝트 문서를 검색합니다.
    절차·정책·가이드라인 관련 질문에 사용하세요."""
    docs = retriever.invoke(query)
    if not docs:
        return "관련 문서를 찾을 수 없습니다."
    return "\n\n".join(
        f"[출처: {d.metadata.get('source', 'N/A')}]\n{d.page_content}"
        for d in docs
    )
```

### Tool docstring 작성 원칙

| 원칙 | 좋은 예 | 나쁜 예 |
|------|--------|--------|
| **용도 명시** | "프로젝트 현황을 요약합니다" | "데이터를 처리합니다" |
| **파라미터 설명** | "status 필터 가능 (예: '진행중', '지연')" | "필터 가능" |
| **사용 시나리오** | "절차·정책·가이드라인 관련 질문에 사용" | "문서를 검색" |

docstring이 부정확하면 Supervisor가 잘못된 Agent를 선택한다. **라우팅 정확도의 50%는 docstring 품질에 의존한다.**

---

## 테스트 설계 가이드

### 라우팅 정확도 테스트

```python
test_questions = [
    ("지연 중인 프로젝트의 예산 현황을 분석해줘",        "excel_agent"),
    ("현재 Blocked 태스크 목록을 보여줘",               "sql_agent"),
    ("리스크 관리 절차를 설명해줘",                     "rag_agent"),
    ("PRJ-001의 예산, 태스크 현황, 관련 절차를 종합 분석해줘", "multi"),
]

print("=== 라우팅 정확도 테스트 ===\n")
for q, expected in test_questions:
    result = app.invoke(
        {"messages": [{"role": "user", "content": q}]},
    )
    answer = result["messages"][-1].content
    print(f"[기대: {expected}]")
    print(f"Q: {q}")
    print(f"A: {answer[:200]}...")
    print("-" * 60)
```

### 기대 라우팅 매트릭스

| 질문 키워드 | 기대 Agent | 근거 |
|-----------|-----------|------|
| 예산, 진행률, 월별 실적 | excel-agent | 엑셀 데이터 분석 |
| 태스크, Blocked, 목록 | sql-agent | DB 조회 |
| 절차, 정책, 가이드 | rag-agent | 문서 검색 |
| 예산 + 태스크 + 절차 | multi (순차 호출) | 복합 질문 |

---

## 새 Agent 추가 가이드

SKILL.md 기반 시스템에서 새 Agent를 추가하는 절차:

### 1단계: SKILL.md 작성

```python
REPORT_SKILL = """\
---
name: report-agent
description: 검색 결과와 분석 데이터를 종합하여 정형 보고서를 생성하는 Agent
allowed-tools: generate_report
---

# 보고서 생성 Agent

## 역할
다른 SubAgent의 결과를 종합하여 경영진 보고서 형식으로 변환합니다.

## 행동 규칙
- 제목, 요약, 상세 내용, 조치 사항 구조로 작성
- 수치 데이터는 표로 정리
- 리스크는 등급별로 색상 코딩 명시
"""
```

### 2단계: SubAgent dict 추가

```python
report_subagent = {
    "name": "report-agent",
    "description": "보고서 생성 전문 Agent — 검색/분석 결과를 정형 보고서로 변환",
    "system_prompt": REPORT_SKILL,
    "tools": report_tools,
}
subagents.append(report_subagent)
```

### 3단계: Supervisor 프롬프트 업데이트

```python
SUPERVISOR_PROMPT += """
- 보고서 생성 요청 → report-agent
"""
```

### 4단계: Supervisor 재컴파일

```python
app = create_deep_agent(
    model=llm,
    system_prompt=SUPERVISOR_PROMPT,
    subagents=subagents,
)
```

**기존 코드 수정 없이** SKILL.md 하나 + SubAgent dict 하나 + 프롬프트 1줄 추가로 새 Agent가 통합된다.

---

## 아키텍처 확장 방향

| 확장 | 설명 | 구현 포인트 |
|------|------|-----------|
| **Handoff 패턴** | `Command(goto=...)` 로 Agent 간 상태 전환 | LangGraph Command 객체 활용 |
| **Human-in-the-Loop** | 공정 이상 조치 시 엔지니어 승인 | `interrupt_before` / `interrupt_after` 노드 |
| **보고서 Agent** | 검색 + 분석 결과를 정형 보고서로 변환 | 위 가이드 참조 |
| **Docker 배포** | `langgraph serve`로 프로덕션 배포 | LangGraph Platform |
| **트레이싱** | LangSmith/Langfuse로 SubAgent 위임 품질 모니터링 | 콜백 핸들러 설정 |

## 관련 문서

- [Agentic RAG 구현](./agentic-rag-implementation.md) — RAG Agent 구현 (이 문서에서 재사용)
- [RAG 확장 기법](./rag-extensions.md) — HyDE, MemorySaver 확장
- [LangGraph 고급 패턴](../langgraph/langgraph-advanced.md) — Subgraph, Human-in-the-Loop
- [LangChain-LangGraph 실전 플레이북](../langchain-langgraph/rag-tool-calling-playbook.md)

## 참고 자료 (References)

- [LangGraph Supervisor 공식 문서](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangChain Agent 가이드](https://python.langchain.com/docs/how_to/#agents)
- [SKILL.md 설계 패턴](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
