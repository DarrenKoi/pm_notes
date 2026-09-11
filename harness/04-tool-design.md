---
tags: [harness-engineering, tool-design, mcp, aci]
level: intermediate
last_updated: 2026-09-11
---

# 04. 도구 설계 (Tool Design)

> 도구는 사람이 쓰는 API가 아니라 모델이 읽는 프롬프트다. 도구 이름, 설명,
> 반환값, 에러 메시지가 모두 컨텍스트에 들어간다.

## 왜 필요한가? (Why)

에이전트는 가진 도구 수준만큼만 일한다. 흔한 실패는 대부분 도구 설계에서 나온다.

- 기존 REST API를 1:1로 감싼 도구 40개를 줬더니 모델이 비슷한 도구 사이에서
  헷갈린다
- 검색 도구가 결과 5,000건을 그대로 반환해 컨텍스트가 터진다
- `user_id: 8f3a...` 같은 UUID만 돌려줘서 모델이 다음 호출에 엉뚱한 ID를 넣는다
- 에러가 `500 Internal Server Error`뿐이라 모델이 무엇을 고쳐야 할지 모른다

Anthropic은 도구 설계를 ACI(Agent-Computer Interface)라고 부르며, 사람용
UI(HCI)만큼 공을 들이라고 권한다.

## 핵심 개념 (What)

### 1. Anthropic의 도구 설계 원칙

| 원칙 | 나쁜 예 | 좋은 예 |
|---|---|---|
| 적고 영향력 큰 도구 | `list_users`, `list_events`, `create_event` 따로 | `schedule_event` (빈 시간 찾기와 생성을 한 번에) |
| 네임스페이스 | `search` | `logs_search`, `docs_search` |
| 의미 있는 반환값 | `{"id": "8f3a...", "uid": 1932}` | `{"name": "홍길동", "team": "ITC", "id": "8f3a..."}` |
| 토큰 효율 | 결과 전체 반환 | 페이지네이션, 필터, 기본 limit, 잘림 안내 |
| 출력 형식 선택 | 항상 상세 | `response_format: "concise" \| "detailed"` |
| 설명은 신입 온보딩처럼 | "Searches logs." | 언제 쓰는지, 인자 형식, 예시, 흔한 실수 |

### 2. 에러 메시지는 교정 지시문이다

OpenAI 팀은 커스텀 린터의 에러 메시지에 **고치는 방법**까지 적었다. 에러가 곧
다음 턴의 프롬프트가 되기 때문이다.

```text
나쁨: ValueError: invalid date
좋음: ERROR: since="어제" is not a valid ISO-8601 datetime.
      Use e.g. since="2026-09-10T00:00:00+09:00",
      or use since_minutes=1440 instead.
```

### 3. 부수효과(side effect) 등급을 붙인다

도구마다 되돌릴 수 있는지 표시해 두면, 권한
정책([06](./06-guardrails-and-permissions.md))과 재시도·재개
정책([07](./07-state-and-recovery.md))을 자동으로 정할 수 있다.

| 등급 | 예 | 기본 정책 |
|---|---|---|
| read-only | 파일 읽기, 로그 검색 | 자동 허용, 자유롭게 재시도 |
| reversible write | 작업 폴더에 파일 쓰기, draft 생성 | 샌드박스 안에서 허용 |
| irreversible / external | 메일 발송, DB 삭제, 결재 요청 | 사람 승인 + idempotency key |

### 4. 범용 도구 vs 전용 도구

- **범용 (bash, 코드 실행)**: 표현력이 가장 크다. 모델이 이미 bash와 Python을 잘
  안다. 대신 샌드박스가 필수이고 감사(audit)가 어렵다.
- **전용 (`logs_search` 등)**: 안전하고 관측하기 쉽다. 대신 새 요구가 생길
  때마다 도구를 추가해야 한다.

실무에서는 탐색과 가공은 범용 도구(샌드박스 안)로, 외부 시스템 쓰기는 전용
도구(권한 통제)로 나누는 조합이 흔하다.

### 5. MCP와 스킬

- **MCP(Model Context Protocol)**: 도구를 하네스와 분리된 서버로 표준화한다.
  여러 에이전트가 같은 도구를 공유할 수 있다. 다만 서버 하나를 붙이는 순간 도구
  정의가 모두 컨텍스트에 들어가므로, 도구 수를 관리해야 한다.
- **스킬(Skill)**: 이름과 한 줄 설명만 먼저 보여주고, 필요할 때 전체 지침을 읽게
  한다(점진적 공개). 도구 정의보다 컨텍스트 비용이 싸다.

### 6. 오픈소스 모델에서 특히 챙길 것

- JSON Schema로 인자를 검증하고, 실패하면 예외 대신 교정 메시지를 돌려준다
  ([02](./02-agent-loop.md)의 `execute`)
- 서빙 엔진이 지원하면 structured output이나 guided decoding으로 형식 오류
  자체를 줄인다
- 인자는 적고 평평하게 설계한다. 중첩 객체와 선택 인자가 많을수록 형식 오류가
  늘어난다

## 어떻게 사용하는가? (How)

### Step 1. 도구 명세 예시

```python
LOGS_SEARCH_SPEC = {"type": "function", "function": {
    "name": "logs_search",
    "description": (
        "Search application logs for errors or events of a service.\n"
        "- Start narrow: set service and since_minutes. Widen if empty.\n"
        "- Newest first, max `limit` per page. Use `cursor` to page.\n"
        "- response_format='concise' returns one line per hit (default).\n"
        "  Use 'detailed' only for the few hits you need to inspect.\n"
        "Example: logs_search(query='timeout', service='recipe-api',\n"
        "                     since_minutes=120)"
    ),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string",
                  "description": "Keywords, e.g. 'timeout' or 'status=500'"},
        "service": {"type": "string",
                    "description": "Service name, e.g. 'recipe-api'"},
        "since_minutes": {"type": "integer", "default": 60},
        "limit": {"type": "integer", "default": 20, "maximum": 100},
        "cursor": {"type": "integer", "default": 0},
        "response_format": {"type": "string", "default": "concise",
                            "enum": ["concise", "detailed"]},
    }, "required": ["query"]},
}}
```

### Step 2. 페이지네이션과 안내문이 붙은 반환값

```python
def format_page(rows: list, cursor: int, limit: int, render) -> str:
    """결과 한 페이지를 문자열로 만들고, 다음 행동 안내를 붙인다."""
    if not rows:
        return "No results. Try broader keywords or a larger since_minutes."
    page = rows[cursor:cursor + limit]
    lines = [render(r) for r in page]
    nxt = cursor + limit
    if nxt < len(rows):
        lines.append(f"-- {len(rows) - nxt} more results. "
                     f"Call again with cursor={nxt}, or narrow the query.")
    return "\n".join(lines)


rows = [{"ts": f"10:{i:02d}", "level": "ERROR", "msg": f"timeout #{i}"}
        for i in range(45)]
render = lambda r: f"{r['ts']} {r['level']} {r['msg']}"
print(format_page(rows, cursor=0, limit=20, render=render))
```

### Step 3. 에이전트로 도구를 평가한다

Anthropic이 권하는 순서다.

1. 프로토타입 도구를 만든다
2. 실제 업무에 가까운 태스크 여러 개로 eval을 돌린다
   ([05](./05-verification-and-evals.md))
3. 트랜스크립트를 읽는다. 모델이 어느 도구에서 헷갈렸는지, 어떤 인자를 틀렸는지,
   호출 수가 왜 많았는지 본다
4. 도구 설명과 반환값을 고치고 다시 eval을 돌린다. 트랜스크립트를 에이전트에게
   주고 개선안을 받는 것도 효과적이다

## 학습 체크리스트

- [ ] 내 에이전트의 도구 목록을 뽑아 이름이 겹치거나 헷갈리는 쌍을 찾았다
- [ ] 반환값이 가장 큰 도구에 limit, 페이지네이션, 잘림 안내를 붙였다
- [ ] 에러 메시지 5개를 "무엇이 틀렸고 어떻게 다시 호출하는지" 형식으로 바꿨다
- [ ] 모든 도구에 read-only / reversible / irreversible 등급을 붙였다
- [ ] 트랜스크립트를 보고 도구 설명을 고친 뒤 eval 점수 변화를 확인했다

## 참고 자료 (References)

- [Writing effective tools for agents — with
  agents](https://www.anthropic.com/engineering/writing-tools-for-agents) —
  Anthropic
- [Building effective agents — Appendix: Prompt engineering your
  tools](https://www.anthropic.com/engineering/building-effective-agents) —
  Anthropic
- [12-Factor Agents — Tools are just structured
  outputs](https://github.com/humanlayer/12-factor-agents) — HumanLayer
- [Harness engineering](https://openai.com/index/harness-engineering/) — OpenAI,
  교정 지시가 담긴 린터 에러

## 관련 문서

- [02. 에이전트 루프](./02-agent-loop.md)
- [06. 가드레일과 권한](./06-guardrails-and-permissions.md)
- [05. 검증과 평가](./05-verification-and-evals.md)
