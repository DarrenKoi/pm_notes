---
type: review-log
tags: [mcp, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
---

# MCP 문서 정리 기록

## 범위와 분류

원래 Markdown 4개를 모두 읽고 같은 경로에 유지했다. 원래 작성일은 보존했다. 파일 이동·삭제·다른 주제 내용 통합은 없다. 이 묶음에는 실행 파일·첨부가 없다. 새 [버전·실행 조건](./version-and-execution-notes.md)을 SDK/프로토콜 경계의 대표 설명으로 두고 기존 문서에서는 해당 설명으로 연결했다.

프로토콜 기초, LangGraph 통합, LLM 운영 하네스는 용도가 달라 유지한다. README가 세 문서를 모두 안내하도록 보완했다. 기존 다른 ai-dt 주제 링크는 원래 참조로 유지했으며 신규 형제 주제 링크는 추가하지 않았다.

| 원래 문서 | 개별 검토와 변경 |
|---|---|
| [README](./README.md) | 누락된 하네스 문서와 버전 안내를 목차에 추가. 읽기 순서와 역할 구분, 검토 metadata 추가 |
| [MCP 기초](./mcp-basics.md) | 모델 생성/host 실행 책임 구분, 플러그 앤 플레이의 지원·권한 조건, 구현 SDK가 필수라는 오해 수정. 기존 FastMCP 예제를 v1.26.0으로 한정. FastAPI lifespan/session manager와 mount 경로 보완, Desktop `uv --directory` 명령으로 cwd 가정 제거. Inspector 고정 port 가정 제거. 고정 날씨를 실제 조회로 설명하지 않음 |
| [LangGraph 연동](./mcp-langgraph-integration.md) | 독립 adapter 유지보수 중단과 새 통합 경로 명시. sync client 생성/await get_tools 수정, create_agent 사용, 현재 Python·서버 절대 경로 처리. HTTP 예제의 SSE 제목 오류 수정. tool 목록 snapshot·세션 수명·가상 응답·고정 날씨 조건 보완 |
| [LLM 하네스](./harness-engineering-llm.md) | 설계 해석과 공식 요구 구분, 작은 앱에서 세 서비스가 필수라는 오해 수정. Responses 예제가 첫 요청뿐임을 명시. call_id/도구 결과 반환과 strict schema의 범위 설명. 평가 지표는 과제별 분모와 성공 기준을 정해 선택 |

## 기술 근거 · 확인 2026-10-04

- [MCP 2026-07-28 사양](https://modelcontextprotocol.io/specification/2026-07-28), [변경 기록](https://modelcontextprotocol.io/specification/2026-07-28/changelog), [이전 2025-11-25 transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports): 신규 stateless/요청별 협상과 이전 세션 기반 SDK의 맥락을 구분했다. 사양의 공개 날짜가 모든 SDK·host의 지원을 증명하지 않는다.
- [Python SDK v1.26.0](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/README.md), [v2 migration](https://py.sdk.modelcontextprotocol.io/migration/), [현재 SDK README](https://github.com/modelcontextprotocol/python-sdk): FastMCP와 MCPServer import/API, lifespan·mount 경로를 대조했다. v1 예제를 v2까지 실행 가능하다고 주장하지 않는다.
- [독립 adapter README](https://github.com/langchain-ai/langchain-mcp-adapters), [main pyproject 0.3.2](https://github.com/langchain-ai/langchain-mcp-adapters/blob/main/pyproject.toml), [LangChain migration](https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters): 독립 adapter의 await·세션 방식과 새 MCPAdapter/list_tools/context 방식이 다름을 확인했다. main source 버전과 PyPI 배포 일치는 미확인이다.
- [MCP host 연결 안내](https://modelcontextprotocol.io/docs/develop/connect-local-servers): 서버 command/args 등록을 대조했다. Desktop의 cwd 확장 지원을 가정하지 않는다. 실제 Desktop 연결은 하지 않았다.
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [평가 원칙](https://developers.openai.com/api/docs/guides/evaluation-best-practices), [trace grading](https://developers.openai.com/api/docs/guides/trace-grading), [evals](https://developers.openai.com/api/docs/guides/evals): 함수 요청·도구 결과·평가 설계를 대조했다. web/file search와 MCP tool의 공식 안내도 열람했지만 실제 hosted tool/API는 실행하지 않았다.
- 기존 하네스 자료의 OpenTelemetry GenAI, NIST AI 600-1(2024), LangSmith 평가, Promptfoo CI 페이지를 열람했다. 이를 이 예제의 실행 결과나 모든 하네스 구성 요소의 필수성 증거로 사용하지 않는다. Building agents 기존 링크는 열람 실패로 표시했다. Codex 활용 PDF/o3-mini system card의 세부 주장은 재검증하지 않은 과거 참고다.

## Claude 협의

`HERDR_ENV=1`에서 현재 pane 조회가 `pane_not_found`를 반환했다. 전용 Claude pane을 찾지 못해 협의하지 못했다. 관련 없는 다른 pane은 제어하지 않았다. 공개 문서와 일치하는 API 수정은 진행했고, 서로 목적이 다른 문서의 전체 합병과 새 통합으로 모든 예제를 대체하는 판단은 보류했다.

## 세 차례 검증

### 1. 목록과 고유 내용

원래 snapshot의 4개 경로를 대조했다. 입문 도식, 도구/resource/prompt 예제, 산술·고정 날씨 서버, 단일·다중·원격 agent, 가상 응답, 하네스 checklist와 첫 요청 예제를 유지했다. 오류가 있는 API·경로·설명은 위 표에 기록하고 수정했다. 삭제나 폴더 이동은 없다. 새 안내와 이 기록을 포함한 현재 문서는 6개다.

### 2. 기술과 로컬 예제

- Python 3.14 로컬 AST 검사: Python fence 10개 통과. bash fence 3개 구문, JSON 1개 파싱 통과.
- 문서의 실제 산술/날씨 함수를 추출 실행: `(3+5)*2=16`, `3+(-5)=-2`, 없는 도시의 “데이터 없음” 분기 통과.
- 수정한 3개 agent 흐름을 **가짜 MCP client/model**로 실행해 동기 생성과 await 도구 조회를 확인했다. 첫 fixture는 가짜 tool을 문자열로 만들어 `.name` 접근에서 실패했다. fixture를 name을 가진 객체로 수정한 뒤 통과했다. 실제 SDK 통신 검증이 아니다.
- 실제 설치된 `openai==2.30.0`을 로컬 httpx MockTransport로 사용해 첫 요청의 model/tool/strict/required/additionalProperties 직렬화를 확인했다. 외부 API 호출 0이며 실제 도구 실행도 0이다.
- 로컬 `langchain==1.2.15`, `langgraph==1.1.6`에서 `create_agent` import/signature 확인. 로컬 `langchain.mcp`는 없으며 mcp/adapters/FastAPI/uvicorn도 설치되어 있지 않다. 새 예제를 설치하여 실행하지 않았다. Python 3.14에서 LangChain의 Pydantic v1 호환 경고가 발생했으므로 전체 dependency 실행 호환을 증명하지 않는다.

### 3. 링크·메타데이터·읽기 화면

상대 링크·anchor와 YAML/Obsidian properties·탐색을 검사하고 아래 최종 확인에 결과를 남긴다. 기존 다른 주제 참조는 신규 참조와 구분한다.

## 남은 미확인

실제 SDK v1/v2 통신, HTTP lifespan/인증/Inspector/Claude Desktop 연결, 새 LangChain 통합 설치·API·provider 모델 접근, 전체 dependency lock, 전용 Claude 의견은 미완료다. 공개 source main은 이후 바뀔 수 있다. 실제 통합 검증을 완료한 듯 표시하지 않는다.

## 최종 확인

- 원래 4개 파일을 유지했다. 제목/절의 변경은 SSE → Streamable HTTP 정정, OpenAI 요구처럼 보인 제목 → 설계 해석, 필수 지표 → 선택 지표다. 나머지 고유 도식·서버 함수·agent 사용 사례·응답 예시는 보존했다.
- 현재 6개 YAML의 중복 key 없음, 검토일 확인. 상대 링크·anchor 검사: 신규 오류 0. 기존 다른 주제 참조 6개는 원래 링크로 유지한다. 이 묶음에는 첨부·wiki/reference-style 링크가 없다. `git diff --check` 통과.
- 정확한 pm_notes vault에서 Obsidian 1.13.7 CLI properties 6개를 확인했다. 읽기 화면에서 README → 버전·실행 조건 링크 클릭이 `ai-dt/mcp` 경로로 이동했고 한국어 본문·aliases·metadata·버전 표·Python fence가 읽기 콘텐츠로 노출됨을 확인했다. 모든 아래쪽 코드를 개별 시각 검사한 것은 아니다.
- 실제 MCP/HTTP/provider 실행과 Claude 협의가 남아 상태를 `partial`로 유지한다.
