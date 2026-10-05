---
tags: [knowledge-base, index]
aliases: [지식 저장소, pm_notes]
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: vault_index
---

# 한국어 지식 저장소

> 개념의 목적·작동 방식·사용법·적용 조건을 다시 읽는 학습 자료와, 작성 당시의 업무·실험 기록을 구분해 보관한다.

## 시작하는 방법

아래에서 주제를 고른 뒤 **그 폴더의 `README.md`**를 연다. 각 목차의 읽기 순서를 따라 개념 → 예제 → 적용 조건 → 검토 기록으로 읽는다. 최상위 폴더는 독립 주제이며 서로의 문서를 합치거나 새 크로스 링크로 연결하지 않는다. 루트는 전역 안내만 제공한다.

## 대·중·소분류로 탐색하기

각 주제의 `README.md`에 있는 **주제별 분류 목차**를 열면 대분류 → 중분류 → 소분류 순서로 문서를 찾을 수 있다. 예를 들어 `AI·DT → RAG → OpenSearch 검색`, `웹 개발 → 품질 검증 → 소프트웨어 테스트`처럼 읽는다. 최상위 폴더와 기존 파일 경로는 유지하고, 폴더별 `taxonomy-index.md`와 문서 속성으로 재분류했다.

| 속성 | 의미 |
|---|---|
| `category_major` | 넓은 분야인 대분류 |
| `category_middle` | 기술·업무 영역인 중분류 |
| `category_minor` | 구체적인 학습·작업 주제인 소분류 |
| `note_kind` | 학습·실습·템플릿·업무 기록·원문 자료 등 문서 용도 |
| `classified_on` | 분류를 적용한 날짜; 기술 검증일과 구분 |

새 문서는 같은 주제 안의 분류를 선택해 속성을 채우고 그 폴더의 분류 목차에 링크를 추가한다. 불변 원문과 과거 작업 원문은 속성을 덧붙이지 않고 목차에서만 분류한다. 로컬 전용 사내 보고와 `.remember/` 작업 메모는 공개 탐색 목차에서 제외한다. 분류 검색 방법과 보호 범위는 각 분류 목차, 실제 검증 결과는 각 `classification-review.md`에서 확인한다.

분류 속성은 Obsidian의 기본 속성 기능으로 읽으며 별도 플러그인을 설치하지 않는다. 기존 Markdown 상대 링크도 지원되는 형식이다. [속성 공식 안내](https://help.obsidian.md/properties), [내부 링크 공식 안내](https://help.obsidian.md/links).

## 독립 주제 입구

| 독립 주제 폴더 | 내용과 읽는 목적 |
|---|---|
| [ai-dt/](./ai-dt/README.md) | 모델·RAG·MCP·데이터 처리·평가/운영 학습. 하위 주제의 목차에서 해당 분야만 선택 |
| [web-development/](./web-development/README.md) | Python/TypeScript·웹 프레임워크·테스트 학습과 예제 앱 |
| [Codes/](./Codes/README.md) | 실행 가능한 Python 예제의 사용법·환경·검증 한계 |
| [dev-environment/](./dev-environment/README.md) | CLI·터미널·개발 도구 설정과 조건 |
| [harness/](./harness/README.md) | 코딩 에이전트 실행/평가·하니스 설계와 보존한 모델 자료 |
| [orchestration/](./orchestration/README.md) | 에이전트 협업 설계와 운영 계약 |
| [platform/](./platform/README.md) | 플랫폼 연구·기획·티켓/검토 기록 |
| [my-task/](./my-task/README.md) | 업무 보고·AIX 방법론 출처·재사용 양식·적용 기획/발표 |
| [RAG/](./RAG/README.md) | 특정 RAG 사례/벤치마크를 읽는 단발 분석 |
| [docs/](./docs/README.md) | 에이전트 운영·이슈/도메인 문서 작성 규칙 |
| [_workspace/](./_workspace/README.md) | 당시 세션의 임시·보존 기록. 현재 기술 설명의 대표로 사용하지 않음 |

기존 탐색 입구: [AI/DT 목차](./ai-dt/README.md) · [웹 개발 목차](./web-development/README.md) · [개발 환경 목차](./dev-environment/README.md) · [실행 예제 목차](./Codes/README.md).

## 검증 상태를 읽는 방법

`reviewed_on`은 검토일이고 `last_updated`는 원래 작성/갱신일이다. 검토일이 있다고 모든 예제·외부 시스템·회사 실적이 검증된 것은 아니다. `review_status: reviewed_with_limits`와 폴더의 `organization-log.md`에서 실제 확인 범위·출처·버전/조건·미확인을 함께 읽는다. 당시 업무 기록의 완료/계획/승인을 오늘의 사실로 고쳐 쓰지 않았다.

원문 416개와 운영 문서 42개의 검토 결과를 대조했다(최종 대조 2026-10-05). 2026-10-04 기술 검토에서 변동 가능한 기술 설명은 확인 가능한 공식 근거와 대조하고,확인되지 않은 회사 환경·배포·성능은 미확인으로 남겼다. Claude 협의는 Herdr pane 연결 불가로 확보하지 못했고,협의가 필요한 통합/분류 결정은 보류했다. Obsidian CLI는 정확한 `pm_notes` 경로에 연결됐지만 읽기 화면 검증은 일부 실패해 미완료다. [전역 정리 기록](./organization-log.md)에서 완료 범위와 보류를 확인한다.

일반 Markdown 상대 링크·기본 frontmatter/tags/aliases/callout을 사용한다. 추가 플러그인 설치는 필요하지 않다. 코드·첨부·로컬 전용 정보와 기존 사용자 변경은 문서 정리 범위에서 보호한다.

## 기존 주제별 바로 읽기

아래 날짜는 기존 목차에 기재된 작성/갱신 기록이며 기술의 최신성이나 2026-10-04 현재 검증일을 뜻하지 않는다. 현재 읽기 순서와 조건은 각 주제 폴더 목차가 대표한다.

| 분야 | 주제 | 기존 목차의 갱신일 |
|------|------|----------------|
| AI/DT | [RAG/LangGraph](./ai-dt/rag/langgraph/README.md) | 2026-01-31 |
| AI/DT | [RAG/Milvus](./ai-dt/rag/milvus/README.md) | 2026-01-31 |
| AI/DT | [MCP](./ai-dt/mcp/README.md) | 2026-01-31 |
| AI/DT | [데이터 처리 (Airflow + MinIO)](./ai-dt/data-handling/airflow-minio-tutorial.md) | 2026-05-02 |
| AI/DT | [데이터 정규화 시리즈](./ai-dt/data-handling/normalization/README.md) | 2026-05-02 |
| Web | [TypeScript 프로젝트 설정](./web-development/typescript/tsconfig-setup.md) | 2026-02-01 |
| Web | [Vite 기초](./web-development/typescript/vite-basics.md) | 2026-02-01 |
| Web | [TypeScript/Vue](./web-development/typescript/vue/README.md) | 2026-02-01 |
| Codes | [조직 계층 트리 유틸리티 안내](./Codes/README.md#문서가%20없는%20실행%20모듈) | 2026-02-02 |
| Codes | [대화 메모리 → OpenSearch](./Codes/python/history-opensearch/README.md) | 2026-02-08 |
| AI/DT | [RAG/OpenSearch](./ai-dt/rag/opensearch/README.md) | 2026-02-08 |
| Dev Environment | [터미널 필수 명령어](./dev-environment/terminal/README.md) | 2026-02-09 |
| Web | [Unit Testing 기초](./web-development/testing/unit-testing-basics.md) | 2026-02-19 |
| Web | [E2E Testing 기초](./web-development/testing/e2e-testing-basics.md) | 2026-02-19 |
| Web | [테스트 프레임워크 비교](./web-development/testing/testing-frameworks.md) | 2026-02-19 |
