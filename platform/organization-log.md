---
tags: [platform, organization, verification]
aliases: [플랫폼 문서 정리 기록]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
---

# 플랫폼 문서 정리 기록

기준일: **2026-10-04**. 이 기록은 단계별 검토 결과와 남은 미확인을 구분한다. Markdown 원본 42개와 HTML 2개를 목록화했다. 다른 주제 폴더로 이동하거나 통합하지 않는다. 실행 코드·첨부·HTML은 보존한다. 커밋·푸시는 하지 않는다.

## 이번 수정과 판단

- README: 설계·현재 검토·조사·구현 티켓의 역할과 읽기 순서를 추가했다. “확정 전제”는 이 저장소의 설계에서 채택한 전제이며 사내 실측 완료를 뜻하지 않도록 표현했다.
- 01~05: 목적이 달라 통합하지 않았다. 요구·기술 구조·운영 책임·산정·작성 규약을 각각 유지하고 현재 적용 조건을 연결했다. 기존 코드 블록과 고유 예제를 보존했다.
- 04: 헬스 실패 시 자동 tier 강등이라는 설명을 02·03 및 상태 전이 티켓에 맞춰 `degraded` 자동 전환·등급 유지로 수정했다. public sub만으로 issuer·사번 매핑 위험이 제거된다는 표현을 조건부로 수정했다.
- 06: EE/메뉴 유무로 티어·비활성을 단정하는 규칙을 제거했다. ORAS 저장·조회 성공과 서명·삭제 방지 검증을 분리하고 버전·media type 조건을 남겼다. PAT 명령 인자를 대화형 로그인으로 바꾸고 Protocol 예제에 import를 추가했다. 사내 업로드·계정 변경은 실행하지 않았다.
- CONVENTIONS 및 PLAT-L2-002: Motor의 deprecated 일정과 신규 PyMongo Async 검토 필요를 표시했다. 승인된 기존 의존성 선택 자체는 변경하지 않았다.
- PLAT-L0-001: pytest 마커 제외는 수집 뒤 선택 해제라는 공식 동작에 맞게 AC-3와 검증 표를 수정했다. grep 출력 개수와 종료 코드를 혼동하지 않도록 했다.
- 신규 `review-notes.md`: 확인일·버전·적용 조건·출처와 보류를 한곳에 모았다. 기존 역사 기록을 현재 사실로 고쳐 쓰지 않았다.

근거: [현재 검토](./review-notes.md)에 공식 GitLab, ORAS 1.3, OpenID Connect Core 1.0, RFC 9068, Motor 3.7.1, pytest 문서를 항목별 연결했다. 사내 설치 버전은 미확인이다.

## Claude 협의

`HERDR_ENV=1`을 확인했다. 작업 환경에서 `herdr pane current --current`가 `pane_not_found`로 실패했다. 작업 전용 Claude pane을 확인하지 못했으며 다른 프로젝트 pane을 제어하지 않았다. Claude 의견을 받았다고 주장하지 않는다.

Motor 의존성 변경, 조직 개편 겹침 임계값과 분모, 기록 통합·삭제 등 협의가 필요한 판단은 보류했다. 자신의 권고안과 모순 사례는 [현재 검토의 보류 · Claude 협의가 필요한 보류 절](./review-notes.md)에 남겼다. 공식 동작·기존 설계 계약으로 명확하게 정정할 수 있는 작업을 진행했다.

## 첫 단계의 문서별 검토 상태

| 대상 | 결과 |
|---|---|
| README.md | 목차·자료 역할 정리, 설계 전제와 실측 구분 |
| 01-requirements.md | 본문 검토, 요구·SLO는 제안/설계 조건, 사내 구현 미확인 |
| 02-architecture.md | 본문 검토, 신원·불변성·표준 호환성 추가 검증 중 |
| 03-operations.md | 본문 검토, RACI·운영 수치 제안 상태 유지 |
| 04-plan.md | 본문 검토, 자동 강등·public sub 표현 수정, 일정·인력은 추정 |
| 05-ticket-conventions.md | 본문 검토, 작성 양식과 설계 문서 역할 유지 |
| 06-gitlab-check.md | 공식 근거로 판정·예제 수정, 사내 실측 미실행 |
| tickets/CONVENTIONS.md | 본문 검토, Motor 선택 변경은 협의 대기 |
| tickets/PLAT-L0-001.md | 본문 검토, 마커 실행 대상과 수집 구분 수정 |
| tickets/PLAT-L2-002.md | 본문 검토, Motor 일정 안내, 실제 구현 미확인 |
| tickets/PLAT-L2-007.md | 본문 검토, 50% 분산/60% 임계값 모순 협의 대기 |
| 나머지 원본 Markdown 31개 | 목록화 및 일부 읽기, 개별 최종 검토·근거 대조 미완료 |
| HTML 2개 | 원본 보존, 개별 내용·참조 확인 미완료 |

## 첫 단계의 세 단계 검증

1. 원본 42개 존재와 수정 대상 외 본문 동일성, 코드 블록 보존, HTML 등 비 Markdown 해시를 대조한다.
2. 공식 근거와 수정 문구를 대조한다. Protocol import·실행, 임시 pytest 마커 선택/수집 fixture, 티켓 30개 선행 ID 존재·순환 검사를 수행한다. 별도 구현 앱과 사내 서버는 실행하지 않는다.
3. 이번 수정 대상의 링크·앵커·frontmatter와 Obsidian CLI 속성·읽기 탐색을 검사한다. 폴더 전체 검증은 나머지 대상 검토 후 다시 수행한다.

실제 실행 결과는 아래에 추가한다. 현재 이 폴더와 저장소 전체가 완료되었다는 뜻은 아니다.


## 이번 단계의 실제 검증 결과

- **1차 목록·내용 대조:** 원본 Markdown 42개 모두 존재한다. 이번 검토 원본 11개 외 31개 본문은 단계 시작 snapshot과 동일하다. 06에서 명시적으로 수정한 로그인·Protocol 외 검토 대상 코드 블록은 동일하다. HTML 2개는 작업 시작 SHA-256과 일치한다. 삭제·이동·고유 예제 손실 없음.
- **2차 기술·로컬:** Python Protocol 예제는 AST와 실행을 통과했다. pytest **9.0.2** 임시 fixture에서 `1/2 tests collected (1 deselected)`와 `1 passed, 1 deselected`를 확인했다. 제외된 integration 모듈의 import 흔적도 확인해 수집/실행 차이를 입증했다. 티켓 30개의 선행 ID는 모두 존재하며 의존성 순환은 없다. 이것은 실제 구현 테스트 통과를 뜻하지 않는다.
- **3차 참조·메타데이터·앱:** 이번 대상 원본 11개+신규 2개=13개의 상대 링크·앵커·중복 frontmatter 키 검사에서 새 문제 0개, 기존 문제 0개다. 정확한 `pm_notes` vault에서 Obsidian CLI 속성 13개를 읽고 검토일을 확인했다. Obsidian **1.13.7** 읽기 화면에서 `platform/README` → 현재 검토 링크 → `platform/review-notes` 경로를 확인했다. 한국어·tags·alias·검토일·callout 렌더링을 screenshot으로 확인했다. 모든 문서의 모든 화면·첨부 렌더링을 검증한 것은 아니다.
- **남은 범위:** 나머지 원본 31개의 개별 최종 검토·공식 근거 대조, HTML 참조, 폴더 전체 최종 검증이 남아 있다. 이번 단계 결과만으로 platform 완료를 선언하지 않는다.


## 추가 정리와 판단

- 원본 42개 모두 본문·예제·완료 보고를 검토했다. 조사 3개는 업무 설계 과정의 역사 기록으로 분류하여 본문·인용을 보존하고 현행 적용 조건을 상위 검토에 모았다. 유사 주제를 삭제·통합하는 결정은 Claude 협의 부재로 보류했다.
- research/README를 추가해 조사 질문별 읽기 순서를 제공했다. 기존 폴더·파일명을 유지하여 참조 이동을 만들지 않았다.
- 02의 DB 필수 일반화·캐시 무효화 불필요 단정·COMPLIANCE 회수 설명을 공식 조건에 맞췄다. 도입 선택과 D# 자체는 바꾸지 않았다.
- tickets/README의 wave 3에는 L2-004 선행과 소비자가 동시에 있었다. 명시된 선행 계약으로 wave를 8개로 재계산했다. 동시 파일 편집이나 구현 작업은 실행하지 않았다.
- PLAT-L6B-001의 L2-004 소유 경로가 catalog/ownership.py로 잘못 적혀 있었다. 해당 티켓의 auth/tuples.py·auth/decide.py 계약과 맞췄다.
- HTML 2개는 원본 해시를 보존했다. HTMLParser로 외부/상대 href·src가 없음을 확인했으며 bento의 내장 data 자료는 출력·변경하지 않았다. HTML은 현재 문서 수정이 반영된 새 결과물이 아니다.
- 추가 Herdr 읽기 확인도 pane_not_found였다. 불확실한 엔진·의존성·인가/계측 계약 변경에 Claude 의견은 없으며 현재 검토에 자신의 권고와 보류 이유를 기록했다.

## 전체 원본의 개별 검토 결과

아래 결과는 문서 검토다. 제품 전체 변동 주장 재검증·별도 코드 실행 성공을 뜻하지 않는다.

| 원래 문서 | 검토 결과와 남은 적용 조건 |
|---|---|
| 01-requirements.md | 요구·SLO 제안과 구현 사실 구분; 고유 요구 보존 |
| 02-architecture.md | 운영 DB 일반화·캐시 회수·COMPLIANCE 회수 설명 정정; D# 선택 보존, 호환 계약 미확인 |
| 03-operations.md | RACI·등급 유지·승격·인시던트·조직 절차 검토; 수치는 제안이며 운영 실측 미확인 |
| 04-plan.md | 자동 tier 강등 모순·public sub 위험 제거 단정 수정; 인력/기간 추정 유지 |
| 05-ticket-conventions.md | 작성 양식·공유 계약 규약 보존; 실제 코딩 작업 지시로 실행하지 않음 |
| 06-gitlab-check.md | 기능 판정·자격증명 예제·Protocol import 수정; 사내 업로드 미실행 |
| README.md | 자료 역할·읽기 순서·조사 목차·보존 HTML 목차 추가 |
| research/agent-skill-registry-research.md | 본문·예제 검토; 09-03 기록 보존. Object Lock·MCP 버전·OTel·폐쇄망 서명 일반화는 현재 검토로 정정. 나머지 변동 기능 미확인 |
| research/authorization-at-scale-google.md | 본문·예제 검토; 09-03 기록 보존. RBAC 객체 권한·외부 조직 코드/내부 ID 구분·일관성·전파 최대값·memory 모드 정정 안내. quota·버전별 운영은 미확인 |
| research/gitlab-as-distribution-backend.md | 본문·예제 검토; 09-03 기록 보존. tier/버전 조건과 서명·불변성 구분 안내. 나머지 기능 표의 현행 가용성과 사내 topology 미확인 |
| tickets/CONVENTIONS.md | 별도 코드 저장소 전제 유지; Motor deprecated 안내·의존성 변경 협의 대기 |
| tickets/PLAT-L0-001.md | pytest 수집/선택 해제 AC와 명령 정정; 임시 fixture 검증 |
| tickets/PLAT-L1-001.md | 키·식별자·Protocol 3개 계약 보존; 원자적 발행·버전 참조 계약 협의 대기 |
| tickets/PLAT-L1-002.md | Object Lock 보호 버전과 key 재발행 거부 구분; 동시성·실제 MinIO 미검증 |
| tickets/PLAT-L2-001.md | 자산 3종·정확 버전·상태 schema 계약 보존; 표준 확장 호환성 미검증 |
| tickets/PLAT-L2-002.md | 저장·삭제 증분 계약 보존; Motor 신규 선택 재검토 표시 |
| tickets/PLAT-L2-003.md | 등급 유지 전이·불변 필드 검토; 모든 버전 삭제 시 isLatest 의미 협의 대기 |
| tickets/PLAT-L2-004.md | 파생·만료·승계·fail-closed 계약 검토; 조회 2회 범위와 팀 lookup 협의 대기 |
| tickets/PLAT-L2-005.md | ID·표시명·외부 코드 구분 검토; 1,000회 fixture는 전역 충돌 불가 증명 아님 |
| tickets/PLAT-L2-006.md | 실패 snapshot·부분 응답 보호 계약 검토; 코드 변경 시 팀 동일성 협의 대기 |
| tickets/PLAT-L2-007.md | 50% 분산/60% 임계값 모순 확인; 겹침 분모·분할 우선순위 협의 대기 |
| tickets/PLAT-L3-001.md | 페이지 1~100·필터·계약 경계 검토; 과거 테스트 보고 보존 |
| tickets/PLAT-L3-002.md | upsert/delete·ack·재시도·멱등 계약 검토; 실제 큐/색인 미검증 |
| tickets/PLAT-L3-003.md | 권한 제외·count·degraded 정렬 계약 검토; 선행 인가 이후 wave로 이동 |
| tickets/PLAT-L3-004.md | 배치·checkpoint·실패 시 alias 유지 계약 검토; 실제 동시 카탈로그 변경 미검증 |
| tickets/PLAT-L5-001.md | issuer/aud·서명·로그 비노출 검토; typ·nbf·sub 사번 매핑 협의 대기 |
| tickets/PLAT-L5-002.md | 401/403·scope·판정 오류 기본 거부·감사 필드 검토; 실제 앱 미검증 |
| tickets/PLAT-L6-001.md | 읽기 전용 투영·지원 런타임 필터 계약 검토; 과거 보고 보존 |
| tickets/PLAT-L6-002.md | marketplace·정확 버전·digest·rename 투영 계약 검토; 클라이언트 실설치 미검증 |
| tickets/PLAT-L6-003.md | 설치/원격 투영·권한·원본 ID 보존 검토; 표준 응답 실호환 미검증 |
| tickets/PLAT-L6B-001.md | Broker 모델·SLO 제안 검토; L2-004 소유 경로를 auth/tuples.py·decide.py로 정정 |
| tickets/PLAT-L6B-002.md | 5분 due·timeout 실패관측·claim 중복 방지 검토; 실제 스케줄러 미검증 |
| tickets/PLAT-L6B-003.md | 6회 장애·tier 유지·성공 복구 검토; 과거 실행 보고 보존 |
| tickets/PLAT-L6B-004.md | 권한·상태·만료·회수·rate limit 경계 검토; 실제 토큰 발급 미실행 |
| tickets/PLAT-L6B-005.md | 호출 결과별 계량·민감 본문 배제 검토; 실제 endpoint/OTel 미검증 |
| tickets/PLAT-L7-001.md | ack/nack·시도 상한·윈도우 반환값 계약 검토; 과거 보고 보존 |
| tickets/PLAT-L7-002.md | claim·ack·nack·실패 큐 계약 검토; 실제 Redis 미검증 |
| tickets/PLAT-L7-003.md | INCR/EXPIRE Lua·한도·TTL 계약 검토; 실제 Redis 원자성 미검증 |
| tickets/PLAT-L8-001.md | 관측 가능 범위·표준/사내 속성 검토; 모든 이벤트 gen_ai.agent.id 매핑 협의 대기 |
| tickets/PLAT-L8-002.md | 성공/오류 span·metric·민감 값 배제 검토; 실제 SDK/Collector 미검증 |
| tickets/PLAT-L8-003.md | unavailable 유지·출처/기간·완전성 검토; 누락률 분모 협의 대기 |
| tickets/README.md | 선행 ID로 wave 0~7 재계산; same-wave 무충돌 단정 제거, 전체 티켓 링크 유지 |
| architecture-overview.html | 원본 SHA-256 유지; 구조도 snapshot, href/src 참조 없음 |
| agent-platform-research.bento.html | 원본 SHA-256 유지; 압축 runtime·내장 data snapshot 보존, href/src 외부 참조 없음 |

신규 문서는 review-notes.md(현재 근거), organization-log.md(검토 증거), research/README.md(조사 목차) 3개다. 외부 제품의 나머지 당시 기능 표·quota·가격/라이선스 운영 해석과 사내 구성은 미확인이며 전체 최신이라고 단정하지 않는다.


## 전체 검증 결과 (2026-10-04)

1. **목록·내용 보존**: 원본 Markdown 42개가 모두 남아 있고 개별 검토 결과를 위 표에 기록했다. 현재 Markdown은 신규 3개를 포함해 45개다. 조사 3개의 역사 본문, 티켓 30개의 과거 완료 보고, 명시적으로 고친 06을 제외한 원본 코드 블록 본문을 원본 스냅샷과 대조했다. HTML 2개의 SHA-256이 원본과 같다. 삭제·이동은 없다.
2. **기술·예제**: 현재 검토의 수정 주장은 공식 출처·적용 조건·확인일을 연결했다. Protocol 예제와 pytest 9.0.2 임시 fixture를 검증했다. JSON/JSONC 예제 8개 블록의 값 10개가 파싱되며, diff 형태의 PATCH 예제는 독립 JSON 파일이 아님을 표시했다. 티켓 30개의 명시적 선행 관계가 wave 0~7에서 항상 이전 wave에 놓이는지 검사했다. 별도 구현 앱과 사내 서비스는 실행하지 않았다.
3. **참조·메타데이터·읽기**: Markdown 45개의 상대 링크·앵커와 메타데이터 검사에서 새 깨진 참조 0개, 기존 깨진 참조 0개다. Obsidian CLI로 동일한 pm_notes vault의 45개 속성 및 reviewed_on을 확인했다. 읽기 화면에서 조사 README의 한국어·별칭·날짜·3행 목차를 확인하고, 상위 현재 검토 링크를 클릭해 platform/review-notes로 이동함을 breadcrumb로 확인했다. 표 안 링크의 첫 클릭은 이동하지 않았고 좌표 재시도는 windowNotFoundAtPosition 오류였으므로 그 클릭 성공은 주장하지 않는다. HTML runtime 렌더링과 모든 문서의 모든 화면을 개별 확인한 것은 아니다.

검사 증거는 이번 세션의 원본 스냅샷·검사 JSON 및 CLI 결과로 남겼다. 외부 제품의 모든 당시 기능 표가 현재 재검증된 것은 아니다. Claude 협의 대기 계약과 사내 실행 검증은 보류 상태이며, partial은 이 경계를 뜻한다.


## 추가 참조 재검증 — 2026-10-05

2026-10-04기술검토에대한최종호환검사에서신규GitHub slug앵커5개를발견했다. 기존Obsidian검사에서그방식의절이동실패가관측됐으므로4개문서의해당참조를명시적노트경로로바꾸고표시용어/절이름을남겼다. 파일/제목·개념관계·기술본문/예제는유지하며추가내용통합은없다. 선택지는renderer별앵커병기또는노트+절이름참조였고,추가플러그인/보장없는HTML앵커를요구하지않는후자를권고/적용했다. Claude연결불가로추가설계협의는없다.45개파일참조/metadata·diff재검사에서새/기존참조오류0이다. 실제절클릭/전수화면·회사실행은여전히미완료다.
