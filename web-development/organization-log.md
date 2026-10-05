---
tags: [web-development, organization, verification]
aliases: [웹 개발 문서 정리 기록]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
category_major: "웹 개발"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# 웹 개발 문서 정리 기록

기준일 **2026-10-04**. 원본 Markdown 14개를 목록화하고 임시 원본 스냅샷을 저장했다. 실행 코드·lock·설정·첨부·생성된 .nuxt 파일은 수정하지 않는다. 이 폴더의 원본 개별 검토와 세 단계 검증 결과를 아래에 기록한다. 외부 실행·Claude 협의 대기는 남아 있다.

## 확인한 분포와 읽기 순서 후보

Python 환경 2개·Redis 1개, 테스트 3개, TypeScript 목차·설정·Vite·Bun 4개, Vue 목차·타입·상태 관리 3개, Bun 실행 예제 README 1개다. 최상위 README와 Python/testing 목차가 없다. TypeScript 로드맵의 미작성 문서 링크와 Bun의 Windows 절대 경로는 이후 수정 대상이다. job-scheduler에는 이번 탐색에서 IDE/.nuxt 생성 자료만 확인됐으며 완성된 원본 앱이 있다는 사실을 추정하지 않는다.

## 이번 수정과 근거

- python/uv-package-manager.md: uv add와 pip install, run과 셸 activate를 구분했다. 요구 파일 셸 분해를 공식 `uv add -r`로 바꾸고 환경 생성 순서를 수정했다. 속도 배수·모든 문제 해결·플랫폼 재현성을 보장으로 쓰지 않는다. FastAPI 미작성 링크를 기존 마이그레이션 문서로 교체했다.
- python/pip-to-uv-migration.md: 버전 핀을 자동 제거하는 예제를 없애고 요구 파일 파서로 이전한다. 최소 init, 기존 환경·빌드 설정 보존, 인덱스 선택, lock 검사, Docker 소스 복사 순서·런타임 Python 조건을 명시했다. 셸 블록의 TOML을 분리했다. uv.lock 한쪽 강제 선택·기존 파일 일괄 삭제 지시를 검토 절차로 바꿨다.
- python/redis-python.md: pipeline 배치와 transaction, rollback 부재를 구분했다. BLMOVE와 원문 ACK를 사용하고 pending 복구·중복·TTL 락 제한을 표시했다. 잘못된 Sliding Window 설명을 첫 요청 기준 고정 윈도우로 수정하고 INCR/EXPIRE를 Lua로 묶었다. Stream의 모든 ResponseError를 숨기지 않는다. 캐시와 Hash의 같은 키 타입 충돌·핸들러 생략·실제 연결 미확인을 설명했다.

현재 근거는 각 문서 끝에 공식 출처·버전·확인일을 연결했다. uv CLI는 0.12.13이며 설치된 최신 버전이라고 단정하지 않는다. Redis 서버·클라이언트 설치 버전은 미확인이고 열람한 redis-py pipeline 문서는 8.1.0이다.

## Claude 협의와 보류

HERDR_ENV=1 환경의 읽기 전용 `herdr pane current --current`도 pane_not_found였다. 관련 없는 pane을 제어하지 않았고 Claude 의견이 있는 것으로 쓰지 않는다. 공통 설치 설명과 테스트 예제를 대표 문서로 완전 통합할지, 고유 시나리오를 어느 문서로 옮길지는 협의 연결까지 보류했다. 직접 확인 가능한 명령·동작 정정은 계속한다.

## 중간 검증 결과

- uv 0.12.13 임시 폴더에서 bare init이 기존 README/main을 보존했다. 외부 네트워크 없이 합성 wheel로 requirements의 정확 버전·환경 마커가 이전됨을 확인했다. uv는 python_version 마커를 동등한 python_full_version 형태로 정규화했다. 최초 fixture의 --frozen/--no-sync 조합은 CLI에서 거부되어 수정했고, 마커 문자열 그대로 비교하는 잘못된 검사는 의미 비교로 고쳤다. 실제 패키지 설치·사내 인덱스 호출은 하지 않았다.
- 수정 Python 문서의 Python AST 11개·TOML 6개가 파싱됐다. Redis 처리 목록 예제는 fake fixture에서 RIGHT→LEFT 인자와 공백 있는 원문 LREM 보존을 확인했다. 실제 Redis의 Lua 원자성·만료·복구·failover 실행 증거는 아니다.
- Bun 1.3.6의 기존 app.test.ts는 4 pass, 0 fail, 10 expect다. 코드·lock을 수정하거나 서버를 시작하지 않았다. 전체 입력·배포·TypeScript typecheck의 증명은 아니다.
- git diff --check는 통과했다. 전체 14개 문서의 메타데이터·링크 검사와 Obsidian CLI/읽기 화면 검증은 아직 미완료다.

## 원본별 검토 상태

| 문서 | 상태 |
|---|---|
| python/uv-package-manager.md | 본문 검토·공식 명령 정정·중간 파싱/fixture 완료; 최종 참조/UI 대기 |
| python/pip-to-uv-migration.md | 본문 검토·이전/빌드 조건 정정·중간 파싱/fixture 완료; 최종 참조/UI 대기 |
| python/redis-python.md | 본문 검토·보장/예제 정정·중간 파싱/fake fixture 완료; 실제 Redis·최종 참조/UI 대기 |
| typescript/README.md | 본문 검토·계획/실제 파일 구분·깨진 계획 링크 제거; 최종 검증 대기 |
| typescript/tsconfig-setup.md | paths/runtime·TS6 baseUrl·Git glob 정정; Git attribute fixture pass, tsc/lint 미확인 |
| typescript/vite-basics.md | Vite8 Oxc/Rolldown·manualChunks·env 조건 정정; Vite7 고유 vendor 예제 보존, build 미실행 |
| typescript/bun-basics.md | Windows 절대 참조 복구·bunx 설치/runtime·타입 검사·lifecycle 조건 정정; 배포 미확인 |
| typescript/bun/example-task-api/README.md | 한국어 실행·요청·한계 안내; 기존 테스트4pass, null 입력 TypeError 로컬 확인; 코드 보존 |
| typescript/vue/README.md | 타입→상태 관리 읽기 순서 추가; 실제 파일 참조 |
| typescript/vue/vue3-with-typescript.md | macro 대안 분리·optional 기본값·응답 envelope 일치·정적/런타임 검증 구분; SFC 컴파일 미확인 |
| typescript/vue/props-emit-vs-pinia.md | 중첩 props 변경·공유 상태 판단 일반화·생략된 import/type/SSR 조건 안내; 실사용 미확인 |
| testing/unit-testing-basics.md | 본문 검토·보장/비율 조건·Vitest 설정·boolean 반환·로케일 정정; calculator10pass, 외부 모듈 생략 조건 명시 |
| testing/testing-frameworks.md | 본문 검토·jsdom/config·전역 mock 복원·플러그인 이름/버전 정정; pytest6pass/unittest4pass, Vitest 실실행 미확인 |
| testing/e2e-testing-basics.md | 본문 검토·locator/import·use.baseURL·인증 page·retry/report·Actions 조건 정정; 브라우저/CI 미실행 |


## TypeScript·Vue 단계의 변경과 증거

- 원본 TypeScript/Vue 계열 8개를 검토·정정하고 reviewed_on을 추가했다. TypeScript 목차의 미작성 주제는 링크 대신 계획으로 보존한다. Bun의 Windows 절대 경로를 같은 폴더의 상대 경로로 복구했다. props/emit macro 대안을 별도 SFC 블록으로 나누고 컴파일 오류를 숨기지 않았다.
- Vite는 공식 8 문서에서 Oxc 변환·Rolldown 빌드/최적화, 객체 manualChunks 제거를 확인했다. 기존 Vue/Router/Pinia vendor 묶음은 7 이하 정책 조각으로 보존했다. TypeScript 6.0 baseUrl deprecated와 paths 출력 불변을 공식 문서에 연결했다. 어떤 버전도 이 저장소 설치 최신이라고 단정하지 않는다.
- TS/JS·SFC script 조각 33개를 Bun.Transpiler로 구문 검사했다. 최초 검사는 원래 잘못된/권장 예의 중복 user 선언 때문에 실패해 unsafeUser로 구분한 뒤 통과했다. transpiler는 의미 타입 검사나 Vue SFC compiler가 아니다. tsc·node_modules가 없어 정적 타입·lint·Vue compiler 실행은 미확인이다.
- 임시 Git 저장소에서 정정한 gitattributes가 png/jpg/jpeg/gif/webp/ico 6종에 binary를 적용하고 svg에는 적용하지 않음을 확인했다. 실제 저장소 설정은 수정하지 않았다.
- Bun 원본 handler에 Request(body="null")를 전달하니 TypeError였다. 기존 테스트의 4 pass가 이 입력까지 보증하지 않음을 사용 안내에 반영했다. 운영 JSON 검증·인증·영속 저장·GET /tasks/:id는 미구현이며 코드는 보존했다.
- 최상위·Python·testing·Bun 하위 README 4개를 추가했다. 원본 14개는 그대로 있고 Markdown은 19개다. 비Markdown 원본 185개의 해시가 초기 상태와 같다.
- HERDR_ENV=1을 다시 확인했으며 current pane은 여전히 pane_not_found다. 실제 Claude 검토 의견은 없고 공통 설명 통합 등 협의가 필요한 결정은 보류한다.

위 TypeScript 단계 시점에는 테스트 문서와 전체 검증이 남아 있었다. 아래 최종 단계 결과가 현재 상태다. partial은 외부 실행 미확인과 협의 대기를 뜻한다.

### 단계별 참조 검사

현재 Markdown 19개를 검사한 결과 새 깨진 상대 참조는 0개다. 기존 누락 참조 1개는 아직 정정 전인 testing/unit-testing-basics.md의 FastAPI 링크다. 테스트 원본 3개는 아직 메타데이터 검사 대상에서 제외했으며 원본과 일치함을 확인했다. 따라서 이 결과는 전체 문서 검토/메타데이터 완료 증거가 아니다. git diff --check도 통과했다.


## 테스트 문서의 정정

- Unit: 비용 배수·전체 검증 보장을 설명용 가정/범위로 정정했다. Vitest test 옵션의 import, optional chaining의 boolean 반환, 가격 문자열의 로케일을 명시하고 빈 함수 이름 조각에 pass를 추가했다. FastAPI 미작성 참조를 계획으로 표시했다. 이메일·비밀번호 최소 예제는 실제 보안 정책이 아니다.
- Frameworks: jsdom 설치, Vitest config 타입, Object.is matcher, 전역 fetch stub 복원, vi hook import, pytest-django 이름을 수정했다. Jest 단순 치환·React 기본 포함·시장 표준/상대 속도 단정을 제거/조건화했다. 열람한 Vitest 5 조건과 실제 설치되지 않은 도구를 구분했다.
- E2E: get_by_name을 CSS name locator로 바꾸고 re/expect/pytest import를 복구했다. 저장 context에서 실제 page를 생성하고 기본 fixture와의 연결을 설명했다. baseURL을 use로 옮기고 retry/html reporter·CI server reuse 조건을 추가했다. upload-artifact 공식 사용 예의 v7과 GHES 별도 제약·사내 runner 미확인을 표시했다. 고유 Google·로그인·storage·codegen·trace 예제는 유지했다.

## 세 단계 최종 검증 (2026-10-04)

1. **목록·고유 내용**: 원본 14개 모두 같은 경로에 남아 있고 신규 5개를 포함해 Markdown은 19개다. 개별 검토 표를 전부 갱신했다. 원본/현재 섹션·코드 블록 목록을 대조했다. Vue 목차·예제 README의 한국어 재구성과 같은 문서 내 Run/Test/요청 블록 결합 외에는 문서를 삭제·이동하지 않았다. Vite7 vendor 정책은 별도 버전 조각, Vue macro 대안은 별도 SFC로 보존했다. uv의 핀 제거·일괄 삭제 명령은 잘못된 절차로 정정했고 같은 전환 목적을 보존했다. 학습 비교·고유 handler/테스트·Troubleshooting 문맥을 남겼다. 완전 통합의 판단은 Claude 연결까지 보류하며 각각의 목적을 목차/본문에서 설명한다.
2. **기술·로컬 예제**: 수정 주장의 공식 출처·확인일·적용 조건은 각 문서 검토 절에 있다. pytest9.0.2의 calculator 10개/basic6개 및 unittest4개가 임시 폴더에서 통과했다. TypeScript 순수 함수는 Bun으로 이메일4조건·로케일·clamp를 확인했다. Python AST31블록과 TS/JS/SFC script 구문49블록이 파싱됐다. 앞 단계의 uv 오프라인 합성 wheel·Redis fake 원문 ACK·Git attrs fixture 및 Bun 기존4테스트 결과도 유효하다. AST·transpiler는 semantic typecheck나 Vue SFC compiler가 아니다. Playwright·Vue·Redis 서버·Docker·Actions·사내 인덱스 실행은 미확인이다.
3. **참조·속성·읽기**: 전체 19문서의 상대 링크/앵커 검사에서 새/기존 깨진 참조 모두0개다. YAML으로19개 reviewed_on과 자료형을 확인하고 중복 속성 키도 검사했다. Obsidian CLI는 실제 pm_notes vault에서19개 속성·날짜를 읽었다. 읽기 화면에서 web-development 목차→testing 목차→unit-testing-basics를 클릭해 breadcrumb가 같은 주제 안의 정확한 경로임을 확인했다. 한국어 본문·원 날짜/검토 날짜·태그·읽기 모드 screenshot을 확인했다. 모든 문서의 모든 화면·외부 사이트까지 렌더링한 증거는 아니다.

원본 비Markdown185개의 SHA-256이 초기 상태와 같다. 실행 코드·설정·lock·IDE/.nuxt 자료는 보존했다. git diff --check 통과, 커밋·푸시 없음. 최종 기록 이후 참조/속성 검사를 다시 수행한다. 앱과 vault 연결은 성공했으며 미완료는 외부 실행·정적 typecheck/컴파일러·Claude 협의다.
