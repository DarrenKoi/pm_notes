---
tags: [platform, gitlab, registry, distribution, research]
level: advanced
last_updated: 2026-09-03
---

# GitLab을 배포 백엔드로 쓸 수 있는가 — 1차 출처 조사

> 사내에 GitLab(self-managed)이 제공된다는 전제에서, [02-architecture.md](../02-architecture.md) 의 L1~L10·L6b 중 어디까지를 GitLab이 실제로 대체하는지 docs.gitlab.com 기준으로 확인한 결과. 결론부터: **아티팩트 저장·인증·발행 게이트는 대체 가능, 카탈로그·검색·텔레메트리·연합·Endpoint Broker 는 대체 불가.**

---

## 왜 필요한가? (Why)

현재 설계는 가용 스택에 OCI 레지스트리가 없다는 전제(G4)에서 출발해 **MinIO Object Lock + 자체 매니페스트**로 digest 불변성을 직접 구현하기로 했고, 인증·발행 파이프라인·심사 게이트를 전부 자체 FastAPI 앱 안에 넣었다. GitLab 이 제공된다면 그 중 상당 부분이 이미 만들어져 있는 기능이다. 무엇이 실제로 겹치는지 확인하지 않으면 두 가지 실패 중 하나로 간다.

| 실패 | 증상 |
|---|---|
| 과소평가 | 이미 있는 것(불변 태그, 서명 저장, 승인 규칙, 감사)을 다시 만든다 |
| 과대평가 | "GitLab이 다 해준다"고 가정했다가 티어 부족·기능 부재로 설계가 중간에 무너진다 |

**티어가 답을 결정한다.** 아래 조사에서 핵심 기능 다수가 Premium/Ultimate 이고, 그 중 하나(불변 컨테이너 태그)는 Ultimate 전용이다. 회사 라이선스가 Free 이면 이 문서의 결론이 절반으로 줄어든다 → [열린 질문 GQ1](#열린-질문-사용자가-답해야-할-것).

---

## 핵심 개념 (What)

### 1. 조사 결과 요약 — 기능별 티어·self-managed 표

모든 행의 근거는 §2 이후에 개별 링크로 있다. `SM` = GitLab Self-Managed 지원.

| 기능 | 티어 | SM | 상태 | 비고 |
|---|---|---|---|---|
| Generic package registry (publish/download) | **Free** | O | GA | 프로젝트 레벨 전용 |
| Container registry | **Free** | O | GA | **관리자가 인스턴스에서 켜야 함** |
| Protected container tags | **Free** | O | GA (17.11) | metadata DB 필요 |
| **Immutable container tags** | **Ultimate** | O | GA (18.10) | metadata DB 필요, 프로젝트당 규칙 5개 |
| Container registry metadata DB | Free | O | GA (17.3) | PostgreSQL 별도 구성 |
| Deploy token / group access token | Free | O | GA | 스코프 분리 가능 |
| CI_JOB_TOKEN | Free | O | GA | 인스턴스 경계 넘지 못함 |
| GitLab = OIDC provider | **Free** | O | GA | `.well-known/openid-configuration` |
| 인스턴스 전역 SAML SSO (외부 IdP) | **Free** | O | GA | |
| MR 승인 **규칙**(강제) | **Premium** | O | GA | Free 는 승인만 가능, 차단 불가 |
| CODEOWNERS | **Premium** | O | GA | |
| 그룹/프로젝트 감사 이벤트 + API | **Premium** | O | GA | 보존 무기한 |
| Advanced search | **Premium** | O | GA | 패키지·레지스트리 **미색인** |
| Geo 복제 | **Premium** | O(SM 전용) | GA | 단일 논리 인스턴스 |
| Pull mirroring | **Premium** | O | GA | 리포지터리만 |
| Push mirroring | Free | O | GA | 리포지터리만 |
| 프로젝트 export/import | Free | O | GA | **패키지·레지스트리 이미지 제외** |
| SAST (기본 분석기) | Free | O | GA | 취약점 관리 UI 는 Ultimate |
| Secret detection | Free | O | GA | |
| Dependency proxy | Free | O | GA | **Docker Hub 전용 → 폐쇄망 무의미** |

---

### 2. Package Registry — generic packages

- **티어·오퍼링:** "Tier: Free, Premium, Ultimate", "Offering: GitLab.com, GitLab Self-Managed, GitLab Dedicated" — https://docs.gitlab.com/user/packages/generic_packages/
- **API:** 발행 `PUT /projects/:id/packages/generic/:package_name/:package_version/:file_name`, 다운로드 `GET /projects/:id/packages/generic/:package_name/:package_version/:file_name` — https://docs.gitlab.com/user/packages/generic_packages/
- **인증:** personal access token(`api`), project access token(`api`, Developer 이상), CI/CD job token, deploy token(`read_package_registry`/`write_package_registry`), HTTP Basic — https://docs.gitlab.com/user/packages/generic_packages/
- **체크섬:** 업로드 파일마다 SHA256 이 자동 계산·저장되고 `X-Checksum-SHA256` 응답 헤더와 API 의 `file_sha256` 필드로 노출된다 — https://docs.gitlab.com/user/packages/generic_packages/ , https://docs.gitlab.com/api/packages/ . 포맷별 지원표에도 generic 의 hash 지원이 SHA256 으로 명시돼 있다 — https://docs.gitlab.com/user/packages/package_registry/supported_functionality/
- **불변성(중요):** 기본값은 **중복 허용**이다. 같은 name+version 으로 다시 올리면 기존 패키지에 파일이 추가된다. Owner 역할이 이 동작을 끌 수 있고, 정규식으로 예외를 둘 수 있다 — https://docs.gitlab.com/user/packages/generic_packages/ . 포맷 지원표도 generic 의 duplicate 처리를 "Yes (configurable)" 로 적는다 — https://docs.gitlab.com/user/packages/package_registry/supported_functionality/
  - → **이것은 "중복 발행 거부" 설정이지 Object Lock 수준의 불변성이 아니다.** 패키지 삭제는 Maintainer/Owner 가 할 수 있다(https://docs.gitlab.com/user/permissions/). 패키지 레지스트리 문서 어디에도 "발행된 버전을 삭제 불가로 만드는" 기능은 없다 — https://docs.gitlab.com/user/packages/package_registry/
- **크기 한도:** generic 기본 5 GB. self-managed 관리자는 Admin Area 또는 Rails 콘솔(`Plan.default.actual_limits.update!(...)`)로 변경하고 `0` 은 무제한 — https://docs.gitlab.com/administration/instance_limits/ . 별도로 S3 호환 스토리지의 단일 PUT 요청 한도 5 GB 가 겹친다 — https://docs.gitlab.com/user/packages/generic_packages/
- **활성화:** 패키지 레지스트리는 self-managed 에서 기본적으로 켜져 있다("automatically turned on") — https://docs.gitlab.com/user/packages/package_registry/

---

### 3. Container Registry / OCI 아티팩트 — 여기서 기대가 깎인다

- **티어:** Free/Premium/Ultimate, GitLab.com·Self-Managed·Dedicated. 단 **"An administrator must enable the container registry for your GitLab instance."** — https://docs.gitlab.com/user/packages/container_registry/
- **지원 포맷:** "The container registry supports the Docker V2 and Open Container Initiative (OCI) image formats" 이고 OCI distribution spec 을 따른다 — https://docs.gitlab.com/user/packages/container_registry/
- **임의 OCI 아티팩트(핵심 질문):**
  - `subject` 필드는 지원한다 — "you can use the OCI 1.1 manifest `subject` field to associate container images with Cosign signatures", 문서가 `cosign sign --registry-referrers-mode oci-1-1` 예시를 든다 — https://docs.gitlab.com/user/packages/container_registry/
  - 그러나 **"it does not fully implement the OCI 1.1 Referrers API"** 라고 문서가 직접 못 박는다 — https://docs.gitlab.com/user/packages/container_registry/
  - **`artifactType` 을 쓰는 빈 config 임의 아티팩트(ORAS 스타일)의 지원 여부는 docs.gitlab.com 에서 확인되지 않았다.** 관련 이슈가 열려 있는 상태로 보이나(https://gitlab.com/gitlab-org/container-registry/-/issues/967 , https://gitlab.com/gitlab-org/gitlab-foss/-/issues/43202), 이는 공식 문서가 아니므로 **"지원된다"고 가정하면 안 된다.** 실제 사내 GitLab 버전에서 ORAS push 를 한 번 시험해 확인해야 한다 → [GQ4](#열린-질문-사용자가-답해야-할-것)
- **Referrers API 엔드포인트:** Container Registry API 문서에 referrers 전용 엔드포인트는 없다. 대신 저장소·태그 목록/삭제, 그룹 단위 목록(`GET /groups/:id/registry/repositories`), 태그 digest 노출은 있다 — https://docs.gitlab.com/api/container_registry/
- **불변 태그 (Immutable container tags):**
  - **Ultimate 전용**, GitLab.com·Self-Managed. 18.1 experiment → 18.2 beta → **18.10 GA**. 프로젝트당 규칙 최대 5개, RE2 정규식 100자 한도. 여러 규칙이 걸리면 가장 엄격한 것이 적용 — https://docs.gitlab.com/user/packages/container_registry/immutable_container_tags/
  - 주의: 불변 규칙이 하나라도 있으면 **패턴과 무관하게** 해당 프로젝트의 직접 manifest 삭제 요청이 전부 막힌다. 규칙 변경은 JWT 만료(기본 5분) 후 반영. **self-managed 는 container registry metadata database 가 필수** — https://docs.gitlab.com/user/packages/container_registry/immutable_container_tags/
- **보호 태그 (Protected container tags):** Free 티어, 17.9 experiment → **17.11 GA**. 프로젝트당 5개 규칙, 태그별로 push/delete 최소 역할(Maintainer/Owner/Administrator)을 지정 — https://docs.gitlab.com/user/packages/container_registry/protected_container_tags/
- **metadata database:** PostgreSQL 기반, **17.3 GA**. 보호·불변 태그 등 신기능은 metadata DB 버전에만 구현된다. self-managed 는 전용 PostgreSQL 준비, 기존 레지스트리는 1단계/3단계 import(읽기 전용 구간 발생), 백업 절차에 레지스트리 DB 포함, 업그레이드 시 마이그레이션 실행이 필요 — https://docs.gitlab.com/administration/packages/container_registry_metadata_database/
- **cosign:** GitLab 공식 서명 예제는 **Sigstore public infrastructure 기반 keyless** 만 다룬다(Fulcio/Rekor). GitLab OIDC 토큰으로 단명 키를 받고 certificate transparency log 에 기록한 뒤 폐기하는 방식 — https://docs.gitlab.com/ci/yaml/signing_examples/ . **자체 키 서명이나 private Sigstore 배포는 이 문서가 다루지 않는다.** 폐쇄망에서는 keyless 를 쓸 수 없으므로 기존 설계대로 cosign 자체 키를 유지해야 한다(§10).

**요약:** GitLab 컨테이너 레지스트리는 digest 기반 불변성과 서명 첨부를 **컨테이너 이미지에 대해서는** 사실상 제공한다. 그러나 (a) Ultimate 이어야 하고, (b) metadata DB 구성이 선행돼야 하고, (c) 임의 OCI 아티팩트로 skill 번들을 넣는 경로는 문서로 보증되지 않는다. **"MinIO + 자체 매니페스트를 통째로 버릴 수 있다"는 결론은 현재 근거로는 나오지 않는다.**

---

### 4. Rename 문제 — GitLab 이 실제로 푸는가

우리 요구사항(G5: 팀명이 매년 바뀐다)에 대한 GitLab 의 답을 항목별로.

| 질문 | 답 | 출처 |
|---|---|---|
| 경로와 분리된 안정적 숫자 ID 가 있는가 | **있다.** REST API 의 `id` 는 "ID that is unique across all projects" 이고 API 로 직접 주소지정 가능(`GET /projects/42/...`) | https://docs.gitlab.com/api/rest/ |
| 경로로도 주소지정 가능한가 | 가능. `NAMESPACE/PROJECT_PATH` 를 URL 인코딩(`/` → `%2F`) | https://docs.gitlab.com/api/rest/ |
| rename 시 리다이렉트가 생기는가 | 생긴다. "When a repository path changes, GitLab handles the transition from the old location to the new one with a redirect", 그룹도 "The old group URL is redirected to the new group URL" | https://docs.gitlab.com/user/project/repository/ , https://docs.gitlab.com/user/group/manage/ |
| 리다이렉트가 깨지는 조건 | **다른 그룹·사용자·프로젝트가 옛 경로를 차지하면 즉시 사라진다** ("as long as the original path is not claimed by another group, user, or project") | https://docs.gitlab.com/user/project/repository/ |
| 리다이렉트를 따라가지 **않는** 것 | CI/CD `include`(컴포넌트 제외, 실패 시 syntax error), **경로 기반 API 호출**, **CI 의 Docker 이미지 참조**, 프로젝트/네임스페이스를 지정하는 변수, **CODEOWNERS** — 전부 수동 갱신 대상 | https://docs.gitlab.com/user/project/repository/ |
| 레지스트리 경로가 프로젝트 경로를 따라가는가 | **따라간다.** "The path of a container repository always matches the related project's repository path, so renaming or moving only the container registry is not possible." | https://docs.gitlab.com/user/packages/container_registry/ |
| 이미 발행된 이미지가 있는데 rename 하면 | **막힌다.** "It is not possible to rename a namespace if it contains a project with Container Registry tags, because the project cannot be moved." | https://docs.gitlab.com/user/group/manage/ |
| 이미 발행된 패키지는 | 그룹 엔드포인트를 쓰는 패키지는 재설정이 필요하고, 인스턴스 레벨 엔드포인트를 쓰면 패키지 이름 자체를 바꿔야 할 수 있다. npm 규약 이름 패키지가 있으면 최상위 그룹으로의 이전이 실패한다 | https://docs.gitlab.com/user/group/manage/ |
| GitLab 자신의 권고 | 경로 변경 대신 **"create a new group and transfer projects to it instead"** | https://docs.gitlab.com/user/group/manage/ |

#### 판정

**GitLab 은 npm-scope 실패모드를 "절반만" 피한다.**

- npm 보다 나은 점: 안정적 숫자 ID 가 있고 API 가 그것으로 주소지정할 수 있다. 리다이렉트가 있어 rename 이 곧 전 패키지 재발행을 뜻하지 않는다.
- npm 과 **똑같이 나쁜 점**: 사람이 보고 쓰는 식별자(그룹/프로젝트 path, 컨테이너 이미지 경로, 패키지 URL)는 여전히 **경로 기반**이고, 컨테이너 태그가 존재하면 rename 자체가 차단된다. 리다이렉트는 옛 경로 재사용 한 번으로 소멸하는 **약한 보장**이다.

**따라서 우리 설계의 §6 원칙(네임스페이스에 팀명 금지, `team_id` 사용)은 GitLab 을 도입해도 그대로 유지해야 한다.** GitLab 그룹 path 를 `dram-ai-solution-2team` 이 아니라 `t-004821` 처럼 불변 코드로 만들고, 표시명(group name)만 매년 바꾸는 방식이다. GitLab 이 rename 문제를 대신 풀어주는 게 아니라, **우리가 rename 을 안 하도록 GitLab 을 쓰는 것**이 답이다.

---

## 어떻게 사용하는가? (How)

### 5. 계층 대체 판정표 (가장 중요한 산출물)

| 계층 | 대체 가능? | GitLab 기능 | 티어 | 남는 문제 |
|---|---|---|---|---|
| **L1 Artifact Store** | **부분** | Generic package registry(https://docs.gitlab.com/user/packages/generic_packages/) + Container registry(https://docs.gitlab.com/user/packages/container_registry/) | Free (불변 태그는 **Ultimate**) | 패키지 레지스트리에는 불변성이 없고 Maintainer 가 삭제 가능(https://docs.gitlab.com/user/permissions/). 컨테이너 불변 태그는 Ultimate + metadata DB 필요. 임의 OCI 아티팩트 지원 미확인 |
| **L2 Catalog** | **불가** | 없음 | — | GitLab 에 "정규화된 자산 문서 + 소유권 튜플 + 심사 상태"를 담을 질의 가능한 저장소가 없다. 리포지터리 안의 JSON 파일로 둘 수는 있으나 인덱싱·질의·상태전이가 안 된다. **MongoDB 유지** |
| **L3 Search Index** | **불가** | Advanced search | Premium/Ultimate | 색인 스코프는 code, comments, commits, groups, work items, MR, milestones, projects, users, wikis 뿐이고 **패키지·컨테이너 레지스트리는 색인 대상이 아니다**(https://docs.gitlab.com/user/search/advanced_search/). 기본 검색에도 없다(https://docs.gitlab.com/user/search/). **OpenSearch 유지** |
| **L4 API** | **부분** | REST API(https://docs.gitlab.com/api/rest/) | Free | 저장·인증은 GitLab API 로 위임 가능하나, `server.json` 확장 스키마·`/v0.1/servers`·`marketplace.json` 계약은 GitLab 이 제공하지 않는다. **FastAPI 는 남되 얇아진다** |
| **L5 Auth** | **전부(조건부)** | GitLab as OIDC provider(https://docs.gitlab.com/integration/openid_connect_provider/), 인스턴스 SAML SSO(https://docs.gitlab.com/integration/saml/), deploy token(https://docs.gitlab.com/user/project/deploy_tokens/), group access token(https://docs.gitlab.com/user/group/settings/group_access_tokens/), CI_JOB_TOKEN(https://docs.gitlab.com/ci/jobs/ci_job_token/) | Free | GitLab 이 IdP 역할을 하면 우리 앱이 RP 가 된다. 단 §7 의 RFC 8693 token exchange 는 GitLab 문서에서 확인되지 않았다. 사내 IdP 를 GitLab **뒤**에 둘지 **옆**에 둘지 결정 필요 |
| **L6 Distribution** | **부분** | Generic package download API, presigned object storage URL | Free | 다운로드 표면은 대체 가능. 런타임별 어댑터(`marketplace.json` 동적 생성)는 여전히 우리 몫 |
| **L6b Endpoint Broker** | **불가** | 없음 | — | hosted agent 헬스체크·접근 토큰 발급·호출 계량에 대응하는 GitLab 기능이 없다. 전량 자체 구현 |
| **L7 Cache/Queue** | **불가** | 없음 | — | Redis 유지 |
| **L8 Observability** | **거의 불가** | 감사 이벤트(https://docs.gitlab.com/user/compliance/audit_events/), 웹훅(https://docs.gitlab.com/user/project/integrations/webhook_events/) | 감사는 **Premium** | §7 참조. **패키지 다운로드 수·레지스트리 pull 이벤트가 없다.** OTel 파이프라인 유지 |
| **L9 Federation** | **부분(방향이 안 맞음)** | Geo(https://docs.gitlab.com/administration/geo/), 미러링(https://docs.gitlab.com/user/project/repository/mirror/), export/import(https://docs.gitlab.com/user/project/settings/import_export/) | Geo·pull mirror **Premium** | §8 참조. 셋 다 "1차 → 2차 승격" 의미론에 맞지 않는다. **자체 push+pull 승격 유지** |
| **L10 Governance** | **부분(꽤 큼)** | Protected container tags(Free), MR 승인 규칙(**Premium**), CODEOWNERS(**Premium**), protected tags, CI 파이프라인, SAST(Free), secret detection(Free) | 혼재 | 심사 게이트를 GitLab MR + CI 로 옮길 수 있다. 단 **강제 승인은 Premium 부터**. cosign keyless 는 폐쇄망 불가 → 자체 키 유지 |

---

### 6. 발행 파이프라인을 GitLab CI 로 옮길 수 있는가 — 대체로 예

| 우리 단계(§8) | GitLab 대응 | 티어 | 근거 |
|---|---|---|---|
| ① 매니페스트 검증 | CI job (스크립트) | Free | — |
| ② 정적 스캔 | SAST + Secret detection | Free (취약점 관리 UI 는 Ultimate) | https://docs.gitlab.com/user/application_security/sast/ , https://docs.gitlab.com/user/application_security/secret_detection/ |
| ③ 서명 | CI job 에서 cosign 자체 키 | Free | keyless 는 Sigstore 공용 인프라 의존 → 폐쇄망 불가(https://docs.gitlab.com/ci/yaml/signing_examples/) |
| ④ 아티팩트 커밋 | generic package publish / registry push (CI_JOB_TOKEN) | Free | https://docs.gitlab.com/ci/jobs/ci_job_token/ |
| ⑤ 카탈로그 등록 | **GitLab 기능 없음** — 우리 API 호출 | — | L2 미대체 |
| ⑥ 사람 심사 | MR 승인 규칙 + CODEOWNERS | **Premium** | Free 에서는 Developer 이상이 승인할 수 있으나 승인 없이도 머지가 가능해 **차단 게이트가 되지 않는다**(https://docs.gitlab.com/user/project/merge_requests/approvals/) |
| ⑦ active + 색인 | 우리 워커 | — | |

**SAST 의 한계(중요):** 지원 언어 목록에 C/C++/C#/Go/Java/JS/PHP/Python/Ruby/TS/YAML 등이 있고, Apex/Elixir/Groovy/Kotlin/Scala 는 표준 분석기 전용이다. **셸 스크립트/bash 는 지원 언어로 명시돼 있지 않다** — https://docs.gitlab.com/user/application_security/sast/ . Skill 번들의 상당수가 셸·설정 파일이라면 **GitLab SAST 는 우리 §8-② 게이트를 대체하지 못하고 보완만 한다.** 자격증명 하드코딩은 secret detection 이 잡지만, "외부 네트워크 호출·셸/파일시스템 접근 범위" 검사는 여전히 자체 규칙이 필요하다.

**오프라인 실행:** SAST 는 폐쇄망에서 `SECURE_ANALYZERS_PREFIX` 를 사내 레지스트리로 돌려 실행할 수 있다 — https://docs.gitlab.com/user/application_security/sast/ . 단 분석기 이미지·시그니처 DB 를 인터넷 연결 중계 시스템으로 주기적으로 가져와야 하고, GitLab Self-Managed 오프라인 운영에는 **"opt-out exemption of cloud licensing"** 이 사전에 필요하다 — https://docs.gitlab.com/user/application_security/offline_deployments/

---

### 7. 텔레메트리 — GitLab 이 못 주는 것 (솔직하게)

- **패키지 다운로드 카운트가 없다.** Packages API 는 패키지 단위로 `last_downloaded_at` 만 준다. 누적 다운로드 수나 사용 통계는 없다 — https://docs.gitlab.com/api/packages/
- **패키지 발행·레지스트리 push/pull 웹훅이 없다.** 웹훅 이벤트 목록은 push, tag, MR, issue, comment, pipeline, job, deployment, **release**, milestone, feature flag, emoji, wiki, vulnerability, access token expiry 이고 **패키지·컨테이너 레지스트리 이벤트는 목록에 없다** — https://docs.gitlab.com/user/project/integrations/webhook_events/
- **감사 이벤트는 Premium 부터.** Free 에서는 "successful sign-in events are the only audit events available at all tiers". 그룹·프로젝트 감사 이벤트는 Premium/Ultimate 이고 보존은 무기한, API 는 조회 구간 최대 30일 — https://docs.gitlab.com/user/compliance/audit_events/
- 그룹 웹훅 중 member/project/subgroup 이벤트는 Premium/Ultimate — https://docs.gitlab.com/user/project/integrations/webhook_events/

→ **§10 의 "설치형은 설치 수로 판정한다"는 전략이 GitLab 위에서도 그대로 성립하지 않는다.** GitLab 을 저장소로 쓰면 다운로드는 GitLab 을 통과하는데, GitLab 이 그 횟수를 세어주지 않는다. 세려면 (a) 다운로드를 우리 API 가 프록시하거나(그러면 GitLab presigned URL 의 장점을 버림), (b) GitLab 앞단 리버스 프록시/게이트웨이 로그를 집계하거나, (c) 런타임 OTLP 에 의존해야 한다. **(b) 가 가장 현실적이다.**

---

### 8. 연합 — GitLab 이 우리 1차→2차 승격에 쓸 수 있는 것은 사실상 없다

| 후보 | 티어 | 실제로 복제하는 것 | 별도 인스턴스 간 가능? | 우리 용도 적합성 |
|---|---|---|---|---|
| **Geo** | Premium (SM 전용) | 리포지터리, LFS, 첨부. 패키지·레지스트리도 언급되나 레지스트리는 각 사이트에 별도 외부 PostgreSQL 필요 | **아니다.** "Geo operates within a single logical GitLab deployment, not between independent instances." 2차 사이트는 읽기 시 프록시, 쓰기는 primary 로 리다이렉트되는 active-passive | **부적합.** 승격은 선택적·단방향 발행이지 전체 복제가 아니다 |
| **Pull mirroring** | Premium | 브랜치·태그·커밋 | 문서가 GitLab↔GitLab 을 명시하지 않음 | 리포지터리만. 패키지·레지스트리 언급 없음 |
| **Push mirroring** | Free | 브랜치·태그·커밋 | 상동 | 상동 |
| **프로젝트 export/import** | Free | 리포지터리, wiki, 이슈, MR, 릴리스, LFS 등 | 가능. 대상 인스턴스 버전이 원본과 같거나 상위, 최대 2 마이너 뒤까지 | **결정적 결격: "Package and container registry images" 가 export 에서 제외된다** |
| **Dependency proxy** | Free | Docker Hub 이미지 pull-through 캐시 | — | **Docker Hub 전용.** 다른 GitLab 레지스트리 프록시는 문서에 없고, 애초에 인터넷 필요 → 폐쇄망 무의미 |

출처: https://docs.gitlab.com/administration/geo/ , https://docs.gitlab.com/user/project/repository/mirror/ , https://docs.gitlab.com/user/project/settings/import_export/ , https://docs.gitlab.com/user/packages/dependency_proxy/

**결론: §9 의 "결정은 push, 데이터는 pull, 아티팩트는 lazy copy" 는 GitLab 이 있어도 그대로 우리가 구현해야 한다.** GitLab 이 제공하는 것은 그 파이프라인이 읽고 쓰는 **엔드포인트**(패키지 API, 레지스트리 API)뿐이다.

**만약 회사가 GitLab 인스턴스를 하나만 준다면** 토폴로지가 근본적으로 바뀐다.
- 사업부 경계가 인스턴스가 아니라 **최상위 그룹**이 된다 → G1 의 "인스턴스 경계 = 보안 경계" 전제가 깨진다. 경계는 그룹 권한·visibility 로 내려간다.
- 반대급부로 승격이 **인스턴스 간 연합 문제에서 그룹 간 이동/참조 문제로 축소된다.** 크로스-Biz 접근(§7.3)도 토큰 연합이 아니라 그룹 멤버십 문제가 된다.
- CI_JOB_TOKEN 이 **인스턴스 경계를 넘지 못한다**는 제약(https://docs.gitlab.com/ci/jobs/ci_job_token/)이 사라지고, 대신 job token allowlist(그룹 200개·프로젝트 200개 상한)로 크로스-프로젝트 접근을 제어하게 된다.
- 반면 "한 사업부 장애가 다른 사업부에 안 번진다"는 격리 이점은 없어진다.
→ 이 분기가 아키텍처에 미치는 영향이 티어 다음으로 크다 → [GQ2](#열린-질문-사용자가-답해야-할-것)

---

### 9. 후보 토폴로지 3안

#### A안 — GitLab = 아티팩트 저장소 + 인증, 카탈로그는 우리 것 (권장 출발점)

- GitLab: generic package registry(L1), OIDC provider(L5), CI 발행 파이프라인(§8 ①~④), MR 승인 심사(⑥, Premium 시)
- 우리: MongoDB 카탈로그(L2), OpenSearch(L3), FastAPI(L4), 어댑터(L6), Endpoint Broker(L6b), OTel(L8), 승격(L9)
- **얻는 것:** MinIO Object Lock + 자체 매니페스트 + 자체 서명 저장 + 자체 토큰 발급 구현이 사라진다. 감사·권한이 GitLab 것을 재사용한다.
- **잃는 것:** 불변성이 Object Lock(GOVERNANCE) 만큼 강하지 않다 — generic package 는 Maintainer 가 지울 수 있다. 다운로드 계량을 별도로 붙여야 한다.
- **필요 티어:** Free 로도 동작. 심사 강제(⑥)에만 Premium.

#### B안 — GitLab = 레지스트리 전부, 우리는 얇은 discovery UI

- 자산 하나 = GitLab 프로젝트 1개. 버전 = release + generic package. 소유권 = 그룹 멤버십. 심사 = MR 승인. 검색 = ?
- **얻는 것:** 개발량이 가장 적다. 권한·감사·CI 가 전부 공짜.
- **잃는 것 — 치명적:**
  - **검색이 성립하지 않는다.** advanced search 가 패키지를 색인하지 않으므로, 카탈로그 검색을 하려면 결국 우리가 OpenSearch 를 세워 GitLab API 를 긁어야 한다 → "얇은 UI" 가 아니게 된다.
  - **hosted agent(자산 3종 중 하나)가 GitLab 모델에 안 들어간다.** 아티팩트가 없고 헬스체크·엔드포인트 계약·접근 토큰이 필요한데 대응 기능이 없다.
  - 런타임 중립 카탈로그(§2-1)와 `marketplace.json` 투영을 GitLab 이 못 만든다.
- **평가:** 자산이 Skill 하나뿐이었다면 유효했을 안. **D10(3종 분리) 때문에 성립하지 않는다.**

#### C안 — GitLab = CI 발행 파이프라인만, 저장·배포는 기존 MinIO

- GitLab 은 소스 보관·MR 심사·CI 실행만 하고, CI 마지막 단계가 우리 FastAPI 로 발행 API 를 호출한다.
- **얻는 것:** 기존 설계를 거의 안 건드린다. Object Lock 불변성을 그대로 유지한다.
- **잃는 것:** MinIO presigned URL·자체 토큰·자체 다운로드 API 를 여전히 다 만든다. GitLab 도입 이득이 "심사 UI + CI 러너" 로 축소된다.
- **평가:** Free 티어이고 컨테이너 레지스트리가 꺼져 있다면 이게 현실적인 안이다.

**권고: A안에서 시작하고, 사내 GitLab 이 Ultimate + 컨테이너 레지스트리 활성 + metadata DB 구성이라면 L1 을 컨테이너 레지스트리로 승격하는 것을 별도 검토.**

---

### 10. GitLab 이 우리에게 해줄 수 없는 것 (단정)

1. **자산 카탈로그.** `server.json` 확장 문서, 상태 전이(pending/approved/active/deprecated), 등급(experimental/verified/official), 승격 상태를 담을 질의 가능한 저장소가 없다.
2. **카탈로그 검색.** advanced search 는 Premium 이면서 패키지·레지스트리를 **색인하지 않는다**(https://docs.gitlab.com/user/search/advanced_search/). 어떤 티어에서도 "에이전트를 검색한다" 가 안 된다.
3. **hosted agent 지원 일체.** 엔드포인트 카탈로그, 헬스체크, SLA 추적, 접근 토큰 발급·만료, 호출 계량(L6b) — 대응 기능 없음.
4. **사용 텔레메트리.** 패키지 다운로드 카운트 없음, 레지스트리 pull 이벤트 웹훅 없음(https://docs.gitlab.com/api/packages/ , https://docs.gitlab.com/user/project/integrations/webhook_events/).
5. **1차→2차 승격.** Geo 는 단일 논리 배포 내부이고, export/import 는 패키지·레지스트리 이미지를 제외한다(https://docs.gitlab.com/user/project/settings/import_export/).
6. **패키지 레지스트리의 진짜 불변성.** 중복 발행 차단 설정은 있으나 삭제 방지가 없다. Object Lock 등가물은 컨테이너 불변 태그(Ultimate)뿐이다.
7. **Skill 번들에 대한 의미 있는 정적 분석.** SAST 지원 언어에 셸이 없다(https://docs.gitlab.com/user/application_security/sast/).
8. **폐쇄망 keyless 서명.** 공식 cosign 예제는 Sigstore 공용 인프라 전제(https://docs.gitlab.com/ci/yaml/signing_examples/).
9. **rename 을 안전하게 만드는 것.** 리다이렉트는 옛 경로 재점유로 소멸하고, 컨테이너 태그가 있으면 rename 이 차단된다(§4).
10. **런타임별 배포 표면.** `marketplace.json` 동적 생성, `/v0.1/servers` 계약은 전부 우리 몫.

---

### 11. 기존 결정에 미치는 영향

| 결정 | 영향 |
|---|---|
| **D2** (아티팩트 직접 호스팅) | **수정 후보.** MinIO 대신 GitLab generic package registry. 단 불변성이 약해지므로 D8 과 충돌 |
| **D8** (Object Lock GOVERNANCE) | **위협받음.** GitLab 패키지 레지스트리에는 등가물이 없다. 컨테이너 불변 태그는 Ultimate + metadata DB |
| **Q6** (OCI 레지스트리를 스택에 추가할 것인가) | **부분 해소.** GitLab 컨테이너 레지스트리가 답이 될 수 있으나 (a) Ultimate, (b) 관리자 활성화, (c) 임의 OCI 아티팩트 지원 미확인의 3중 조건 |
| **Q7** (사내 IdP 가 OIDC 인가) | **우회로 생김.** GitLab 이 OIDC provider 를 Free 로 제공하므로 GitLab 을 IdP 브로커로 둘 수 있다 |
| **Q11** (서명 키 보관) | 변함없음. keyless 불가이므로 자체 키 유지 |
| **D5/D6** (N-to-1, push+pull 승격) | 변함없음. GitLab 이 대체 수단을 주지 않음 |

---

## 열린 질문 (사용자가 답해야 할 것)

| ID | 질문 | 갈리는 것 |
|---|---|---|
| **GQ1** | **회사 GitLab 라이선스 티어는?** (Free / Premium / Ultimate) | Ultimate 이 아니면 불변 태그가 없어 L1 대체가 약해진다. Premium 이 아니면 심사 강제(MR 승인 규칙)·CODEOWNERS·감사 이벤트·advanced search 가 전부 빠진다 |
| **GQ2** | **인스턴스가 사업부마다 하나인가, 전사 하나를 공유하는가?** | 하나면 G1(인스턴스=보안경계)이 깨지고 L9 연합이 그룹 문제로 축소된다. §8 참조 |
| **GQ3** | **컨테이너 레지스트리가 켜져 있는가? metadata database 는 구성돼 있는가?** | 관리자 활성화가 필요하고, metadata DB 없이는 보호·불변 태그가 전부 불가 |
| **GQ4** | **사내 GitLab 버전에서 임의 OCI 아티팩트(ORAS, `artifactType`, 빈 config) push 가 되는가?** — 문서로 확인 불가, 실측 필요 | 되면 skill 번들을 컨테이너 레지스트리에 넣고 digest 불변성을 얻는다. 안 되면 generic package 로 가야 하고 불변성을 우리가 다시 만든다 |
| **GQ5** | **GitLab 을 IdP 로 쓸 것인가, 사내 IdP 를 그대로 쓰고 GitLab 은 저장소로만 쓸 것인가?** | 전자면 L5 가 통째로 사라지지만 사내 SSO 정책과 충돌 가능. RFC 8693 token exchange 지원 여부는 GitLab 문서에서 확인되지 않음 |
| **GQ6** | **GitLab 프로젝트/그룹 생성 권한을 각 팀이 갖는가, 플랫폼 팀이 통제하는가?** | 팀이 자유롭게 만들면 §6 의 `team_id` 네임스페이스 규율을 강제할 수 없다 |
| **GQ7** | **패키지 다운로드 계량을 위해 GitLab 앞단에 리버스 프록시를 둘 수 있는가?** | GitLab 이 다운로드 수를 세지 않으므로(§7) 이것이 설치형 텔레메트리의 유일한 경로 |
| **GQ8** | **오프라인 라이선스(cloud licensing opt-out)를 이미 받았는가?** | 폐쇄망 self-managed 운영과 보안 스캐너 오프라인 실행의 전제 |

---

## 확인하지 못한 것 (명시)

- **임의 OCI 아티팩트(`artifactType` + 빈 config manifest) 지원 여부.** docs.gitlab.com 에 명시가 없다. 공개 이슈 트래커에 진행 중 항목이 보이나 공식 문서가 아니므로 근거로 삼지 않았다.
- **Referrers API 의 정확한 미구현 범위.** 문서는 "does not fully implement" 라고만 한다.
- **cosign 자체 키 서명을 GitLab 레지스트리에 붙이는 공식 절차.** GitLab 서명 예제 문서는 keyless 만 다룬다. (기술적으로 `cosign sign --key` 는 OCI 레지스트리 일반에 동작하지만, GitLab 문서가 이를 보증하지는 않는다.)
- **GitLab↔GitLab 인스턴스 간 리포지터리 미러링의 공식 지원 여부.** 미러링 문서가 명시하지 않는다.
- **RFC 8693 token exchange 지원 여부.** GitLab 문서에서 확인되지 않았다.
- **Advanced search 의 Elasticsearch/OpenSearch 버전 요구사항.** 조사한 페이지에 버전 명시가 없었다(별도 관리자 문서 확인 필요).
- **패키지 레지스트리 전용 rate limit.** instance_limits 문서에 패키지 업로드 전용 rate limit 항목이 없다.
- **MinIO 의 공식 지원 여부.** 객체 스토리지 문서는 Amazon S3 / Google Cloud Storage / Azure Blob 만 "actively tests" 대상으로 명시하고, MinIO 는 지원·미지원 어느 쪽으로도 언급하지 않는다. "S3 호환 API 를 노출하는 온프레미스 어플라이언스" 범주에 들어가며 이 범주는 커뮤니티 문서화 상태다 — https://docs.gitlab.com/administration/object_storage/ . 또한 **컨테이너 레지스트리와 백업은 consolidated form 대상이 아니어서 별도 설정이 필요**하고, presigned URL 다운로드는 클라이언트가 객체 스토리지에 직접 접근해야 해서 폐쇄망 구성이 까다롭다는 점이 같은 문서에 명시돼 있다.

---

## 참고 자료 (References)

패키지·레지스트리
- https://docs.gitlab.com/user/packages/generic_packages/
- https://docs.gitlab.com/user/packages/package_registry/
- https://docs.gitlab.com/user/packages/package_registry/supported_functionality/
- https://docs.gitlab.com/api/packages/
- https://docs.gitlab.com/user/packages/container_registry/
- https://docs.gitlab.com/user/packages/container_registry/immutable_container_tags/
- https://docs.gitlab.com/user/packages/container_registry/protected_container_tags/
- https://docs.gitlab.com/api/container_registry/
- https://docs.gitlab.com/administration/packages/container_registry_metadata_database/
- https://docs.gitlab.com/user/packages/dependency_proxy/

이름·경로·API
- https://docs.gitlab.com/user/project/repository/
- https://docs.gitlab.com/user/group/manage/
- https://docs.gitlab.com/api/rest/

권한·인증
- https://docs.gitlab.com/user/permissions/
- https://docs.gitlab.com/user/project/deploy_tokens/
- https://docs.gitlab.com/user/group/settings/group_access_tokens/
- https://docs.gitlab.com/ci/jobs/ci_job_token/
- https://docs.gitlab.com/integration/openid_connect_provider/
- https://docs.gitlab.com/integration/saml/

검색·감사·이벤트
- https://docs.gitlab.com/user/search/
- https://docs.gitlab.com/user/search/advanced_search/
- https://docs.gitlab.com/user/compliance/audit_events/
- https://docs.gitlab.com/user/project/integrations/webhook_events/

연합·운영
- https://docs.gitlab.com/administration/geo/
- https://docs.gitlab.com/user/project/repository/mirror/
- https://docs.gitlab.com/user/project/settings/import_export/
- https://docs.gitlab.com/administration/instance_limits/
- https://docs.gitlab.com/administration/object_storage/

거버넌스·보안
- https://docs.gitlab.com/user/project/releases/
- https://docs.gitlab.com/user/project/merge_requests/approvals/
- https://docs.gitlab.com/user/project/codeowners/
- https://docs.gitlab.com/user/application_security/sast/
- https://docs.gitlab.com/user/application_security/secret_detection/
- https://docs.gitlab.com/user/application_security/offline_deployments/
- https://docs.gitlab.com/ci/yaml/signing_examples/

공식 문서가 아니어서 근거로 채택하지 않은 참고 링크 (임의 OCI 아티팩트 관련, 상태 확인용)
- https://gitlab.com/gitlab-org/container-registry/-/issues/967
- https://gitlab.com/gitlab-org/gitlab-foss/-/issues/43202
