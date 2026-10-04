---
tags: [platform, gitlab, verification, checklist]
level: intermediate
last_updated: 2026-09-03
reviewed_on: 2026-10-04
review_status: partial
document_type: specification
---

# 사내 GitLab 실측 체크리스트

> [!info] 2026-10-04 검토 범위
> 설계·양식 또는 별도 구현 저장소의 티켓이다. 현재 구현·사내 운영 증거로 간주하지 않는다. [현재 검토와 적용 조건](./review-notes.md)을 먼저 확인한다. 과거 완료 보고와 승인 전제는 당시 기록으로 유지하며, 이번 검증은 [정리 기록](./organization-log.md)에 구분한다.

> GitLab 을 배포 백엔드로 쓸 수 있는지 판정하는 데 필요한 4가지를 문서가 아니라 실제 인스턴스에서 확인하는 절차.

조사 결과([research/gitlab-as-distribution-backend.md](./research/gitlab-as-distribution-backend.md))로 대체 후보와 제약을 조사했다. 결론은 버전·티어·설정·사내 계약에 따라 달라진다. 남은 것은 **사내 GitLab 이 그 기능들을 실제로 갖고 있는가**이고, 이건 공식 문서로 답이 안 나온다. 아래 시간은 준비와 권한이 갖춰진 경우의 계획값이다. 설치·계정 발급·관리자 확인 시간은 포함하지 않는다.

---

## 왜 필요한가? (Why)

지금 모르는 것이 4개이고 그중 **2개는 아키텍처를 바꾼다.**

| 모르는 것 | 모르면 생기는 일 |
|---|---|
| **티어** | 심사 강제(MR 승인)·감사 이벤트·불변 태그를 쓸 수 있는지 모른다 → 거버넌스 설계가 붕 뜬다 |
| **컨테이너 레지스트리 활성 여부** | L1(아티팩트 저장) 후보가 하나 사라진다 |
| **임의 OCI 아티팩트 지원** | push 성공만으로 서명 검색·검증·삭제 방지까지 증명되지는 않는다 |
| **인스턴스 구성** | 전사 1대면 "인스턴스 경계 = 보안 경계" 전제가 GitLab 그룹 권한으로 내려온다 |

그렇다고 답을 기다리며 멈추지 않는다. §5 의 설계 대응으로 결정을 미룰 수 있게 해뒀다.

---

## 핵심 개념 (What)

### 1. "컨테이너 레지스트리"가 무엇이고 왜 묻는가

GitLab 은 아티팩트를 담는 방법을 **두 가지** 제공한다. 이름이 비슷해서 헷갈리지만 성질이 꽤 다르다.

| | **Package Registry (generic)** | **Container Registry** |
|---|---|---|
| 원래 용도 | 임의 파일 업로드·다운로드 | 도커 이미지 저장 |
| 주소 방식 | 이름 + 버전 (`my-skill/1.0.0/bundle.tar.gz`) | **내용 해시(digest)** |
| 같은 버전 재업로드 | **기본 허용** (파일이 추가됨). Owner 가 막을 수 있음 | digest 가 다르면 다른 것 |
| 삭제 | **Maintainer 가 지울 수 있음** | 태그 보호 가능. **불변 태그는 Ultimate** |
| 서명 부착 | 별도로 관리해야 함 | cosign 서명을 **함께 저장** 가능 |

**왜 컨테이너 레지스트리를 묻는가.** 우리가 필요한 건 "발행된 버전은 절대 안 바뀐다"는 보장인데, 패키지 레지스트리에는 그게 없다. 컨테이너 레지스트리는 **주소 자체가 내용 해시**라서 내용이 바뀌면 주소가 바뀐다 — 불변성이 구조에서 나온다.

그리고 최근 몇 년 사이 컨테이너 레지스트리는 도커 이미지만 담는 곳이 아니게 됐다. **임의의 파일 묶음(OCI artifact)** 을 담을 수 있는 규격이 생겼다. 그게 되면 우리 skill 번들도 여기 넣을 수 있고, 내용 해시 기반 주소 지정은 가능하지만 서명 도구·발견·검증·삭제 방지는 별도 확인해야 한다. **GQ4 가 묻는 게 이것이다.**

GitLab 문서는 OCI 이미지 포맷 지원과 cosign 의 `subject` 필드는 확인해주지만, **"OCI 1.1 Referrers API 를 완전히 구현하지는 않는다"** 고만 적혀 있고 이 문서의 번들 media type과 클라이언트 조합까지 보증하지는 않는다. 그래서 실측이 필요하다.

### 2. "사번으로 repo 를 발급받는다"가 알려주는 것

사용자 답변에 따르면 개인 사번용·시스템용으로 repo 를 발급받는 형태다. 여기서 추론되는 것:

- 사내 GitLab 은 **공용 서비스**다. 우리가 인스턴스를 소유하지 않는다.
- 우리가 받는 것은 **그룹 또는 프로젝트**이지 관리자 권한이 아니다.
- 따라서 **인스턴스 설정(티어, 컨테이너 레지스트리 활성화, 오브젝트 스토리지)은 우리가 못 바꾼다.** 관리자에게 물어야 한다.

이게 GQ2 의 실질적 답에 가깝다 — "사업부별 GitLab 인스턴스"는 아마 없고, **전사 1대를 그룹으로 나눠 쓰는 형태**일 가능성이 높다. 확인은 필요하다.

---

## 어떻게 확인하는가? (How)

### 3. 15분짜리 확인 — UI 만으로

GitLab 에 로그인해서 순서대로.

| # | 확인할 것 | 어디서 | 판정 |
|---|---|---|---|
| 1 | **티어** | 우측 상단 `Help` → `About GitLab`, 또는 `/help` 페이지 | edition·버전을 기록한다. EE 표기만으로 유료 티어를 확정하지 않고 관리자에게 활성 구독·만료를 확인한다 |
| 2 | **컨테이너 레지스트리 활성** | 아무 프로젝트 → 좌측 메뉴에 `Deploy` → `Container Registry` 항목이 **보이는가** | 보이면 해당 계정의 화면 접근을 확인한다. 안 보이면 `미확인`: 인스턴스·프로젝트 설정과 역할을 따로 확인한다 |
| 3 | **패키지 레지스트리 활성** | 같은 메뉴에 `Deploy` → `Package Registry` | 보이면 해당 계정의 화면 접근을 확인한다. 안 보이면 `미확인`: 인스턴스·프로젝트 설정과 역할을 따로 확인한다 |
| 4 | **MR 승인 규칙** | 프로젝트 → `Settings` → `Merge requests` → `Merge request approvals` 섹션에 **승인 규칙을 추가**할 수 있는가 | 사용 가능 여부만 기록한다. 없거나 잠겨 있으면 티어·역할·설정·만료를 확인하며 Free로 단정하지 않는다 |
| 5 | **감사 이벤트** | 그룹 → `Secure` 또는 `Settings` 에 `Audit events` | 해당 계정의 가용 여부를 기록한다. 메뉴 부재만으로 티어를 역추론하지 않는다 |
| 6 | **그룹 생성 권한** | 우리가 그룹을 직접 만들 수 있는가, 신청해야 하는가 | 네임스페이스 설계에 영향 |

UI는 기능 가용성을 관측하는 수단이다. 실제 라이선스 티어는 관리자 확인과 인스턴스 버전 기록을 기준으로 한다. 공식 기능 티어와 관측 결과를 대조한다.

### 4. 30분짜리 확인 — 실제로 밀어보기

**GQ4는 아래 버전·media type·인증 조합의 push/pull로 실측한다.** 다른 조합까지 지원한다고 일반화하지 않는다. 컨테이너 레지스트리가 켜져 있어야 한다(위 #2).

```bash
# 준비: ORAS CLI (폐쇄망이면 사내 미러 또는 반입 필요)
# 아무 테스트 프로젝트에서

# 1. 버전 기록 후 대화형 로그인 (PAT를 명령 인자나 문서에 넣지 않는다)
oras version
oras login <사내-gitlab-registry-호스트>
# 프롬프트에 사용자명과 테스트용 자격증명을 입력한다. 자동화는 --password-stdin 사용.

# 2. 더미 번들 만들기
mkdir -p /tmp/skilltest && echo "test" > /tmp/skilltest/SKILL.md
tar czf /tmp/bundle.tar.gz -C /tmp/skilltest .

# 3. 임의 artifactType 으로 push — 여기서 갈린다
oras push <호스트>/<그룹>/<프로젝트>:0.0.1 \
  --artifact-type application/vnd.skhynix.skill.v1+json \
  /tmp/bundle.tar.gz:application/vnd.skhynix.skill.layer.v1.tar+gzip

# 4. 되돌려 받기
oras pull <호스트>/<그룹>/<프로젝트>:0.0.1 -o /tmp/pulled

# 5. digest 로 주소 지정이 되는가
oras manifest fetch <호스트>/<그룹>/<프로젝트>:0.0.1 --descriptor
```

| 결과 | 판정 | 다음 |
|---|---|---|
| 3~5 전부 성공 | 해당 조합의 저장·조회·digest 확인 성공. 서명·삭제 방지·태그 갱신 정책은 미검증 | 서명·불변 규칙·권한을 별도 실측한 뒤 L1 채택 판단 |
| 3 실패 (`unsupported media type` 등) | 해당 조합에서 실패. 권한·클라이언트·서버 버전·media type을 분리 진단 | generic package registry 또는 MinIO |
| 2번(레지스트리) 자체가 없음 | | C안 검토 |

추가로 확인할 것:

- 내려받은 파일의 SHA-256을 원본과 비교하고 descriptor의 digest로 다시 조회한다. 태그가 새 digest를 가리켜도 기존 digest 주소는 별개다.
- 서명 생성·검색·검증은 선택한 cosign 버전과 사내 키/신뢰 정책으로 따로 시험한다. GitLab은 OCI 1.1 `subject`를 지원하지만 Referrers API 전체 구현을 보증하지 않는다.
- 태그 재push·삭제 권한·cleanup 영향을 폐기 가능한 시험 자산에서 확인한다. 불변 태그는 Ultimate, 18.10 GA 기준이며 Self-Managed metadata DB와 Owner 설정 권한이 필요하다. digest 확인은 삭제 방지 시험을 대체하지 않는다.
- 아래 업로드는 사내 실측용 템플릿이다. 실제 PAT는 보안 자격증명 공급 방식으로 전달하고 명령 이력·로그에 넣지 않는다. 이번 정리에서는 업로드를 실행하지 않았다.

```bash
# 같은 버전을 다시 밀면 어떻게 되는가 (불변성 실측)
# generic package 쪽
curl --header "PRIVATE-TOKEN: <token>" --upload-file /tmp/bundle.tar.gz \
  "https://<호스트>/api/v4/projects/<id>/packages/generic/test/0.0.1/bundle.tar.gz"
# 두 번 실행해서 두 번째가 거부되는지, 덮어써지는지, 파일이 추가되는지 본다
```

### 5. 관리자에게 물을 것 (우리가 못 보는 것)

| 질문 | 왜 |
|---|---|
| 라이선스 티어와 만료 | 거버넌스 기능 가용성 |
| 컨테이너 레지스트리 **metadata DB** 구성 여부 | 불변 태그(Ultimate)의 전제 조건 |
| 패키지·레지스트리 **저장 용량 한도**와 프로젝트별 상한 | 번들 크기 정책 |
| 오브젝트 스토리지 백엔드가 무엇인지 | 조사에서 **MinIO 지원 여부는 확인도 부정도 못 했다** |
| GitLab 을 **OIDC provider** 로 쓸 수 있는가 (앱 등록 허용 여부) | L5 를 GitLab 에 위임할지 |
| 사내 IdP 와 GitLab 의 관계 (GitLab 이 SSO 뒤에 있는가) | 인증 토폴로지 |
| **인스턴스가 몇 대인가** | GQ2 |

---

## 결정을 미루기 위한 설계 대응

답을 기다리는 동안 개발이 멈추면 안 된다. **아티팩트 저장을 좁은 인터페이스 하나 뒤에 둔다.**

```python
# 계약은 이것뿐. 프레임워크를 만들지 않는다.
from typing import BinaryIO, Protocol

class ArtifactStore(Protocol):
    def put(self, key: str, data: BinaryIO) -> str: ...      # 반환: digest 또는 버전 식별자
    def get_url(self, key: str, ttl: int) -> str: ...        # 만료되는 다운로드 URL
    def exists(self, key: str) -> bool: ...
```

| 구현 | 언제 |
|---|---|
| `MinioArtifactStore` | 기본. 지금 바로 만든다. Object Lock GOVERNANCE |
| `GitlabPackageStore` | GitLab 도입 확정 시. generic package API |
| `OciArtifactStore` | GQ4 성공과 별도 서명·보존·권한 계약 검증 뒤 검토 |

**이 seam 하나면 충분하다.** 인증·카탈로그·검색까지 추상화하지 않는다 — 그건 결정이 안 미뤄지는 영역이고, 구현이 하나뿐인 인터페이스는 부채다.

---

## 확인 결과 기록

| 항목 | 결과 | 확인일 | 확인자 |
|---|---|---|---|
| GQ1 티어 | | | |
| GQ2 인스턴스 구성 | | | |
| GQ3 컨테이너 레지스트리 활성 | | | |
| GQ4 임의 OCI 아티팩트 | | | |
| generic package 중복 발행 동작 | | | |
| MR 승인 규칙 사용 가능 | | | |
| 감사 이벤트 사용 가능 | | | |
| 오브젝트 스토리지 백엔드 | | | |

---

## 참고 자료

- [GitLab 조사 문서](./research/gitlab-as-distribution-backend.md) — 계층 대체 판정표, 토폴로지 3안, 티어별 기능
- [아키텍처 설계서](./02-architecture.md)


## 현재 확인한 공식 근거

확인일: **2026-10-04**. 사내 GitLab·ORAS·cosign 설치 버전과 시험 결과는 미확인이다.

- [ORAS 1.3 login](https://oras.land/docs/commands/oras_login/): 대화형 로그인과 `--password-stdin`을 지원한다.
- [GitLab edition·license 확인](https://docs.gitlab.com/administration/license/): edition/version 표시와 활성 구독을 구분한다.
- [GitLab Container Registry](https://docs.gitlab.com/user/packages/container_registry/): 관리자 활성화, `subject` 지원과 Referrers API 제한.
- [Immutable container tags](https://docs.gitlab.com/user/packages/container_registry/immutable_container_tags/): Ultimate, 18.10 GA, Self-Managed metadata DB 조건.
