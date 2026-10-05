---
tags: [platform, verification]
aliases: [플랫폼 현재 적용 조건]
reviewed_on: 2026-10-04
review_status: partial
document_type: technical_review
category_major: "Agent·Skill 플랫폼"
category_middle: "플랫폼 운영"
category_minor: "적용 조건"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# 플랫폼 현재 검토와 적용 조건

이 문서는 설계·조사·티켓을 실제 도입에 적용할 때 필요한 근거와 미확인을 모은다. 원본 문서의 개별 검토는 마쳤으며, 협의 대기와 외부 실행 미확인은 남아 있다. [정리 기록](./organization-log.md)의 문서별 상태가 검토 범위의 기준이다. 기존 `last_updated`는 원 작성·수정일로 보존하고 `reviewed_on`과 구분한다.

> [!important] 설계와 실행 증거
> `pm_notes/platform`에는 설계 문서와 과거 구현 완료 보고가 있다. 티켓의 `src/registry`, `tests`는 별도 구현 저장소 경로다. 이번 정리에서 해당 앱·사내 IdP·GitLab·MinIO를 실행한 증거는 없다. 과거 “전체 통과”를 현재 재검증으로 읽지 않는다.

## 문서의 단일 출처와 중복 판단

요구사항은 01, 기술 구조·D#는 02, 역할·승인 프로세스는 03, 비용·일정 가정은 04, 작성 양식은 05, 인스턴스 실측은 06이 담당한다. 유사 설명이 각 문서의 목적에 필요한 경우 유지한다. 현재 기술 조건은 이 문서에 모아 링크로 안내한다. 조사 기록·승인 결정의 본문을 통합·삭제하려면 Claude와 문맥을 대조해야 하므로 현재 보류한다.

## GitLab 도입 확인

[실측 체크리스트](./06-gitlab-check.md)는 메뉴 부재를 티어·인스턴스 비활성으로 단정하지 않도록 수정했다. edition, 활성 구독, 버전, 계정 역할, 프로젝트·인스턴스 설정을 함께 기록해야 한다.

공식 확인일 **2026-10-04**: GitLab의 [Container Registry](https://docs.gitlab.com/user/packages/container_registry/)는 OCI 1.1 `subject`를 지원하지만 Referrers API를 완전히 구현하지 않는다. [불변 태그](https://docs.gitlab.com/user/packages/container_registry/immutable_container_tags/)는 Ultimate, 18.10 GA이고 Self-Managed에서는 metadata DB가 필요하다. ORAS push/pull 성공은 해당 media type 조합의 저장·조회만 확인한다. 서명·검증·삭제 방지는 각각 별도 시험 대상이다. [ORAS 1.3 login](https://oras.land/docs/commands/oras_login/)은 대화형 입력과 `--password-stdin`을 제공한다. 사내 버전은 미확인이다.

## 신원 식별자와 public sub

[OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html#SubjectIDTypes)에서 public subject는 **같은 issuer**의 클라이언트들에 동일하게 제공된다. [안정적인 식별 조합](https://openid.net/specs/openid-connect-core-1_0.html#ClaimStability)은 `iss`와 `sub`다. 서로 다른 issuer의 sub가 같은 사번임을 보증하지 않는다. 사내 public sub·사번 매핑은 IdP 계약 확인 대기다. 04의 “신원 리스크 제거” 표현을 조건부로 수정했다. 확인일: 2026-10-04.

[RFC 9068 §4](https://www.rfc-editor.org/rfc/rfc9068.html#section-4)는 해당 JWT access-token profile을 채택할 때 `typ`, issuer, audience, 서명 등의 검증을 요구한다. OIDC ID token과 API access token은 용도가 다르다. `PLAT-L5-001`의 sub→empno 직접 추출·검증 범위는 사내 프로필과 추가 대조가 필요하며 아직 계약을 변경하지 않았다.

## MongoDB 드라이버

[Motor 3.7.1 공식 안내](https://motor.readthedocs.io/en/stable/)를 2026-10-04 확인했다. Motor는 2026-05-14 deprecated 상태이며 중요 버그 수정은 2027-05-14까지 제공한다고 안내한다. 신규 개발은 PyMongo Async로 이동하도록 권고한다. `tickets/CONVENTIONS.md`와 `PLAT-L2-002`의 Motor는 당시 설계 선택으로 보존하고 도입 전 재검토를 표시했다. 승인된 의존성 계약을 바꾸는 결정은 Claude 협의가 연결될 때까지 보류한다. 실제 PyMongo·MongoDB 호환 버전과 마이그레이션은 미검증이다.

## pytest 마커와 실행 대상

[공식 마커 문서](https://docs.pytest.org/en/stable/example/markers.html)의 `-m`은 수집 뒤 선택/선택 해제한다. “integration을 실행하지 않음”은 “모듈도 수집하지 않음”과 다르다. `PLAT-L0-001`의 AC-3와 검증 표를 일치시켰다. 확인일: 2026-10-04. 전체 구현 티켓의 pytest 명령은 별도 저장소가 없으므로 실행하지 않았다.

## 운영 상태와 등급

02·03과 `PLAT-L2-003`, `PLAT-L6B-003`은 헬스 장애 시 `degraded`와 신규 접근 차단을 자동 적용하되 심사 등급(tier)은 유지한다. 04 리스크 표의 “자동 강등”을 이 계약에 맞췄다. 기간·SLA·6.5 FTE는 설계 제안값이며 운영 실적·외부 표준값이 아니다.

## Claude 협의가 필요한 보류

- Herdr 환경 `HERDR_ENV=1`에서 `herdr pane current --current`는 `pane_not_found`다. 작업 전용 Claude pane이 확인되지 않아 의견을 받지 못했다. 다른 프로젝트 pane은 제어하지 않았다.
- MongoDB 신규 드라이버 전환: 권고는 PyMongo Async이나 승인 계약 변경은 보류.
- `PLAT-L2-007` AC-3의 50% 분산을 AC-7에서 임계값 60%로 낮추면 code_changed가 된다고 설명한다. 같은 겹침 정의라면 50<60이라 성립하지 않는다. 겹침의 분모와 동률·분할 우선순위·새 임계값을 Claude와 정해야 하므로 티켓 본문 변경은 보류.
- MinIO Object Lock의 신규 버전·같은 키 재발행·삭제 권한, 표준 server.json 확장의 클라이언트 호환성, OTel semantic convention 적용 범위는 추가 공식 근거 검토 중이다.

상기 보류는 해결된 설계 결정이 아니다. 아래 재검토 항목 밖의 조사 기록에 있는 변동 가능한 제품 수치·기능·플래그는 2026-10-04 현재 재검증하지 못했으며, 현행 도입 근거로는 미확인이다.

## Object Lock과 발행 불변성

[MinIO AIStor 공식 문서](https://docs.min.io/aistor/administration/object-locking-and-immutability/) 확인일: **2026-10-04**. 잠금은 객체 버전별이다. 잠긴 버전의 삭제·덮어쓰기를 막지만, version ID 없는 삭제는 delete marker를 만들 수 있다. GOVERNANCE는 우회 권한 조건이 있고 COMPLIANCE도 카탈로그·다운로드 접근 정책을 대신하지 않는다. 버킷 생성 시 설정은 가능한 경로이며 AIStor `RELEASE.2025-05-20T20-30-00Z` 이후 기존 버킷 설정 경로도 있다. 이 문서는 AIStor 기준으로, 사내 Community/AIStor 버전은 미확인이다.

목적은 심사한 **정확한 bytes**가 나중에도 내려받아지는 것이다. 적용 시 `(name, version)` 재발행 거부, 저장 객체 version/digest와 카탈로그 연결, SHA-256 확인, 우회·삭제 권한을 각각 검증한다. `exists` 후 `put` 순서만으로 동시 재발행 거부를 입증할 수 없다. 이 보장은 Object Lock 설정 1개만으로 완성되지 않는다. PLAT-L1-002의 계약과 버전 지정 방식은 별도 구현에서 검증해야 한다.

## 인가 모델의 정확한 비교

[NIST RBAC 자료](https://csrc.nist.gov/projects/role-based-access-control)는 사용자·역할뿐 아니라 권한·operation·object도 모델 요소로 둔다. 따라서 “RBAC은 객체 단위 권한을 표현할 수 없다”는 당시 비교는 지나친 단순화다. 관계 모델의 장점은 특정 객체의 owner와 그룹 userset·관계 파생을 직접 표현하는 데 있으며 RBAC과 함께 사용할 수도 있다. §4.1의 “필요 없다”는 외부 HR 조직 코드에 대한 설명으로 읽어야 한다. 내부 그룹을 참조할 안정적인 식별자는 여전히 필요하다. 확인일: 2026-10-04.

[SpiceDB datastores](https://authzed.com/docs/spicedb/concepts/datastores)는 테스트용 memory와 영속 저장소를 구분한다. memory는 종료 시 데이터가 사라지고 여러 인스턴스가 공유하지 못한다. “세 엔진 모두 반드시 새 관계형 DB”라는 일반화는 테스트 모드까지 포함하면 맞지 않는다. 실제 운영 저장소·버전·지원 범위를 검토해야 한다. 기존 D15의 MongoDB 선택은 이 사실 정정으로 변경하지 않았다.

## 권한 회수와 일관성

[Zanzibar 2019 논문 §2.2](https://www.usenix.org/system/files/atc19-pang.pdf)은 ACL 변경의 순서를 무시하거나 새 콘텐츠에 오래된 ACL을 적용하는 것을 new enemy 문제로 설명한다. 다지역 구성만의 현상이 아니다. 단일 DB여도 앱이 이전 허용 결과를 캐시하거나 권한 판정과 콘텐츠 읽기의 순서를 잘못 연결하면 문제가 남는다.

[MongoDB 인과 일관성 문서](https://www.mongodb.com/docs/manual/core/causal-consistency-read-write-concerns/)는 세션과 read/write concern 조합별 보장을 구분하며, 내구성을 포함한 모든 인과 일관성 보장은 causally consistent session에서 majority 읽기·쓰기를 조건으로 설명한다. primary 읽기만으로 이 조건을 대체하지 않는다. 1일 조직 배치는 조직 원천의 허용 지연이고 즉시 회수·인가 캐시 무효화 요구와는 별도다. zookie를 반드시 도입한다는 결론은 아니며, 캐시·동시 변경·failover의 회수 계약은 구현 전 협의 대기다. 확인일: 2026-10-04. 사내 MongoDB 구성은 미확인이다.

[Google IAM 전파 안내](https://docs.cloud.google.com/iam/docs/access-change-propagation)의 정책 변경은 보통 2분, 7분 이상도 가능하고 그룹 변경은 수 시간 이상도 가능하다. 이를 “최대 7분/몇 시간”이나 우리 앱의 SLA로 옮기면 안 된다. 멤버십 직접 조회 역시 원천 자체의 지연을 0으로 만들지 않는다. 확인일: 2026-10-04.

## 규격 버전과 계측 의미

[MCP 인가 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)은 Client ID Metadata Documents를 SHOULD, Dynamic Client Registration을 MAY로 두며 DCR은 이전 호환용 deprecated로 설명한다. 2025-06-18 조사의 RFC 나열을 모두 같은 강도의 현행 필수 구현으로 읽지 않는다. 채택 버전과 클라이언트·AS 기능을 먼저 합의한다. OAuth AS와 OIDC 로그인 지원도 구분한다.

[MCP Registry draft schema](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/draft/server.schema.json)는 `registryType`을 string과 예시로 정의한다. 사내 `internal-minio`가 스키마 검사에 들어갈 수 있다는 사실은 일반 클라이언트가 그 다운로드 방식을 이해한다는 증거가 아니다. draft와 배포용 날짜 스키마를 구분하고 표준 어댑터의 실제 계약을 검증한다.

[OTel GenAI 속성 원본](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md)에서 `gen_ai.agent.id`는 **Development**이며 hosted agent resource의 안정적인 ID를 설명한다. 설치된 Skill의 설치 건수를 자동으로 같은 에이전트 실행 지표로 바꿀 수 없다. `PLAT-L8-001`의 모든 이벤트 매핑은 사내 확장인지 표준 의미인지 합의해야 한다. 확인일: 2026-10-04, 위 main/draft는 조회 당시 개발 원본이며 사내 설치 버전은 미확인이다.

## 서명과 폐쇄망

[Sigstore 자체 키 서명](https://docs.sigstore.dev/cosign/key_management/signing_with_self-managed_keys/)은 키를 직접 관리하는 경로다. 자체 키나 `--upload=false`가 모든 외부 호출을 제거한다는 뜻은 아니다. [Cosign 개발 원본의 sign 옵션](https://github.com/sigstore/cosign/blob/main/cmd/cosign/cli/options/sign.go)은 서명 업로드와 transparency-log 업로드를 별개로 두고 signing-config도 다룬다. 설치 버전의 도움말·사내 trust root·log·TUF 설정을 확인한 뒤 외부 연결이 없는 환경에서 서명·검증을 따로 시험해야 한다. 이 기록의 단순 명령을 그대로 폐쇄망 검증 완료로 취급하지 않는다. 확인일: 2026-10-04. Cosign 릴리스/설치 버전과 실행 결과는 미확인이다.

## 티켓 실행 순서와 남은 계약

개별 티켓 30개의 선행 ID는 모두 존재하고 순환하지 않는다. 기존 README의 wave 3에 선행 L2-004와 소비자 L3-003·L6-002/003·L6B-001이 함께 있었다. 선행 완료가 이전 wave에 있도록 재계산했다. 같은 wave라도 공유 경로·계약 검토 뒤 배정한다. 실제 동시 구현을 실행하지 않았다.

`PLAT-L2-003`의 모든 버전 삭제 시 isLatest 의미, `PLAT-L2-004`의 조회 2회에 팀 승계 조회가 포함되는지, `PLAT-L2-006/007`의 코드 변경 시 동일 팀 추적, `PLAT-L5-001`의 typ·nbf·사번 매핑, `PLAT-L8-003`의 누락률 분모는 아직 구현 근거가 없다. 기존 수용 기준과 고유 예제를 유지하고 Claude 협의가 연결되면 이 계약들을 먼저 정할 것을 권고한다. “미확인”을 false·0·성공으로 대체하지 않는다.
