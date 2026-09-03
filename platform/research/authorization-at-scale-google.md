---
tags: [platform, authorization, zanzibar, iam, rebac, scim]
level: advanced
last_updated: 2026-09-03
---

# 대규모 인가(Authorization) 1차 자료 조사 — Google 은 어떻게 수만 명에게 권한을 주는가

> Zanzibar 논문과 Google Cloud IAM 공식 문서를 1차 출처로 삼아 "팀명이 매년 바뀌고 불변 조직 코드가 있는지조차 확인 불가"인 우리 레지스트리의 인가 모델을 결정하기 위한 근거를 정리한다.

**이 문서의 규칙**: 모든 사실 주장에 1차 출처 URL 을 인라인으로 붙였다. 1차 자료에서 확인하지 못한 것은 "확인 불가" 라고 명시했고 추정으로 메우지 않았다.

**조사 시점의 전제 변화**: 사원 마스터에 불변 조직 코드가 있는지 없는지 확인 불가다. 확실한 것은 `empno`(사번)가 불변이라는 것뿐이다. 따라서 §4 의 권고 모델은 조직 코드의 존재 여부에 대해 불변(invariant)이어야 한다는 제약을 추가로 받는다.

---

## 왜 필요한가? (Why)

[02-architecture.md](../02-architecture.md) 의 §5.2 는 소유권을 관계 튜플(`resource, relation, subject`)로 두기로 했고 §5.3 은 불변 `teamId` + 가변 `displayName` 분리를 두었으며 §6 은 `team_id` 발급 주체를 미결(Q2)로 남겼다. 그런데 Q2 의 전제가 흔들렸다. 사원 마스터에 불변 조직 코드가 있는지 자체를 지금 확인할 수 없다.

그래서 답해야 할 질문은 "조직 코드가 있느냐"가 아니라 **"조직 코드의 유무와 무관하게 성립하는 인가 모델이 존재하느냐"** 로 바뀐다. Google 이 수십억 사용자에게 권한을 부여하는 방식은 이 질문에 직접적인 선례를 준다. Google 역시 사람의 소속을 리소스 정책에 박지 않기 때문이다.

동시에 반대 방향의 위험도 있다. Zanzibar 는 조 단위 튜플을 다루는 시스템이고 우리는 수천 명 규모다. 어디까지가 배울 것이고 어디부터가 과설계인지를 논문이 명시한 숫자로 가르는 것이 이 조사의 두 번째 목적이다.

---

## 핵심 개념 (What)

### 1. Zanzibar — 관계 튜플 모델의 원전

출처: [Zanzibar: Google's Consistent, Global Authorization System (USENIX ATC '19)](https://www.usenix.org/system/files/atc19-pang.pdf) · [초록/서지](https://research.google/pubs/zanzibar-googles-consistent-global-authorization-system/)

#### 1.1 관계 튜플의 정확한 형태

논문 §2.1 이 제시하는 문법은 다음과 같다 ([논문 §2.1](https://www.usenix.org/system/files/atc19-pang.pdf)).

```
⟨tuple⟩   ::= ⟨object⟩ '#' ⟨relation⟩ '@' ⟨user⟩
⟨object⟩  ::= ⟨namespace⟩ ':' ⟨object id⟩
⟨user⟩    ::= ⟨user id⟩ | ⟨userset⟩
⟨userset⟩ ::= ⟨object⟩ '#' ⟨relation⟩
```

- 튜플을 식별하는 기본키는 `⟨namespace⟩, ⟨object id⟩, ⟨relation⟩, ⟨user⟩` 네 개다.
- `⟨namespace⟩` 와 `⟨relation⟩` 은 **클라이언트 설정(namespace config)에 사전 정의**되고, `⟨object id⟩` 는 문자열, `⟨user id⟩` 는 정수다.
- **`⟨userset⟩` 이 있다는 것이 핵심이다.** 논문: *"One feature worth noting is that a ⟨userset⟩ allows ACLs to refer to groups and thus supports representing nested group membership."*

논문 Table 1 의 예시가 우리 관심사와 정확히 겹친다.

| 튜플 | 의미 |
|---|---|
| `doc:readme#owner@10` | 사용자 10 은 `doc:readme` 의 owner |
| `group:eng#member@11` | 사용자 11 은 `group:eng` 의 member |
| `doc:readme#viewer@group:eng#member` | `group:eng` 의 **멤버들이** `doc:readme` 의 viewer |
| `doc:readme#parent@folder:A#...` | `doc:readme` 는 `folder:A` 안에 있다 |

그룹은 별도의 개념이 아니다. 논문 §2.1: *"In Zanzibar, ACLs are collections of object-user or object-object relations represented as relation tuples. **Groups are simply ACLs with membership semantics.**"* 즉 `group:eng#member@11` 은 다른 어떤 튜플과도 구조가 같다. 그룹 테이블이 따로 없다.

논문이 밝힌 설계 이유: *"Defining our data model around tuples, instead of per-object ACLs, allows us to unify the concepts of ACLs and groups and to support efficient reads and incremental updates."*

#### 1.2 Userset rewrite rules — 저장하지 않고 파생시키는 규칙

관계 튜플만으로는 "editor 인 사람은 자동으로 viewer" 같은 객체 무관(object-agnostic) 규칙을 표현할 수 없다. 논문 §2.3.1: *"While such relationships between relations can be represented by a relation tuple per object, storing a tuple for each object in a namespace would be wasteful and make it hard to make modifications across all objects."* 그래서 namespace config 에 **userset rewrite rule** 을 둔다. 규칙은 object ID 를 입력받아 userset 표현식 트리를 출력하는 함수이며 리프 노드는 셋 중 하나다 ([논문 §2.3.1](https://www.usenix.org/system/files/atc19-pang.pdf)).

| 리프 | 하는 일 |
|---|---|
| `_this` | 저장된 튜플에서 해당 `⟨object#relation⟩` 의 모든 사용자를 반환 (간접 ACL 포함). rewrite 규칙이 없을 때의 기본 동작 |
| `computed_userset` | 같은 객체의 **다른 relation** 으로 userset 을 계산. `viewer` 가 `editor` 를 포함하게 하는 상속 |
| `tuple_to_userset` | 입력 객체에서 tupleset 을 계산해 튜플을 가져오고, **가져온 튜플마다** userset 을 계산. 논문 표현: *"look up the parent folder of the document and inherit its viewers"* |

표현식은 union / intersection / exclusion 으로 조합된다. 논문 Figure 1 의 실제 config:

```protobuf
name: "doc"
relation { name: "owner" }
relation { name: "editor"
  userset_rewrite { union {
    child { _this {} }
    child { computed_userset { relation: "owner" } } } } }
relation { name: "viewer"
  userset_rewrite { union {
    child { _this {} }
    child { computed_userset { relation: "editor" } }
    child { tuple_to_userset {
      tupleset { relation: "parent" }
      computed_userset { object: $TUPLE_USERSET_OBJECT  # 부모 폴더
                         relation: "viewer" } } } } } }
```

RBAC 는 `(주체, 역할)` 만 표현한다. Zanzibar 는 `(주체, 역할, **어느 객체에 대해**)` 를 표현하고, 나아가 역할의 주체 자리에 다른 객체의 역할집합을 넣을 수 있다(`@group:eng#member`, `@folder:A#viewer`). "이 에이전트의 owner", "이 팀 멤버 전원이 이 에이전트의 maintainer" 는 RBAC 로는 역할 이름을 리소스마다 폭발적으로 늘려야만 표현되고, ReBAC 에서는 튜플 1건이다. **이것이 평범한 RBAC 가 못 하는 지점이다.**

#### 1.3 간접·중첩 그룹 멤버십의 평가와 비용

체크의 정의는 재귀적이다 ([논문 §3.2.3](https://www.usenix.org/system/files/atc19-pang.pdf)):

```
CHECK(U, ⟨object#relation⟩) =
    ∃ tuple ⟨object#relation@U⟩
  ∨ ∃ tuple ⟨object#relation@U'⟩, where U' = ⟨object'#relation'⟩ s.t. CHECK(U, U')
```

논문이 그 비용을 직접 인정한다: *"Finding a valid U' ... involves evaluating membership on all indirect ACLs or groups, recursively. This kind of 'pointer chasing' works well for most types of ACLs and groups, but **can be expensive when indirect ACLs or groups are deep or wide.**"* 지연을 줄이려고 boolean 표현식 트리의 모든 리프를 동시 평가한다.

깊고 넓은 그룹 중첩을 위해 Zanzibar 는 **Leopard** 라는 전용 인덱스를 따로 만든다 ([논문 §3.2.4](https://www.usenix.org/system/files/atc19-pang.pdf)). Leopard 는 그룹 그래프의 도달가능성 문제로 환원해 `GROUP2GROUP` 을 비정규화(flatten)하고 멤버십 판정을 집합 교집합으로 바꾼다:

```
U 가 G 의 멤버인가  ⟺  (MEMBER2GROUP(U) ∩ GROUP2GROUP(G)) ≠ ∅
```

인덱스 튜플은 skip list 같은 정렬된 정수 리스트로 저장되어 교집합이 `O(min(|A|,|B|))` seek 로 끝난다. 오프라인 인덱스 빌더가 스냅샷에서 shard 를 만들어 전역 복제한다. 일관성을 위해서는 오프라인 스냅샷 이후의 변경분을 담는 incremental layer 를 Watch API 로 실시간 유지한다.

**우리 규모 판단에 직접 쓰일 숫자**: 논문은 *"a single Zanzibar tuple addition or deletion may yield potentially **tens of thousands** of discrete Leopard tuple events"* 라고 적는다. 즉 비정규화 인덱스는 쓰기 증폭이 수만 배다. 이 비용을 감수할 이유는 그룹 중첩이 깊고 넓을 때뿐이다.

#### 1.4 Zookie 와 일관성 — "new enemy" 문제

논문 §2.2 가 정의하는 회피 대상은 "new enemy" problem 이다. 두 시나리오를 그대로 옮긴다 ([논문 §2.2](https://www.usenix.org/system/files/atc19-pang.pdf)).

- **Example A (ACL 갱신 순서 무시)**: Alice 가 폴더 ACL 에서 Bob 을 제거 → Alice 가 Charlie 에게 새 문서를 그 폴더로 옮기라고 요청(문서 ACL 은 폴더에서 상속) → **Bob 이 새 문서를 보면 안 되는데**, 두 ACL 변경 간 순서를 무시하면 볼 수 있다.
- **Example B (새 내용에 낡은 ACL 적용)**: Alice 가 문서 ACL 에서 Bob 을 제거 → Alice 가 Charlie 에게 새 내용을 추가하라고 요청 → **Bob 이 새 내용을 보면 안 되는데**, 제거 이전의 낡은 ACL 로 판정하면 볼 수 있다.

이를 막으려면 두 성질이 필요하다: external consistency 와 bounded staleness 스냅샷 읽기. Zanzibar 는 ACL 을 [Spanner](https://www.usenix.org/system/files/atc19-pang.pdf) 에 저장해 TrueTime 이 마이크로초 해상도 타임스탬프를 부여하게 하고 모든 체크를 단일 스냅샷 타임스탬프에서 평가한다.

**Zookie 는 그 위의 프로토콜이다.** 논문 §2.4: *"A zookie is an opaque byte sequence encoding a globally meaningful timestamp that reflects an ACL write, a client content version, or a read snapshot."*

동작:
1. 클라이언트가 **콘텐츠 버전마다** content-change check 를 통해 zookie 를 받아, 콘텐츠 변경과 **원자적으로 함께 저장**한다. Zanzibar 는 zookie 에 현재 전역 타임스탬프를 인코딩하고 **선행 ACL 쓰기가 모두 그보다 낮은 타임스탬프를 갖도록** 보장한다.
2. 이후 체크 요청에 그 zookie 를 실어 보내면, **평가 스냅샷이 최소한 그 콘텐츠 버전만큼은 신선함**이 보장된다.

zookie 가 주는 보장은 "at-least-as-fresh" 이지 "latest" 가 아니다. 그리고 그 느슨함이 성능의 원천이다. 논문: *"Such freedom in turn allows Zanzibar to serve most checks at a default staleness with already replicated data."* 최신 스냅샷을 매번 강제하면 *"global data synchronization with high-latency round trips and limited availability"* 가 되기 때문이다. 불투명한 쿠키로 만든 이유도 명시돼 있다. 클라이언트가 임의 타임스탬프를 고르는 것을 막고 향후 확장 여지를 남기기 위해서다.

#### 1.5 논문이 실제로 밝힌 규모 숫자

과설계 판단의 기준선이므로 논문 §4 의 수치를 그대로 옮긴다 ([논문 §4](https://www.usenix.org/system/files/atc19-pang.pdf)).

| 항목 | 값 |
|---|---|
| 네임스페이스 수 | **1,500+** (수백 개 클라이언트 앱이 정의) |
| namespace config 크기 | 수십 줄 ~ 수천 줄, **중앙값 500줄** |
| 관계 튜플 총량 | **2조(2 trillion) 이상**, 약 **100 TB** |
| **네임스페이스당 튜플 수** | 수십 ~ 1조, **중앙값 약 15,000** |
| 복제 | 전 세계 **30곳 이상**에 완전 복제 |
| 총 QPS | **1,000만+** client QPS |
| 세부 QPS (2018-12, 7일 표본 피크) | Check **4.2M**, Read **8.2M**, Expand **760K**, Write **25K** |
| 읽기:쓰기 비 | 읽기가 쓰기보다 **2자릿수(2 orders of magnitude)** 많음 |
| 서버 | **10,000+** 대, 수십 개 클러스터. 클러스터당 100 미만 ~ 1,000 초과, 중앙값 500 |
| Check Safe 지연 (p50/p95/p99/p99.9) | 약 **3 / 11 / 20 / 93 ms** (7일 피크) |
| 가용성 | **99.999% 초과**, 3년간. 분기당 전역 다운타임 2분 미만 |
| 내부 위임 RPC | 피크 **2,200만/초** |
| 인메모리 캐시 조회 | 피크 **약 2억/초** |
| Leopard 인덱스 QPS | 중앙값 **1.56M**, p99 **2.22M** |

여기서 가장 중요한 한 줄은 "네임스페이스당 튜플 중앙값 15,000" 이다. 조 단위 총량은 1,500개 네임스페이스의 합이고 전형적인 네임스페이스 하나는 1.5만 튜플짜리다. 우리 레지스트리 전체가 그 "중앙값 네임스페이스" 한 개보다 작다(§5.4).

#### 1.6 Zanzibar 가 RBAC 대비 실제로 푸는 것

논문에 근거해 정리하면 세 가지다.

1. **리소스 단위 관계** — `⟨object⟩#⟨relation⟩@⟨user⟩` 는 주체·역할·**객체**를 한 튜플에 담는다. RBAC 의 `(주체, 역할)` 은 "어느 에이전트의 owner 인가"를 표현하지 못한다.
2. **주체 자리에 집합을 넣을 수 있음(userset)** — `@group:eng#member` 로 그룹 전체를 튜플 1건으로 부여하고, 그룹이 그룹을 포함하는 중첩까지 같은 문법으로 표현한다. 그룹은 별도 개념이 아니다.
3. **관계 간 파생 규칙(userset rewrite)** — "owner ⊂ editor ⊂ viewer", "부모 폴더의 viewer 를 상속" 을 튜플을 저장하지 않고 config 로 표현한다. `tuple_to_userset` 이 계층 상속을 **hop 당 튜플 1건**으로 만든다.

반대로 Zanzibar 가 풀지 않는 것도 있다. 속성 기반 조건(시간, IP, 요청 컨텍스트)은 논문의 모델에 없다. 그것은 아래 IAM Conditions 의 영역이다.

---

### 2. Google Cloud IAM — 제품화된 버전

#### 2.1 리소스 계층과 정책 상속

계층은 Organization → Folder → Project → 리소스 4단계다 ([Resource hierarchy and access control](https://docs.cloud.google.com/iam/docs/resource-hierarchy-access-control)). 상속 규칙은 명시적이다:

> *"The effective allow policy for a resource is the union of the allow policy set at that resource and the allow policy inherited from its parent."*
> — [Resource hierarchy and access control](https://docs.cloud.google.com/iam/docs/resource-hierarchy-access-control)

> *"If you set an allow policy on a container resource, then the allow policy also applies to all resources in that container."*
> — [IAM overview](https://docs.cloud.google.com/iam/docs/overview)

**상속은 가산적(additive)이며 자식은 부모가 준 권한을 취소할 수 없다.** allow policy 만으로는 상위 부여를 하위에서 깎을 방법이 없고 그래서 deny policy 가 별도로 존재한다(§2.5). 접근 원인을 파악하려면 *"the resource's allow policy **and** the resource's ancestors' allow policies"* 를 함께 봐야 한다 ([IAM overview](https://docs.cloud.google.com/iam/docs/overview)).

allow policy 는 **role binding 의 목록**이다: *"Each allow policy contains a list of role bindings that associate IAM roles with the principals who are granted those roles."* ([IAM overview](https://docs.cloud.google.com/iam/docs/overview))

#### 2.2 Principal 식별자 형식

[Principal identifiers](https://docs.cloud.google.com/iam/docs/principal-identifiers) 기준.

| 유형 | 형식 |
|---|---|
| 사용자 | `user:USER_EMAIL_ADDRESS` |
| 서비스 계정 | `serviceAccount:SERVICE_ACCOUNT_EMAIL` |
| **Google 그룹** | `group:GROUP_EMAIL_ADDRESS` |
| Workspace 도메인 | `domain:WORKSPACE_DOMAIN` |
| 인증된 전체 | `allAuthenticatedUsers` |
| 전체 공개 | `allUsers` |
| Workforce identity pool | `principalSet://iam.googleapis.com/locations/LOCATION/workforcePools/POOL_ID/principalSets/...` |
| Workload identity pool | `principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/LOCATION/workloadIdentityPools/POOL_ID/principalSets/...` |
| 삭제된 principal | `deleted:principal:PRINCIPAL_EMAIL?uid=NUMERIC_ID` |

principal 은 크게 사람(Google 계정, Google 그룹, workforce identity pool 의 연합 신원)과 워크로드(서비스 계정, workload identity pool 의 연합 신원)로 나뉜다 ([IAM overview](https://docs.cloud.google.com/iam/docs/overview)).

> ⚠️ **우리 설계에 직결되는 관찰**: 그룹 principal 은 이메일 주소로 바인딩된다. 불변 ID 가 아니다. §4.2 에서 이 긴장을 다룬다.

#### 2.3 역할 — basic / predefined / custom

[Roles overview](https://docs.cloud.google.com/iam/docs/roles-overview) 의 표현 그대로:

> *"Basic roles include thousands of permissions across all Google Cloud services. In production environments, **do not grant basic roles unless there is no alternative.** Instead, grant the most limited predefined roles or custom roles that meet your needs."*

- **predefined role** 은 Google 이 관리하며 서비스에 새 기능이 생기면 **자동 갱신**된다 → 관리자가 유지보수 부담을 지지 않는다. Google 이 predefined 를 권하는 실질적 이유가 이것이다.
- **custom role** 은 *"help you enforce the principle of least privilege, because they help to ensure that the principals in your organization have only the permissions that they need"* ([Roles overview](https://docs.cloud.google.com/iam/docs/roles-overview)) 하지만 조직당/프로젝트당 **300개** 상한이 있고 유지 책임이 사용자에게 있다(§2.6).

최소권한 지침 ([Use IAM securely](https://docs.cloud.google.com/iam/docs/using-iam-securely)):

> *"Grant roles at the smallest scope needed. For example, if a user only needs access to publish Pub/Sub topics, grant the Publisher role to the user for that topic."*
> *"Treat each component of your application as a separate trust boundary... create a separate service account for each of the services, then grant only the required permissions to each service account."*

#### 2.4 IAM Conditions — RBAC 위에 얹은 ABAC 층

조건식은 CEL(Common Expression Language)의 부분집합이며 세 곳에 붙는다 ([Conditions overview](https://docs.cloud.google.com/iam/docs/conditions-overview)).

| 붙는 곳 | 필드 |
|---|---|
| allow policy 의 role binding | `condition` (title, description, expression) |
| deny policy 의 deny rule | `denialCondition` |
| principal access boundary policy binding | `condition` |

사용 가능한 속성 범주:

| 범주 | 예 |
|---|---|
| **리소스 속성** | 리소스 타입 / 이름 / 서비스, 리소스에 붙은 태그 (allow·deny 양쪽) |
| **요청 속성** | **타임스탬프**(업무시간·기간 지정, 타임존 지원), **access level**(VPC 기반 IP·기기 검증), 목적지 IP·포트, IAP 의 URL host/path, API 별 속성(역할 제한) |
| **principal 속성** | principal 타입·신원 (principal access boundary policy 전용) |

제약도 문서화돼 있다:
- *"Don't add more than 100 conditional role bindings to a single allow policy"* — 정책 크기 상한 초과 방지.
- 조건은 **legacy basic role 에는 못 쓰고**, `allUsers`/`allAuthenticatedUsers` 부여에도 못 쓴다.
- Cloud Storage 는 conditional role binding 을 쓰려면 uniform bucket-level access 가 필요하다.

**우리 요구사항 중 "만료되는 크로스-Biz 게스트 접근" 이 정확히 이 층이다.** Zanzibar 튜플만으로는 시간 조건을 표현할 수 없다.

#### 2.5 Deny policy — allow 보다 먼저 평가된다

[Deny policies overview](https://docs.cloud.google.com/iam/docs/deny-overview):

> IAM *"always checks relevant deny policies before checking relevant allow policies"* — 거부가 어떤 역할 부여보다 우선한다.

deny rule 의 구성 요소 네 가지:

| 요소 | 설명 |
|---|---|
| `deniedPrincipals` | *"The principals that are denied permissions"* — 개인·서비스계정·그룹, `principalSet://goog/public:all` 같은 집합도 가능 |
| `exceptionPrincipals` | 그 거부에서 **면제**되는 principal (넓은 거부 안의 좁은 예외) |
| `deniedPermissions` | *"The permissions that the specified principals are unable to use"* — FQDN 형식 (예 `iam.googleapis.com/roles.delete`) |
| `denialCondition` | *"A logic expression that affects when the deny rule applies"* — **리소스 태그 함수만 인식**한다 |

deny 도 계층을 따라 내려간다: *"When you attach a deny policy to a project, folder, or organization, the policy is also effective for all resources inside that project, folder, or organization."* 하위 정책이 상위 거부를 덮을 수 없다.

deny 가 따로 있는 이유는 문서가 밝힌 대로 §2.1 의 가산 상속 때문이다. 조건을 모든 부여마다 붙이는 대신 넓게 부여하고 거부를 예외로 두는 운영이 가능해진다.

#### 2.6 정책 상한 — Google 자신이 "개별 부여를 멈추라"고 말하는 지점

[IAM quotas and limits](https://docs.cloud.google.com/iam/quotas) 기준.

| 항목 | 상한 |
|---|---|
| 리소스당 allow policy | **1개** |
| **단일 allow policy 내 총 principal 수** (모든 role binding + audit-logging 예외 합산, 도메인·그룹 포함) | **1,500** |
| 단일 allow policy 내 **도메인 + Google 그룹** 수 | **250** |
| 같은 (역할, principal) 에 조건만 다른 role binding | **20** |
| role binding 조건식의 논리 연산자 | **12** |
| conditional role binding 권장 상한 | **100** (정책 크기 상한 회피용, [Conditions overview](https://docs.cloud.google.com/iam/docs/conditions-overview)) |
| 리소스당 deny policy / deny rule | 각 **500** |
| deny policy 의 총 principal 수 | **2,500** |
| deny policy 의 도메인 + 그룹 수 | **500** |
| custom role — 조직당 / 프로젝트당 | 각 **300** |
| custom role 의 권한 수 / 총 크기 | **3,000** / **64 KB** |
| principal access boundary — 정책당 rule / 리소스당 정책 / 조직당 정책 | **500 / 10 / 1,000** |

이 한도는 quota 요청으로 상향할 수 없다.

**해석**: principal 1,500 은 사원 수가 수천 명이면 개별 부여로는 물리적으로 불가능하다는 뜻이다. 반면 그룹은 250개까지 넣을 수 있고 그룹 1개가 몇 명을 담든 principal 1개로 계산된다. Google 은 "그룹을 쓰라"를 권고문으로만 말하는 게 아니라 한도로 강제한다.

#### 2.7 전파 지연 — 실제 숫자

[Access change propagation](https://docs.cloud.google.com/iam/docs/access-change-propagation) 이 명시하는 값.

| 변경 종류 | 일반 | 최대 |
|---|---|---|
| **정책(allow/deny) 변경** | *"Typically 2 minutes"* | *"potentially 7 minutes or longer"* |
| **그룹 멤버십 변경** | *"Typically several minutes"* | *"potentially hours or longer"* |
| **중첩 그룹 멤버십 변경** | *"Typically several minutes"* | *"potentially hours or longer"* |

문서가 덧붙이는 두 가지 비대칭:
- *"adding a principal to a group propagates faster than removing a principal from a group"* — **추가가 제거보다 빠르다.** 즉 회수(revocation)가 더 늦게 반영된다.
- *"group membership changes propagate faster than nested group membership changes"* — **중첩이 더 느리다.**

정책은 2분, 그룹 멤버십은 수 시간까지. 우리가 그룹 모델을 채택하면 같은 종류의 지연을 어디선가 감수하거나(캐시 TTL), 아니면 판정 시점에 멤버십을 직접 읽어 지연을 0으로 만들되 부하를 감수해야 한다(§5.4). **이것이 그룹 기반 인가의 대가다.**

---

### 3. 핵심 질문 — Google 은 어떻게 개별 부여를 피하는가

#### 3.1 명시적 권고

> *"**Grant roles to groups instead of individual users when possible.** It is easier to update the members of a group than to update the principals in your allow policies."*
> — [Use IAM securely](https://docs.cloud.google.com/iam/docs/using-iam-securely)

이유가 문서에 그대로 적혀 있다. 정책을 고치는 것보다 그룹 멤버를 고치는 것이 쉽기 때문이다. 여기에 §2.6 의 한도(principal 1,500 vs 그룹 250)가 더해져 대규모에서는 사실상 유일한 선택지가 된다.

#### 3.2 그룹 멤버십이 관리 단위가 되는 메커니즘

> *"Each member of a Google group inherits the Identity and Access Management (IAM) roles granted to that group."*
> *"When you add a member to a Google group, they inherit all IAM roles granted to that group, regardless of their Google Groups role."*
> — [Google groups in the Google Cloud console](https://docs.cloud.google.com/iam/docs/groups-in-cloud-console)

**결과: 사람이 팀을 옮기면 정책을 한 줄도 건드리지 않는다.** 리소스 정책은 `group:X@…` 에 역할을 준 채 그대로 있고 사람은 그룹 A 에서 빠지고 그룹 B 에 들어간다. 정책의 개수는 사람 수와 무관하게 유지된다.

이것이 리소스 정책과 소속 관리의 디커플링이다. 리소스 정책은 "무엇을 할 수 있는가"만 말하고 "누가 그 집단인가"는 디렉터리(그룹)가 말한다. 두 시스템의 변경 주기가 다르다. 정책은 리소스가 생길 때, 멤버십은 사람이 움직일 때.

감사 경로도 존재한다: *"Google Cloud will automatically generate audit logs for actions taken in Google Workspace"* — 멤버 추가·제거 포함 ([Google groups in the console](https://docs.cloud.google.com/iam/docs/groups-in-cloud-console)).

#### 3.3 사람의 지속 신원

Google 이 사람에게 쓰는 principal 은 `user:USER_EMAIL_ADDRESS` 이고, 삭제된 principal 은 `deleted:principal:PRINCIPAL_EMAIL?uid=NUMERIC_ID` 로 표현된다 ([Principal identifiers](https://docs.cloud.google.com/iam/docs/principal-identifiers)). 뒤에 숫자 UID 가 존재한다는 사실은 식별자 형식에서 드러나지만, 정상 상태의 바인딩은 이메일로 되어 있다.

> **확인 불가**: "이메일이 바뀌었을 때 기존 role binding 이 어떻게 되는가" 를 명시한 Google Cloud 공식 문서 페이지를 이번 조사에서 찾지 못했다. `deleted:` 접두사와 `?uid=` 의 존재는 내부적으로 안정 UID 가 있음을 시사하지만 rename 시 바인딩 보존 여부는 1차 자료로 검증하지 못했다. 우리 설계에서는 이 점을 Google 을 따라가지 않는 근거로 삼는다(§4.2).

#### 3.4 중첩 그룹 지원 여부

- **Zanzibar 논문 수준**: 지원한다. `⟨userset⟩` 이 중첩 그룹 멤버십을 표현하고(§1.1), Leopard 가 *"deeply and widely nested group membership"* 을 처리한다(§1.3).
- **Workspace Directory API 수준**: *"A group member can be a user or another group."* 단, **순환 금지** — *"the API returns an error for cycles in group memberships. For example, if group1 is a member of group2, group2 cannot be a member of group1."* 그리고 **지연** — *"there may be a delay of up to 10 minutes before the child group's members appear as members of the parent group."* ([Manage group members](https://developers.google.com/workspace/admin/directory/v1/guides/manage-group-members))
- **Cloud IAM 수준**: [Access change propagation](https://docs.cloud.google.com/iam/docs/access-change-propagation) 이 **"Changes to nested group memberships"** 를 독립 항목으로 다루고 전파 시간을 명시하므로, IAM 이 중첩 그룹을 평가한다는 것은 문서로 확인된다. 다만 [Google groups in the console](https://docs.cloud.google.com/iam/docs/groups-in-cloud-console) 페이지 자체는 중첩을 언급하지 않는다.

정리된 caveat 세 가지: 순환 금지 / 자식 그룹 반영에 최대 10분 / 중첩은 비중첩보다 전파가 느림.

---

### 4. 우리 문제로 매핑 — 팀명이 바뀌고, 조직 코드 유무를 모른다

#### 4.1 Zanzibar/IAM 모델에 안정적인 팀 ID 가 필요한가 — **필요 없다**

논문 §2.1 의 문법을 다시 본다: `⟨object⟩ ::= ⟨namespace⟩ ':' ⟨object id⟩`. **`⟨object id⟩` 는 클라이언트가 정하는 문자열이다** ([논문 §2.1](https://www.usenix.org/system/files/atc19-pang.pdf)). 외부 시스템의 코드를 가져오라는 요구가 문법 어디에도 없다. `group:eng#member@11` 에서 `eng` 는 Google 이 자기 네임스페이스 안에서 붙인 이름이다. HR 시스템의 조직 코드가 아니다.

그래서 답은 이렇다. **팀을 "자체 식별자를 가진 그룹 리소스"로 만들면 조직 코드는 필요 없다.** 팀은 우리 레지스트리 안의 객체가 되고, 멤버십은 `team:<우리가 발급한 id>#member@user:<empno>` 튜플의 집합이 된다.

**이 모델이 조직 코드 유무에 대해 불변인 이유 (논증)**

1. **모델의 모든 참조가 내부 폐포(closure)를 이룬다.** 튜플에 등장하는 식별자는 두 종류뿐이다 — 레지스트리가 발급한 `team_id`/`agent name`, 그리고 사원 마스터의 `empno`. `empno` 는 불변이 확정됐고, `team_id` 는 우리가 발급하므로 정의상 불변이다. **외부에서 불변성을 빌려오는 지점이 하나도 없다.**
2. **조직 코드는 이 폐포에 등장하지 않는다.** 등장하지 않는 것은 있어도 없어도 판정 결과를 바꾸지 않는다. 따라서 인가 판정 경로는 조직 코드의 존재 여부와 **논리적으로 독립**이다.
3. **조직 코드가 나중에 발견되면 붙는 자리는 `teams` 문서의 필드 하나다.** `{ teamId, displayName, bizId, orgCode?: string|null }` — `orgCode` 는 **선택적 외부 참조(external reference)** 이며, 튜플에도 인덱스에도 판정 로직에도 들어가지 않는다. 조회·동기화·리포팅에만 쓰인다. 필드 하나 추가는 스키마 변경이지 모델 변경이 아니다.
4. **역방향도 성립한다.** 조직 코드가 끝내 없다고 밝혀져도 바꿀 것이 없다. `orgCode` 를 채우지 않으면 그만이다.

이 성질은 SCIM 스펙이 `externalId` 를 둔 이유와 정확히 같은 구조다(§6.2). 프로비저닝 클라이언트의 식별자를 서비스 제공자의 내부 `id` 와 분리해 옆에 두는 것이다.

#### 4.2 그룹 이름이 바뀌면 — Google 은 어떻게 하는가

**Workspace 그룹에는 이메일과 별개의 불변 ID 가 있다.** Directory API 의 Group 리소스:

> *"Read-only. The unique ID of a group. A group `id` can be used as a group request URI's `groupKey`."*
> — [Directory API: Groups resource](https://developers.google.com/workspace/admin/directory/reference/rest/v1/groups)

그리고 Google 자신의 권고:

> *"In general, we recommend **not using the group's email address as a key for persistent data**, because the email address is subject to change."*
> — [Manage groups (Directory API)](https://developers.google.com/workspace/admin/directory/v1/guides/manage-groups)

그룹 갱신은 `PUT .../groups/{groupKey}` 로 하며, groupKey 는 이메일이 아니라 불변 `id` 를 쓰라고 안내한다.

Google 은 "이메일을 영속 키로 쓰지 말라"고 하면서, Cloud IAM 의 role binding 은 `group:GROUP_EMAIL_ADDRESS` 로 이메일을 쓴다(§2.2). Google 의 제품화된 IAM 은 논문의 원칙(불변 객체 ID)을 완전히 따르지 않는다. **여기에 우리가 배워야 할 모순이 있다.**

> **확인 불가 (정직하게)**: 그룹 이메일을 변경했을 때 기존 IAM role binding 이 자동으로 따라가는지, 아니면 끊어지는지를 명시한 Google 1차 문서를 찾지 못했다. `deleted:` principal 형식의 존재는 내부 UID 를 시사하지만 rename 케이스의 동작은 문서화된 것을 확인하지 못했다.
>
> **우리의 결론**: 이 불확실성 자체가 답이다. 우리 튜플의 subject 에는 절대 표시명·이메일을 넣지 않는다. `team:T-004821`, `user:E123456` 만 넣는다. Google 이 이메일을 쓰는 것은 Workspace 이메일이 사실상 안정적이라는 운영 관행 위에 얹힌 편의이고, 우리 팀명은 매년 바뀌는 것이 확정된 사실이므로 같은 편의를 빌릴 수 없다.

#### 4.3 그룹 삭제 vs 멤버십 변경 — 팀 해체·병합에 해당하는 케이스

멤버십 변경은 안전하다. 그룹 객체가 유지되므로 그룹에 걸린 부여는 그대로다(§3.2).

삭제는 다르다. Google 은 명시적으로 경고한다:

> *"Deleting a group is irreversible. To avoid unexpected access changes, **revoke all IAM roles from the group, then wait at least 7 days before deleting it.**"*
> — [Google groups in the Google Cloud console](https://docs.cloud.google.com/iam/docs/groups-in-cloud-console)

이 절차가 우리에게 주는 것은 "삭제 전에 권한을 먼저 회수하고, 유예기간을 둔다" 는 순서다. 이유는 §2.7 의 전파 지연과 맞물린다. 그룹을 먼저 지우면 권한이 언제 실제로 사라지는지 예측할 수 없다.

팀 해체·병합에 대한 우리 대응은 [02-architecture.md §5.3](../02-architecture.md) 의 `status: merged | dissolved` + `successorTeamId` 로 이미 설계돼 있다. Google 선례가 더해주는 것은 **삭제하지 말고 상태 전이시키라**는 것이다. Zanzibar 모델에서 팀은 객체이고, 객체를 지우면 그 객체를 subject 로 가진 모든 튜플이 고아가 된다. `dissolved` 로 표시하고 `successorTeamId` 를 따라가는 편이 회복 가능하다.

> **확인 불가**: Workspace/Cloud 문서에서 **그룹 병합(merge)** 이라는 1급 연산을 찾지 못했다. Google 은 병합을 제공하지 않고 "새 그룹 + 멤버 이관 + 옛 그룹 정리" 로 처리하는 것으로 보이나 이를 명시한 1차 문서는 확인하지 못했다.

#### 4.4 최악의 경우 — 사원 마스터가 팀 멤버십을 주지 못한다면

이 경우 `team:<id>#member@user:<empno>` 튜플을 누군가 사람이 유지해야 한다. 1차 자료에서 확인되는 근거는 하나다. **Google 그룹은 그룹 자체가 관리 역할을 가진다.**

[Manage group members (Directory API)](https://developers.google.com/workspace/admin/directory/v1/guides/manage-group-members) 의 역할 정의:

| 역할 | 권한 (문서 표현) |
|---|---|
| `OWNER` | *"change send messages to the group, add or remove members, change member roles, change group's settings, and delete the group. An OWNER must be a member of the group."* |
| `MANAGER` | *"can do everything done by an OWNER role except make a member an OWNER or delete the group."* — Admin console 활성화 시에만 존재 |
| `MEMBER` | *"subscribe to a group, view discussion archives, and view the group's membership list."* |

**그룹은 HR 피드 없이도 자체 관리자(OWNER/MANAGER)에 의해 운영될 수 있도록 설계돼 있다.** 이것이 "사원 마스터가 멤버십을 못 준다"의 1차 자료 기반 대응이다. 그룹에 소유자 역할을 넣고, 소유자가 멤버를 추가·제거한다. 그룹 리소스 자체가 자기 관리 권한을 담는다.

우리 모델로 옮기면, 팀 그룹에도 관계가 필요하다:

```jsonc
{ "resource": "team:T-004821", "relation": "admin",  "subject": "user:E123456" }
{ "resource": "team:T-004821", "relation": "member", "subject": "user:E789012" }
```

`team:...#admin` 이 Google 의 OWNER/MANAGER 에 해당한다. 이 관계가 있으면 팀 멤버십의 유지 주체가 레지스트리 안에 존재하게 되어, HR 피드 유무와 무관하게 모델이 닫힌다.

부트스트랩(최초 admin 은 누가 되는가)에 대해서는:

> **1차 자료 없음**: "최초 발행자를 자동으로 소유자로 등록한다", "자가 신청 + 승인" 같은 구체 절차를 뒷받침하는 Google 1차 문서를 찾지 못했다. Google 이 명시하는 것은 *"An OWNER must be a member of the group"* 이라는 제약과 *"A new group doesn't have any members. You must add a member to a group."* ([Manage groups](https://developers.google.com/workspace/admin/directory/v1/guides/manage-groups)) 뿐이다. 그룹은 빈 상태로 생성되고 명시적 추가가 필요하다는 것까지가 문서로 확인되는 범위다. 부트스트랩 정책은 우리가 정해야 하는 운영 결정이며, 근거는 §5 의 열린 질문으로 넘긴다.

**HR 피드 유무에 따른 3단 fallback (운영 권고 — 1차 자료 근거 없음, 위 재료로부터의 설계 판단)**

| 단계 | 조건 | 멤버십 유지 방식 |
|---|---|---|
| 1 | 사원 마스터가 empno→현 소속을 준다 | 주기 동기화로 `team:*#member` 튜플 재생성. 사람 개입 0 |
| 2 | 마스터가 주지 못한다 | `team:*#admin` 이 수동 관리 (= Google 의 OWNER 모델) |
| 3 | admin 도 공석 | 팀을 `dissolved` 로 두고, 자산은 [02-architecture.md §7.2](../02-architecture.md) 의 **원작성자 empno → 현 소속 조회** 경로로 승계 시도 |

**이 모델이 동작하지 않는 지점 (정직하게)**: 3단계에서 사원 마스터가 empno→소속을 주지 못하면 승계 경로가 끊긴다. 그때 남는 것은 원작성자 empno 로 사람에게 직접 연락하는 수동 절차뿐이다. 자동화할 근거가 없다.

---

## 어떻게 사용하는가? (How)

### 5. 결정을 위한 비교와 권고 모델

#### 5.1 RBAC / ABAC / ReBAC 비교 — 우리 요구사항 기준

| | **RBAC** | **ABAC (IAM Conditions)** | **ReBAC (Zanzibar)** |
|---|---|---|---|
| 표현하는 것 | `(주체, 역할)` — 역할이 권한 묶음 | `(주체, 역할, **조건식**)` — 요청·리소스 속성 평가 | `(객체 # 관계 @ 주체)` — 주체 자리에 **집합**도 가능 |
| 표현 못 하는 것 | "**어느** 리소스의 owner 인가" — 리소스별 역할이 필요하면 역할 수가 폭발 | 관계 그래프 순회. 조건은 **바인딩에 이미 있는** principal 을 필터링할 뿐, 새 주체를 끌어오지 못함 | **시간·컨텍스트 조건** — 논문 모델에 없음 |
| 1차 근거 | [Roles overview](https://docs.cloud.google.com/iam/docs/roles-overview) | [Conditions overview](https://docs.cloud.google.com/iam/docs/conditions-overview) | [Zanzibar §2.1, §2.3.1](https://www.usenix.org/system/files/atc19-pang.pdf) |

**우리 요구사항 5개를 어느 층이 감당하는가**

| 요구사항 | 필요한 층 | 근거 |
|---|---|---|
| 에이전트의 **owner** | ReBAC | 리소스 단위 관계. RBAC 로는 에이전트마다 역할을 만들어야 함 |
| **maintainer** (owner 가 위임) | ReBAC | 같은 리소스에 관계만 다른 튜플 1건 |
| **reviewer** (심사자) | ReBAC (+선택적 RBAC) | 리소스 단위면 ReBAC. 전사 심사자 풀이면 RBAC 스코프로 충분 |
| **팀 상속 접근** (팀 멤버 전원이 maintainer) | ReBAC 필수 | `@team:T-…#member` — userset 이 없으면 사람 수만큼 튜플이 필요 |
| **크로스-Biz 게스트 접근 + 만료** | **ABAC 필요** | Zanzibar 튜플에 시간 개념 없음. IAM 은 `request.time` 조건으로 처리 ([Conditions overview](https://docs.cloud.google.com/iam/docs/conditions-overview)) |

**결론: ReBAC 을 뼈대로, 만료만 ABAC 로 얹는다.** 이는 Google 이 한 것과 같은 층위 구성이다 — Zanzibar 가 관계를, IAM Conditions 가 조건을.

#### 5.2 권고 인가 모델 — 우리 어휘로

**단일 컬렉션, 4개 네임스페이스.** 기존 [02-architecture.md §5.2](../02-architecture.md) 의 `ownership` 컬렉션을 그대로 쓰되 subject 에 `team:` 을 허용하고 `expiresAt` 을 조건 층으로 승격한다.

```jsonc
// ownership 컬렉션 — 하나의 튜플 형태
{
  "resource": "agent:kr.co.skhynix.dram.T-004821/defect-summarizer",
  "relation": "owner",              // owner | maintainer | reviewer | member | admin
  "subject":  "user:E123456",       // user:<empno> | team:<team_id>#member
  "grantedAt": "2026-09-03T00:00:00Z",
  "grantedBy": "user:E999999",
  "expiresAt": null                 // null = 무기한. 값 있으면 판정 시 비교 (ABAC 층)
}
```

**namespace 4개**

| namespace | object id | 관계 |
|---|---|---|
| `agent:` | `kr.co.skhynix.<biz>.<team_id>/<name>` | `owner`, `maintainer`, `reviewer` |
| `team:` | `T-004821` (레지스트리 발급) | `member`, `admin` |
| `user:` | `E123456` (**empno**) | (subject 전용) |
| `biz:` | `BIZ-DRAM` | `guest` |

**relation 파생 규칙** (Zanzibar 의 userset rewrite 에 해당. 우리는 코드 상수 5줄로 충분하다 — config 언어를 만들 이유가 없다):

```
owner      ⊂ maintainer ⊂ reviewer      # 상위가 하위를 포함 (computed_userset 상당)
maintainer ⊃ team:<owner팀>#member       # 소유 팀 멤버는 자동 maintainer (tuple_to_userset 상당)
```

**전형적인 튜플 집합**

```
agent:kr.co.skhynix.dram.T-004821/defect-summarizer#owner      @user:E123456
agent:kr.co.skhynix.dram.T-004821/defect-summarizer#maintainer @team:T-004821#member
team:T-004821#member @user:E123456
team:T-004821#member @user:E789012
team:T-004821#admin  @user:E123456
```

**4가지 시나리오 처리**

**(a) 팀 이름이 바뀐다 (`AI기술팀` → `AI솔루션2팀`)**
```
변경: teams 문서의 displayName 1개 + history 배열에 이전 값 추가
튜플 변경: 0건
에이전트 문서 재발행: 0건
```
`team_id` 도 `agent name` 도 표시명을 담지 않으므로 아무것도 움직이지 않는다. Google 이 이메일을 principal 로 쓰면서 겪는 문제(§4.2)를 우리는 애초에 만들지 않는다.

**(b) 팀이 병합된다 (T-004821 → T-005000 에 흡수)**
```jsonc
// teams: T-004821
{ "status": "merged", "successorTeamId": "T-005000" }

// 튜플: 멤버십만 이관. 에이전트 튜플은 그대로 둔다
- team:T-004821#member @user:E123456
+ team:T-005000#member @user:E123456
```
`agent:…#maintainer @team:T-004821#member` 는 건드리지 않는다. 판정 시 `T-004821` 이 `merged` 면 `successorTeamId` 를 1홉 따라간다. 이유는 Google 이 *"revoke all IAM roles from the group, then wait at least 7 days before deleting it"* 이라고 경고하는 이유와 같다 ([Google groups in the console](https://docs.cloud.google.com/iam/docs/groups-in-cloud-console)). 그룹 객체를 즉시 없애면 회복이 불가능하다. `successorTeamId` 체인은 **최대 1홉만 따라간다**(무한 루프·성능 방어). 2홉 이상 필요한 상태는 데이터 정리 대상으로 경보한다.

**(c) 사원이 팀을 옮긴다 (E123456: T-004821 → T-005000)**
```
- team:T-004821#member @user:E123456
+ team:T-005000#member @user:E123456

agent 튜플 변경: 0건
```
이것이 §3.2 에서 본 Google 의 핵심 메커니즘이다. 리소스 정책은 그대로, 멤버십만 움직인다. 단, `#owner @user:E123456` 로 개인에게 직접 걸린 소유권은 따라간다 (사람이 옮겨도 그가 만든 것의 owner 는 그다). 이는 의도된 동작이며 [02-architecture.md §7.2](../02-architecture.md) 의 "원작성자 empno" 승계 단서와 일치한다.

**(d) 크로스-Biz 게스트 접근 + 만료**
```jsonc
{ "resource": "agent:kr.co.skhynix.dram.T-004821/defect-summarizer",
  "relation": "reviewer",
  "subject":  "user:E555555",           // 타 Biz 사원
  "grantedBy": "user:E123456",          // owner 가 부여
  "expiresAt": "2026-12-31T23:59:59Z" } // ← ABAC 층
```
판정 시 `expiresAt != null && now > expiresAt` 이면 무시한다. Google 이 `request.time` 조건으로 하는 것과 같다 ([Conditions overview](https://docs.cloud.google.com/iam/docs/conditions-overview)). Biz 단위 게스트라면 subject 를 `biz:BIZ-NAND#member` 로 두어 튜플 1건으로 처리한다. 비전이성([02-architecture.md §7.3 경로 C](../02-architecture.md))은 이 튜플이 다른 Biz 로 파생되지 않는다는 규칙으로 유지된다. `biz:` 는 subject 로만 쓰고 `biz:X#guest @biz:Y#member` 같은 튜플을 금지한다.

**deny 는 도입하지 않는다.** Google 이 deny policy 를 만든 이유는 §2.1 의 가산 상속 때문이다. 상위에서 준 것을 하위에서 깎을 방법이 없었다 ([Deny policies overview](https://docs.cloud.google.com/iam/docs/deny-overview)). 우리 모델에는 상속 계층이 없다(Organization/Folder/Project 에 해당하는 것이 없다). 깎아야 할 상위 부여가 없으므로 deny 도 필요 없다. AWS 식 "explicit deny > allow > implicit deny" ([AWS policy evaluation logic](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic.html)) 도 같은 이유로 불필요하다. 기본 거부 + 명시 허용만 둔다.

#### 5.3 폐쇄망에서 실제로 돌릴 수 있는 Zanzibar 구현체

| | **SpiceDB** | **OpenFGA** | **Ory Keto** |
|---|---|---|---|
| 데이터 모델 | definition / relation / permission / relationship. 관계 형태 `document:budget#reader@user:anne` ([Schema](https://authzed.com/docs/spicedb/concepts/schema)) | authorization model(type definitions) + relationship tuple `(user, relation, object)` + 선택적 condition ([Concepts](https://openfga.dev/docs/concepts)) | relation tuple. *"User:x is in readers of Document:y"* ([Keto docs](https://www.ory.com/docs/keto)) |
| 연산자 | union `+`, intersection `&`, exclusion `-`, arrow `->` ([Schema](https://authzed.com/docs/spicedb/concepts/schema)) | DSL → API 문법 변환, CLI 제공 ([Concepts](https://openfga.dev/docs/concepts)) | 문서에서 확인 못 함 |
| 저장소 | CockroachDB(자체 호스팅 권장), Spanner, **PostgreSQL 15+**(단일 리전 권장), MySQL(*"Not recommended"*), memdb(개발용) ([Datastores](https://authzed.com/docs/spicedb/concepts/datastores)) | **memory / postgres / mysql / sqlite** ([Configure OpenFGA](https://openfga.dev/docs/getting-started/setup-openfga/configure-openfga)) | PostgreSQL, MySQL, CockroachDB ([GitHub](https://github.com/ory/keto)) |
| 폐쇄망 자체 호스팅 | 가능 ([Datastores](https://authzed.com/docs/spicedb/concepts/datastores) 가 self-hosted 를 전제로 권장 구성을 제시) | Docker / Kubernetes 설치 가이드 존재 ([Setup overview](https://openfga.dev/docs/getting-started/setup-openfga/overview)) | 자체 호스팅 전제 |
| 라이선스 | **Apache-2.0** ([GitHub](https://github.com/authzed/spicedb)) | **Apache-2.0** ([GitHub](https://github.com/openfga/openfga)) | **Apache-2.0** ([GitHub](https://github.com/ory/keto)) |
| CNCF | - | **CNCF 프로젝트** — 문서 푸터가 *"© 2026 The Linux Foundation"* 이고 CNCF Slack 을 안내 ([openfga.dev](https://openfga.dev/docs/getting-started/setup-openfga/configure-openfga)) | - |
| 유지보수 상태 | 활발 (repo 최근 push 2026-09-02) | 활발 (archived=false) | **활발** — 최신 릴리스 `v26.2.0` (2026-03-20), 최근 push 2026-08-31 |
| Zanzibar 계보 | *"In 2019, Google released the paper 'Zanzibar'... providing the original inspiration for SpiceDB"* ([GitHub](https://github.com/authzed/spicedb)) | JSON 문법이 Zanzibar 논문 참조 ([Concepts](https://openfga.dev/docs/concepts)) | *"the first open-source implementation of the design principles and specifications described in Zanzibar"* ([Keto docs](https://www.ory.com/docs/keto)) |

세 프로젝트 모두 Apache-2.0 / 자체 호스팅 가능 / 인터넷 불필요 — 폐쇄망 제약을 통과한다. 저장소는 셋 다 관계형 DB 를 요구한다. **우리 스택에는 PostgreSQL 이 없다**([02-architecture.md G4](../02-architecture.md) 는 MongoDB/Redis/OpenSearch/MinIO 만 나열). 어느 것을 쓰든 새 DB 를 하나 추가해야 한다.

#### 5.4 판정 — 우리 규모에 Zanzibar 엔진이 정당한가

**우리 규모 추산**

| 항목 | 수량 | 튜플 기여 |
|---|---|---|
| 사원 | 수천 명 | `team:*#member` 약 **수천 건** (1인 1팀 가정) |
| 팀 | 수백 개 | `team:*#admin` 수백 건 |
| 에이전트 | 수천 개 | `agent:*#owner` 수천 + `#maintainer` 수천 |
| 게스트·리뷰어 | 수백 | 수백 건 |
| **합계** | | **약 1~2만 튜플** |

**Zanzibar 논문의 기준선과 나란히 놓으면** ([논문 §4](https://www.usenix.org/system/files/atc19-pang.pdf)):

| | Zanzibar | 우리 (인스턴스 1개) | 배율 |
|---|---|---|---|
| 튜플 총량 | 2조 | ~2만 | **1억분의 1** |
| **네임스페이스당 튜플 중앙값** | **15,000** | ~20,000 | **약 1배** |
| Check QPS 피크 | 4.2M | 수십 (레지스트리는 대화형 카탈로그) | ~10만분의 1 |
| 서버 | 10,000+ | 1~3 (FastAPI) | - |
| 복제 지역 | 30+ | 1 | - |

**핵심 관찰: 우리 레지스트리 전체가 Zanzibar 의 "중앙값 네임스페이스 1개"와 같은 크기다.** Zanzibar 의 아키텍처 요소 대부분은 2조 튜플 × 30지역 × 4.2M QPS 를 감당하기 위한 것이다 — Leopard 비정규화 인덱스, 요청 hedging, 분산 캐시 트리, Slicer, 락 테이블, 타임스탬프 양자화, zookie. 1.5만 튜플에는 필요 없다.

**판정: Zanzibar 스타일 엔진은 도입하지 않는다. MongoDB 의 잘 인덱싱된 관계 튜플 컬렉션으로 충분하다.**

근거 4가지:

1. **데이터가 인덱스 하나에 다 들어간다.** 2만 문서는 MongoDB 에서 수 MB 다. [02-architecture.md §5.3](../02-architecture.md) 이 이미 잡은 `{resource:1, subject:1, relation:1}` unique 인덱스에 `{subject:1, relation:1}` 을 추가하면 양방향 조회가 인덱스 스캔으로 끝난다.
2. **그래프 깊이가 2다.** `user → team → agent`. 논문이 Leopard 를 만든 이유는 *"deep or wide"* 중첩이었고([논문 §3.2.3](https://www.usenix.org/system/files/atc19-pang.pdf)), 우리는 깊이 2에서 고정이다. **중첩 팀을 허용하지 않기로 정하면** 재귀 자체가 사라진다 — 조회 2회(사용자의 팀 목록, 리소스의 튜플)로 판정이 끝난다.
3. **비정규화 비용이 이득을 초과한다.** 논문: 튜플 1건 변경이 Leopard 이벤트 *"tens of thousands"* 를 낳는다. 2만 튜플 규모에서 이 증폭을 감당할 이유가 없다.
4. **zookie 는 우리 문제가 아니다.** "new enemy" 문제(§1.4)는 **다중 지역 복제 + 스테일 읽기**에서 발생한다. 인스턴스가 단일 리전 단일 DB 이고 MongoDB 가 기본적으로 primary 읽기라면, 인과 순서가 스토리지 수준에서 이미 보장된다. **zookie 프로토콜은 우리에게 순수한 복잡도다.**

**대신 Zanzibar 에서 가져오는 것 3가지**: ① 튜플 형태(`object#relation@subject`) ② 그룹은 별도 개념이 아니라 같은 튜플이라는 원칙 ③ 관계 파생 규칙을 데이터가 아니라 코드/설정에 두는 분리(§5.2 의 5줄 상수).

**재검토 트리거 (미리 정해둔다)**: 튜플이 100만 건을 넘거나, 관계 그래프 깊이가 3 이상이 되거나(중첩 팀·중첩 리소스 도입), Check 가 1,000 QPS 를 넘으면 SpiceDB/OpenFGA 를 다시 검토한다. 셋 다 Apache-2.0 이고 데이터 모델이 우리 튜플과 1:1 대응하므로 이관 비용은 스키마 변환 스크립트 수준이다. 지금 안 쓰는 것이 나중을 막지 않는다.

---

### 6. 인접 선례

#### 6.1 AWS IAM — Google 과 대조되는 평가 로직

[AWS policy evaluation logic](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic.html) 의 요약 3줄:

> - *"By default, all requests are implicitly denied with the exception of the AWS account root user, which has full access."*
> - *"Requests must be explicitly allowed by a policy or set of policies following the evaluation logic below to be allowed."*
> - *"**An explicit deny overrides an explicit allow.**"*

[상세 평가 순서](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic_policy-eval-denyallow.html): **Deny 평가 → Organizations RCP → Organizations SCP → 리소스 기반 정책 → 아이덴티티 기반 정책 → permissions boundary → 세션 정책**. 각 단계에서 허용이 없으면 그 자리에서 `Deny` 를 반환한다.

**Google 과의 대조**

| | Google Cloud | AWS |
|---|---|---|
| 조합 방식 | 계층 상속이 **union**(가산) ([Resource hierarchy](https://docs.cloud.google.com/iam/docs/resource-hierarchy-access-control)) | boundary/SCP 는 **intersection**(교집합) — *"the resulting permissions are the intersection of the two categories"* |
| deny | 별도 deny policy, allow 보다 **먼저** 검사 ([Deny overview](https://docs.cloud.google.com/iam/docs/deny-overview)) | 모든 정책 타입 안에 `Deny` statement 로 내장, 어디 하나라도 있으면 즉시 Deny |
| 정책 부착 | 리소스에 allow policy 1개 | 아이덴티티 기반 + 리소스 기반 + boundary + SCP/RCP + 세션 (다중 축) |

**우리가 취할 것**: AWS 의 `default deny → explicit allow` 뼈대만 취한다(§5.2). 다축 정책 조합은 표현력 대비 디버깅 비용이 크고 우리는 상속 계층 자체가 없어 교집합/차집합을 계산할 대상이 없다.

#### 6.2 SCIM (RFC 7643 / 7644) — 사원 마스터가 팀 멤버십을 밀어줄 수 있다면

**`Group` 리소스는 외부 조직 코드 없이 완전히 동작한다.** RFC 7643 §4.2 의 필수 속성은 `displayName` 뿐이고 멤버는 `members` 복합 속성으로 표현된다 ([RFC 7643](https://datatracker.ietf.org/doc/html/rfc7643)).

| `members` 하위 속성 | 의미 |
|---|---|
| `value` | 멤버가 되는 **SCIM 리소스의 `id`** |
| `$ref` | 그 리소스의 URI — **User 또는 Group** 을 가리킬 수 있다 → **중첩 그룹 지원** |
| `type` | 멤버 타입 |

**멤버십은 Group 리소스에 저장된다.** User 쪽의 `groups` 속성은 `readOnly` 이므로 멤버십 변경은 Group 을 통해서만 한다 ([RFC 7643 §4.1.2](https://datatracker.ietf.org/doc/html/rfc7643)). 이 방향성은 우리 튜플 모델(`team:…#member@user:…` 이 team 을 resource 로 둠)과 정확히 일치한다.

**`externalId` — 조직 코드를 나중에 꽂을 자리가 스펙에 이미 있다.** RFC 7643 §3.1 정의를 그대로 옮긴다:

> *"A String that is an identifier for the resource as defined by the provisioning client. The 'externalId' may simplify identification of a resource between the provisioning client and the service provider by allowing the client to use a filter to locate the resource with an identifier from the provisioning domain, **obviating the need to store a local mapping between the provisioning domain's identifier of the resource and the identifier used by the service provider.**"*
> — [RFC 7643 §3.1](https://datatracker.ietf.org/doc/html/rfc7643)

속성 특성: `caseExact = true`, `mutability = readWrite`, optional.

**이것이 §4.1 논증의 스펙 차원 확인이다.**
- SCIM 은 서비스 제공자의 **`id`(불변, 제공자 발급)** 와 프로비저닝 클라이언트의 **`externalId`(선택적, 외부 발급)** 를 **처음부터 분리**해 두었다.
- `externalId` 는 **optional** 이므로 **없어도 스펙이 완결된다** → 조직 코드가 없어도 동작한다.
- 조직 코드가 나중에 밝혀지면 **`externalId` 에 넣기만 하면 된다** → 모델 변경 없음. RFC 의 표현대로 *"obviating the need to store a local mapping"*.

**멤버십 변경은 PATCH 로 표현된다** ([RFC 7644](https://datatracker.ietf.org/doc/html/rfc7644)):

```jsonc
// 추가
{"op":"add","path":"members",
 "value":[{"value":"2819c223-7f76-453a-919d-413861904646",
           "display":"Babs Jensen",
           "$ref":"https://example.com/v2/Users/2819c223-...-413861904646"}]}

// 제거 — 경로 필터로 대상 지정
{"op":"remove","path":"members[value eq \"2819c223-...-413861904646\"]"}
```

이미 멤버인 사용자를 다시 add 하면 리소스를 변경하지 않고 성공을 반환한다(멱등). 이 성질 덕에 주기 동기화를 안전하게 반복할 수 있다.

> **확인 불가**: PATCH 의 경로 필터에서 `externalId` 로 멤버를 지정하는 예시는 RFC 7644 에서 찾지 못했다. 예시는 일관되게 `value`(리소스 `id`)를 쓴다. `externalId` 는 리소스 조회(filter)용이고, 멤버 참조에는 쓰이지 않는다.

**우리에게 주는 결론**: 사원 마스터가 SCIM 을 낼 수 있다면 `Group` 을 그대로 받아 `team:*#member` 튜플로 매핑하면 되고, 조직 코드가 있으면 `externalId` 로, 없으면 그냥 비워두면 된다. 어느 쪽이든 우리 `team_id` 는 우리가 발급한다. SCIM 도 서비스 제공자 `id` 는 제공자가 발급하는 것으로 규정한다.

---

### 7. Google 에서 **복사하지 말아야 할 것**

우리 규모(사원 수천, 팀 수백, 에이전트 수천, 튜플 ~2만)에서 명백한 과설계다.

| 복사하지 말 것 | 이유 (1차 근거) |
|---|---|
| **Zookie / 스냅샷 일관성 프로토콜** | "new enemy" 문제는 다지역 복제 + 스테일 읽기의 산물([논문 §2.2](https://www.usenix.org/system/files/atc19-pang.pdf)). 단일 리전 단일 DB 에는 발생하지 않는다 |
| **Leopard 식 비정규화 인덱스** | 튜플 1건 변경 → *"tens of thousands"* Leopard 이벤트([논문 §3.2.4](https://www.usenix.org/system/files/atc19-pang.pdf)). 깊이 2 그래프에 필요 없다 |
| **hedging / Slicer / 캐시 트리 / 락 테이블 / 타임스탬프 양자화** | 전부 4.2M QPS × 10,000 서버 대응책([논문 §3.2.5–3.2.7](https://www.usenix.org/system/files/atc19-pang.pdf)) |
| **namespace config 언어 (protobuf DSL)** | 1,500 네임스페이스 × 중앙값 500줄 config 를 관리하기 위한 것([논문 §4](https://www.usenix.org/system/files/atc19-pang.pdf)). 우리는 namespace 4개 — 코드 상수 5줄이 맞다 |
| **deny policy / principal access boundary** | Google 이 deny 를 만든 원인은 가산 상속([Deny overview](https://docs.cloud.google.com/iam/docs/deny-overview)). 우리에겐 상속 계층이 없다 |
| **basic/predefined/custom 3층 역할 체계** | predefined 를 Google 이 관리해주는 것이 가치의 핵심([Roles overview](https://docs.cloud.google.com/iam/docs/roles-overview)). 우리는 우리가 관리하므로 3층이 관리 대상만 늘린다. 관계 4종(`owner/maintainer/reviewer/member`)으로 시작 |
| **CEL 조건식 엔진** | IAM Conditions 는 리소스 태그·access level·IP·시간 등 다양한 속성을 위한 것([Conditions overview](https://docs.cloud.google.com/iam/docs/conditions-overview)). 우리가 필요한 조건은 **만료 하나** — `expiresAt` 필드 비교 1줄 |
| **그룹 principal 을 이메일/표시명으로 바인딩** | Google 자신이 *"not using the group's email address as a key for persistent data"* 라고 권고한다([Manage groups](https://developers.google.com/workspace/admin/directory/v1/guides/manage-groups)). Cloud IAM 이 이메일을 쓰는 것은 따라할 관행이 아니다 |
| **중첩 그룹(팀의 팀)** | 순환 금지, 자식 반영 최대 10분([Manage group members](https://developers.google.com/workspace/admin/directory/v1/guides/manage-group-members)), 전파가 비중첩보다 느림([Access change propagation](https://docs.cloud.google.com/iam/docs/access-change-propagation)). 처음부터 **깊이 2 로 고정**하면 재귀 코드가 사라진다 |

**반대로 반드시 복사할 것 4가지**

1. **개인이 아니라 그룹에 부여** — *"Grant roles to groups instead of individual users when possible"* ([Use IAM securely](https://docs.cloud.google.com/iam/docs/using-iam-securely))
2. **삭제 전에 권한 회수 + 유예기간** — *"revoke all IAM roles from the group, then wait at least 7 days before deleting it"* ([Google groups in the console](https://docs.cloud.google.com/iam/docs/groups-in-cloud-console))
3. **그룹은 자체 관리자(admin)를 가진다** — OWNER/MANAGER 모델 ([Manage group members](https://developers.google.com/workspace/admin/directory/v1/guides/manage-group-members)). HR 피드가 없을 때의 유일한 출구(§4.4)
4. **최소 스코프 부여** — *"Grant roles at the smallest scope needed"* ([Use IAM securely](https://docs.cloud.google.com/iam/docs/using-iam-securely))

---

### 8. 새로 생긴 열린 질문

[02-architecture.md 의 Q2](../02-architecture.md) 를 대체·세분화한다.

| ID | 질문 | 갈리는 것 |
|---|---|---|
| **Q2a** | 사원 마스터가 **empno → 현재 소속 팀** 매핑을 제공하는가? (조직 코드 유무와 별개로, **소속 정보 자체**가 있는가) | 있으면 `team:*#member` 를 자동 동기화. 없으면 §4.4 의 2단계(`team:*#admin` 수동 관리)로 떨어진다 |
| **Q2b** | 그 매핑을 **어떻게** 얻는가 — API / DB 뷰 / 파일 덤프 / SCIM? | pull 주기 배치 vs push 이벤트. SCIM 이면 `Group` PATCH 를 그대로 소비 가능(§6.2) |
| **Q2c** | 갱신 주기는? 조직 개편은 연 1회지만 **개인 전배는 상시**인가? | 상시면 동기화 주기가 일 단위여야 한다. Google 도 그룹 멤버십 전파에 *"potentially hours"* 를 허용한다 |
| **Q2d** | 마스터가 **과거 소속 이력**을 보관하는가? | 없으면 감사 로그의 "당시 소속" 복원이 불가능 → 우리가 `team:*#member` 튜플의 변경 이력을 직접 남겨야 한다 |
| **Q2e** | 조직 코드가 **있다면** 그것은 팀 개편 시에도 불변인가, 아니면 조직이 개편될 때 새로 발급되는가? | "있다"와 "불변이다"는 다른 문제다. 불변이 아니면 있어도 `externalId` 수준(§6.2)의 참조로만 쓴다 |
| **Q2f** | `team:*#admin`(팀 그룹 관리자)의 **최초 지정** 규칙 — 자가 신청+승인 / 관리자 지정 / 최초 발행자 자동 등록 중 무엇인가? | 1차 자료 근거 없음(§4.4). 운영 결정 필요 |
| **Q15** | 크로스-Biz 게스트의 **기본 만료 기간**은? (30/90/365일) | [02-architecture.md §7.3 경로 C](../02-architecture.md) 의 "만료·재승인" 을 수치화해야 구현 가능 |
| **Q16** | 인가 판정을 **매 요청 DB 조회**로 할 것인가, **캐시 + TTL** 로 할 것인가? | Google 은 그룹 멤버십 전파에 수 시간을 허용한다([Access change propagation](https://docs.cloud.google.com/iam/docs/access-change-propagation)). 우리는 회수(revocation) 지연을 몇 초까지 허용할 것인가 — 이 답이 캐시 설계를 정한다 |

---

## 참고 자료

### Zanzibar (논문)
- [Zanzibar: Google's Consistent, Global Authorization System — USENIX ATC '19 (PDF)](https://www.usenix.org/system/files/atc19-pang.pdf) — §2.1 튜플 문법, §2.2 new enemy·zookie, §2.3.1 userset rewrite, §3.2.3 체크 평가, §3.2.4 Leopard, §4 규모 수치
- [서지·초록 (Google Research)](https://research.google/pubs/zanzibar-googles-consistent-global-authorization-system/)

### Google Cloud IAM
- [IAM overview](https://docs.cloud.google.com/iam/docs/overview) — allow policy 구조, principal 분류
- [Resource hierarchy and access control](https://docs.cloud.google.com/iam/docs/resource-hierarchy-access-control) — 상속의 union 성질
- [Principal identifiers](https://docs.cloud.google.com/iam/docs/principal-identifiers) — 식별자 형식
- [IAM principals](https://docs.cloud.google.com/iam/docs/principals-overview)
- [Roles overview](https://docs.cloud.google.com/iam/docs/roles-overview) — basic/predefined/custom
- [IAM Conditions overview](https://docs.cloud.google.com/iam/docs/conditions-overview) — CEL, 속성 범주, 조건 한도
- [Deny policies overview](https://docs.cloud.google.com/iam/docs/deny-overview) — deny 우선 평가
- [IAM quotas and limits](https://docs.cloud.google.com/iam/quotas) — principal 1,500 / 그룹 250 등
- [Access change propagation](https://docs.cloud.google.com/iam/docs/access-change-propagation) — 전파 지연 수치
- [Use IAM securely](https://docs.cloud.google.com/iam/docs/using-iam-securely) — 그룹 부여 권고, 최소권한
- [Google groups in the Google Cloud console](https://docs.cloud.google.com/iam/docs/groups-in-cloud-console) — 역할 상속, 삭제 경고

### Google Workspace
- [Directory API: Groups resource](https://developers.google.com/workspace/admin/directory/reference/rest/v1/groups) — 불변 `id`
- [Manage groups (Directory API)](https://developers.google.com/workspace/admin/directory/v1/guides/manage-groups) — 이메일을 영속 키로 쓰지 말 것
- [Manage group members (Directory API)](https://developers.google.com/workspace/admin/directory/v1/guides/manage-group-members) — OWNER/MANAGER/MEMBER, 중첩 그룹, 순환 금지, 10분 지연

### Zanzibar 오픈소스 구현
- [SpiceDB — Schema](https://authzed.com/docs/spicedb/concepts/schema) · [Datastores](https://authzed.com/docs/spicedb/concepts/datastores) · [GitHub (Apache-2.0)](https://github.com/authzed/spicedb)
- [OpenFGA — Concepts](https://openfga.dev/docs/concepts) · [Configure OpenFGA](https://openfga.dev/docs/getting-started/setup-openfga/configure-openfga) · [Setup overview](https://openfga.dev/docs/getting-started/setup-openfga/overview) · [GitHub (Apache-2.0, CNCF)](https://github.com/openfga/openfga)
- [Ory Keto — Docs](https://www.ory.com/docs/keto) · [GitHub (Apache-2.0)](https://github.com/ory/keto)

### 인접 표준
- [AWS: Policy evaluation logic](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic.html) · [평가 순서 상세](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic_policy-eval-denyallow.html)
- [RFC 7643 — SCIM Core Schema](https://datatracker.ietf.org/doc/html/rfc7643) — §3.1 `externalId`, §4.1.2 User `groups`(readOnly), §4.2 Group `members`
- [RFC 7644 — SCIM Protocol](https://datatracker.ietf.org/doc/html/rfc7644) — Group PATCH add/remove

### 내부 문서
- [아키텍처 설계서](../02-architecture.md) — §5.2 소유권, §5.3 조직, §6 식별자, §7 인증·인가
