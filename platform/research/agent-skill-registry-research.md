---
tags: [platform, registry, agent-skills, mcp, oci, otel, supply-chain]
level: advanced
last_updated: 2026-09-03
---

# 사내 Agent/Skill 레지스트리 - 1차 자료 조사

> Biz마다 별도 인스턴스로 각각 배포되는 1-tier 레지스트리 N개 + 그 위의 전사 2-tier 레지스트리 구조를 설계하기 전에 패키징 포맷·네임스페이스 소유권·연합(federation)·인스턴스 경계 인증·텔레메트리·공급망 안전을 각 규격 원문에서 확인한 결과.

배포 위상 전제는 확정됐다. 1차 플랫폼은 사업부마다 독립 인스턴스다. 단일 인스턴스 멀티테넌시가 아니다. 2차(전사) 플랫폼은 그 위에 별도로 존재한다. 이 전제 때문에 §8(N개 인스턴스 연합)과 §9(인스턴스 경계 인증)가 이 조사의 무게중심이다.

이 문서는 설계서가 아니라 조사 결과다. 모든 사실 주장에는 출처 URL을 인라인으로 달았다. 1차 자료로 확인하지 못한 항목은 "미확인"으로 명시했다.

---

## 왜 필요한가? (Why)

풀어야 할 문제 4가지가 각각 다른 선행 사례를 가진다.

1. **팀명이 매년 바뀐다.** 패키지 레지스트리에서 네임스페이스를 표시명(display name)에 묶으면 어떤 일이 벌어지는지에 대한 결정적 반례가 npm에 있다. npm은 **organization 이름 변경을 지원하지 않는다** — 새 org를 만들고, 멤버·팀을 옮기고, `package.json`의 스코프를 바꿔 **패키지를 전부 재발행(republish)** 한 뒤, npm Support에 연락해 구 org를 삭제해야 한다 ([npm Docs, Renaming an organization](https://docs.npmjs.com/renaming-an-organization/)). 이유는 단순하다: npm은 스코프를 계정/조직 이름과 **직접 결합**시켰다 — "When you create an npm user account or organization, you are granted a scope that matches your username or organization name" ([npm Docs, About scopes](https://docs.npmjs.com/about-scopes/)). 매년 팀명이 바뀌는 조직에서 이 모델을 쓰면 매년 전 자산을 재발행해야 한다.

2. **2계층 레지스트리 + 승격.** MCP 공식 레지스트리가 "subregistry / aggregator" 개념을 명문화해 두었고, 이게 우리가 만들려는 것과 가장 가까운 선행 사례다 ([MCP Registry Aggregators](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/registry-aggregators.mdx)).

3. **업로드/다운로드/설치가 쉬워야 한다.** Claude Code plugin marketplace가 이미 "사내 마켓플레이스 → `/plugin install`" 경로와 **엔터프라이즈 managed settings 강제 배포**까지 갖고 있다 ([Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)).

4. **인증·인가·모니터링.** MCP 인가 규격이 OAuth 2.1 + RFC 9728/8414/7591/8707 조합을 규범적으로 요구하고 ([MCP Authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)), OTel GenAI semconv가 에이전트 사용량/지연/에러 속성명을 이미 표준화했다 ([OTel GenAI attributes](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md)).

---

## 핵심 개념 (What)

### 1. Anthropic Agent Skills — 실제 저장/배포해야 할 온디스크 포맷

Skill은 디렉터리다. 필수 파일은 `SKILL.md` 하나다. YAML frontmatter 필수 필드는 `name`, `description` 두 개뿐이다 ([Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)).

필드 제약 (원문 그대로):

| 필드 | 제약 |
|---|---|
| `name` | 최대 64자, 소문자·숫자·하이픈만, XML 태그 불가, 예약어 `anthropic`/`claude` 불가 |
| `description` | 비어 있을 수 없음, 최대 1024자, XML 태그 불가. "무엇을 하는지"와 "언제 쓰는지"를 모두 포함해야 함 |

3단계 점진적 공개(progressive disclosure) 구조 — 이게 레지스트리 카탈로그 설계에 직접 영향을 준다:

| 레벨 | 로드 시점 | 토큰 비용 | 내용 |
|---|---|---|---|
| L1 메타데이터 | 항상(시작 시) | Skill당 약 100 토큰 | frontmatter의 `name`, `description` |
| L2 본문 | Skill 트리거 시 | 5k 토큰 미만 | `SKILL.md` 본문 |
| L3+ 리소스 | 필요 시 | 접근 전까지 0 | 번들 파일 (`FORMS.md`, `scripts/*.py` 등) |

출처: [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).

레지스트리 설계로 옮기면, 카탈로그 검색 인덱스에 넣어야 하는 것은 L1(`name`+`description`)과 L2(본문)뿐이다. L3 번들은 오브젝트 스토리지에 두고 인덱싱하지 않아도 된다.

디렉터리 위치: Claude Code는 `~/.claude/skills/`(개인) 또는 `.claude/skills/`(프로젝트)에서 파일시스템 기반으로 자동 발견한다. **API·claude.ai·Claude Code 간 Skill은 동기화되지 않는다.** 각 surface마다 별도 업로드가 필요하다 ([같은 문서, Cross-surface availability](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)). API 요청당 Skill 최대 20개 ([Skills for enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise)).

### 2. Claude Code Plugin / Marketplace 매니페스트

Skill 하나보다 큰 배포 단위는 plugin이다. `.claude-plugin/plugin.json`의 필수 필드는 `name`(kebab-case) 하나뿐이다. 나머지는 메타데이터(`displayName`, `version`, `description`, `author{name,email,url}`, `homepage`, `repository`, `license`, `keywords`, `defaultEnabled`)와 컴포넌트 경로 필드(`skills`, `commands`, `agents`, `hooks`, `mcpServers`, `lspServers`, `outputStyles`, `workflows`), 그리고 `userConfig`·`dependencies`·`channels`다 ([Plugins reference](https://code.claude.com/docs/en/plugins-reference)).

디렉터리 레이아웃 (원문):

```
plugin-name/
├── .claude-plugin/plugin.json
├── skills/            # <name>/SKILL.md
├── agents/            # subagent .md
├── commands/          # flat .md
├── hooks/hooks.json
├── .mcp.json          # MCP 서버 정의
├── scripts/  bin/
```
컴포넌트 디렉터리는 plugin 루트에 있어야 하며 `.claude-plugin/` 안에 두면 안 된다 ([Plugins reference](https://code.claude.com/docs/en/plugins-reference)).

마켓플레이스 카탈로그는 저장소 루트의 `.claude-plugin/marketplace.json`이며, 필수 필드는 `name`, `owner`, `plugins[]`. 각 plugin 엔트리는 `name`과 `source`가 필수다. source 타입은 상대경로, `github`(`repo`/`ref`/`sha`), `url`(git URL), `git-subdir`(모노레포, `path`), `npm`(`package`/`version`/`registry` — 사내 npm 레지스트리 지정 가능), `archive`(`url` + `sha256`), `command`이다 ([Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)).

엔터프라이즈 강제 배포(managed settings)가 "사원이 쉽게 설치"를 해결하는 기성 경로다:

```json
{ "extraKnownMarketplaces": { "company-tools": { "source": { "source": "github", "repo": "your-org/claude-plugins" } } },
  "strictKnownMarketplaces": [ { "source": "hostPattern", "hostPattern": "^github\\.example\\.com$" } ],
  "enabledPlugins": { "code-formatter@company-tools": true } }
```
`strictKnownMarketplaces`로 사내 호스트만 허용할 수 있다 ([Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)). 조직 설정 배포 경로는 `github`, `url`, `git-subdir`, 상대경로 소스로 제한된다(같은 문서).

> 미확인: 사내 GitHub Enterprise 호스트에 대한 `source: "github"` 동작(호스트 오버라이드 방법)은 위 문서에서 확인하지 못했다. `url`/`git-subdir` + `hostPattern` 조합이 문서상 확인된 경로다.

### 3. MCP Registry — 우리가 만들 것의 가장 가까운 선행 사례

#### 3.1 데이터 모델 (`server.json`)

스키마 원본: [`docs/reference/server-json/draft/server.schema.json`](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/draft/server.schema.json) (`$ref: #/definitions/ServerDetail`, JSON Schema draft-07).

`ServerDetail` **required: `name`, `description`, `version`**.

| 필드 | 타입 | 제약 / 설명 |
|---|---|---|
| `name` | string | **reverse-DNS**. `pattern: ^[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+$`, maxLength 200. 슬래시 정확히 1개로 namespace와 server name 분리 |
| `description` | string | maxLength **100** |
| `version` | string | maxLength 255, semver 권장 |
| `title` | string | maxLength 100. **선택적 human-readable 표시명** |
| `repository` | object | required `url`, `source`; 선택 `id`(호스팅 서비스의 repo ID), `subfolder`(모노레포) |
| `packages[]` | array | 아래 참조 |
| `remotes[]` | array | `streamable-http` / `sse`, `url`(URL 템플릿 `{tenant_id}` 지원), `headers[]` |
| `websiteUrl`, `icons[]`, `$schema`, `_meta` | | `_meta`는 reverse-DNS 네임스페이싱 확장 |

`Package` required: `registryType`, `identifier`, `transport`. 그 외 `registryBaseUrl`, `version`(범위 표기 `^1.2.3` 등은 거부), `runtimeHint`, `runtimeArguments[]`, `packageArguments[]`, `environmentVariables[]`, `fileSha256`(MCPB 필수, 그 외 선택 — 무결성 검증용).

`registryType` 값 예: `npm`, `pypi`, `nuget`, `cargo`, `oci`, `mcpb` ([generic-server-json.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/generic-server-json.md)).

레지스트리는 메타데이터만 호스팅하고 아티팩트는 호스팅하지 않는다. "The MCP Registry only hosts metadata, not artifacts" ([quickstart](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx)). 우리는 MinIO가 있으므로 여기서 **의도적으로 갈라져야 한다**(아티팩트도 호스팅). 그래도 `packages[].identifier` + `fileSha256` 구조는 그대로 재사용 가능하다.

#### 3.2 레지스트리 관리 메타데이터 (`_meta`)

응답의 `_meta["io.modelcontextprotocol.registry/official"]` 객체 ([openapi.yaml](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/openapi.yaml)):

| 필드 | 타입 | 설명 |
|---|---|---|
| `status` | enum | `active` / `deprecated` / `deleted` |
| `statusMessage` | string | maxLength 500 |
| `statusChangedAt` | date-time | |
| `publishedAt` | date-time | 최초 발행 |
| `updatedAt` | date-time | 마지막 갱신 |
| `isLatest` | boolean | |

발행자 확장은 `_meta["io.modelcontextprotocol.registry/publisher-provided"]` 키에만 허용되며 4KB(4096바이트) 제한, 다른 키는 발행 시 조용히 버려진다 ([official-registry-requirements.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)).

subregistry는 자기 키(`com.example.subregistry/custom`)로 사용자 평점·다운로드 수·보안 스캔 결과를 주입하도록 권장된다 ([registry-aggregators.mdx](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/registry-aggregators.mdx)). 사내 승격 점수·심사 결과를 넣을 자리가 이미 규격 안에 있다.

#### 3.3 API 표면

Generic Registry API(`/v0.1`)에서 subregistry가 구현해야 할 최소 집합 ([generic-registry-api.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/generic-registry-api.md)):

| 메서드 | 경로 | 필수/선택 |
|---|---|---|
| GET | `/v0.1/servers` | 코어. 커서 페이지네이션 |
| GET | `/v0.1/servers/{serverName}/versions` | 코어 |
| GET | `/v0.1/servers/{serverName}/versions/{version}` | 코어. `version=latest` 특수값 |
| POST | `/v0.1/publish` | 선택(레지스트리별 인증) |
| PUT | `/v0.1/servers/{serverName}/versions/{version}` | 선택(공식은 admin 전용) |
| DELETE | `/v0.1/servers/{serverName}/versions/{version}` | 선택(공식 미구현) |
| PATCH | `/v0.1/servers/{serverName}/versions/{version}/status` | 선택 |
| PATCH | `/v0.1/servers/{serverName}/status` | 선택(전 버전 일괄, 단일 트랜잭션) |

공식 레지스트리 추가 사항 ([official-registry-api.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/official-registry-api.md)):
- 목록 필터: `updated_since`(RFC3339), `search`(서버명 대소문자 무시 부분문자열), `version=latest`, `include_deleted`. 문서 원문: *"This is intentionally simple. For more advanced searching and filtering, use a subregistry."* → **전문 검색은 subregistry(=우리)의 몫이라고 규격이 명시한다.**
- 인증 엔드포인트: `POST /v0.1/auth/dns`, `/auth/http`, `/auth/github-at`, `/auth/github-oidc`, `/auth/oidc`
- `POST /v0.1/validate` (발행 없이 `server.json` 검증), `GET /v0.1/ping`, `/v0.1/version`, `/v0.1/health`, `GET /metrics`(Prometheus)
- 경로 파라미터는 **URL 인코딩 필수** (`io.modelcontextprotocol%2Feverything`)

확장 규약은 `/v0.1/x/<namespace>/<extension>[/<path>]`다. reverse-DNS 네임스페이스로 커스텀 엔드포인트를 붙인다. 예: `/v0.1/x/com.example/search?q=database` ([extensions.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/extensions.md)). **사내 승격(promotion) API를 여기에 얹는 것이 규격 정합적이다.**

#### 3.4 네임스페이스 소유권 / 검증 모델 ★ (rename 문제의 핵심)

인증 방식이 곧 네임스페이스를 결정한다 ([authentication.mdx](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/authentication.mdx)):

| 인증 | 이름 형식 | 예시 |
|---|---|---|
| GitHub 기반 | `io.github.username/*` 또는 `io.github.orgname/*` | `io.github.alice/weather-server` |
| 도메인 기반 | `com.example.*/*` (도메인의 reverse-DNS) | `io.modelcontextprotocol/everything` |

- DNS 검증: 도메인 **apex**에 TXT 레코드 `v=MCPv1; k=ed25519; p=<base64 pubkey>` (Ed25519 또는 ECDSA P-384). 셀렉터(`_mcp-auth.example.com`) 아래 두면 실패한다 — SPF 스타일 배치이지 DKIM 스타일이 아니다. 키 회전 시 기존 레코드 제거 필수.
- GitHub org 네임스페이스는 **org Owner만** 발행 가능. 단순 멤버십으로는 불충분(2025년 변경). 클래식 PAT는 `read:org`, fine-grained PAT는 Organization → Members → Read-only 권한 필요. 저장소 스코프는 불필요.
- 패키지 소유권 검증: npm이면 `package.json`에 `"mcpName": "io.github.my-username/weather"`를 넣어 레지스트리가 메타데이터와 실제 패키지의 일치를 확인한다 ([quickstart.mdx](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx)).

팀 rename 문제를 두고 이 자료들이 실제로 말해주는 것:

- MCP는 `name`(불변 식별자, reverse-DNS)과 `title`("Optional human-readable title or display name... MAY choose to use this for display purposes")을 **스키마 수준에서 분리**해 두었다 ([server.schema.json](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/draft/server.schema.json)). → 우리도 팀에 대해 `team_id`(불변) + `team_display_name`(가변) 분리를 하면 된다.
- 반대로 npm은 스코프=조직명이라서 rename이 불가능하고 전량 재발행을 요구한다 ([npm renaming an organization](https://docs.npmjs.com/renaming-an-organization/)). **팀명을 네임스페이스에 넣지 말라는 강한 반례.**
- PyPI는 조직 롤(Owner / Manager / Member / Billing Manager)과 프로젝트 롤(Owner / Maintainer)을 분리한다. Project Owner는 "can manage the project and collaborators for the project", Maintainer는 "can upload releases for a project" ([PyPI roles & entities](https://docs.pypi.org/organization-accounts/roles-entities/)). **소유권 = 이름이 아니라 롤 배정**이라는 모델.
- **미확인:** PyPI 프로젝트명 불변성 및 소유권 이전(transfer) 절차의 공식 문서는 위 페이지에서 확인하지 못했다. 설계 근거로 쓰려면 별도 확인 필요.
- **미확인:** MCP Registry에 "네임스페이스 소유권 이전(transfer)" 기능이 있는지는 문서에서 확인하지 못했다. 확인된 것은 발행 시점의 소유권 *증명*(DNS/GitHub)뿐이다. 사내 이전 워크플로는 우리가 직접 설계해야 한다.

#### 3.5 2-tier / 연합 메커니즘 — 실제로 규격이 보장하는 것

MCP 레지스트리가 명시하는 downstream 동기화 방식은 "폴링 기반 스크래핑"이다. 원문: aggregator는 "expected to scrape data on a regular but infrequent basis (e.g., once per hour), and persist the data in their own data store"이며 상위 레지스트리는 "does not provide uptime or data durability guarantees" ([registry-aggregators.mdx](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/registry-aggregators.mdx)).

증분 동기화 메커니즘:
- `GET /v0.1/servers?updated_since=<RFC3339>` — `updated_since`가 주어지면 `include_deleted`가 자동 `true`가 되어 삭제분까지 따라온다 ([official-registry-api.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/official-registry-api.md)).
- 서버 메타데이터는 `status` 필드를 빼면 사실상 **불변(immutable)**. aggregator는 `status`만 최신으로 유지하도록 권고된다.
- "subregistry"는 aggregator이면서 동일한 OpenAPI 스펙을 구현하는 것 — 클라이언트가 표준 인터페이스로 소비할 수 있게 한다.

**즉 "1-tier → 2-tier 승격"의 규격상 실체는 push가 아니라 pull이다.** 상위(전사)가 하위(Biz)를 폴링해 당겨가거나 하위가 상위의 `POST /v0.1/publish`를 호출하거나 둘 중 하나를 우리가 정해야 한다. MCP 문서는 전자만 표준화했다.

OCI 쪽도 확인했다. OCI distribution spec은 미러링/연합을 정의하지 않는다. 정의하는 것은 프록시 위임뿐이다: "A registry MAY operate as a proxy to another registry to delegate functionality... An example of delegating functionality is proxying pull operations to another registry" ([OCI distribution-spec](https://github.com/opencontainers/distribution-spec/blob/main/spec.md)). 정의된 엔드포인트는 pull(`GET /v2/<name>/blobs|manifests/...`), push(`POST/PATCH/PUT .../blobs/uploads/`, `PUT .../manifests/...`), content discovery(`GET /v2/<name>/tags/list`, `GET /v2/<name>/referrers/<digest>`; `artifactType` 필터 지원), content management(`DELETE ...`), 그리고 capability check `GET /v2/`.

> Artifactory·Nexus의 remote/proxy·virtual/group 메커니즘은 별도로 확인했다 → §8 참조.

### 4. OCI artifact로 Skill 번들을 배포하는 선택지

임의 아티팩트를 OCI 레지스트리에 담는 방법은 image-spec의 "Guidelines for Artifact Usage"에 정의돼 있다 ([manifest.md](https://github.com/opencontainers/image-spec/blob/main/manifest.md)):

- `artifactType` 필드가 아티팩트 타입을 식별한다 (예: `application/vnd.skynix.skill.v1+json` 같은 사내 미디어 타입).
- config가 없으면 **empty descriptor**를 쓴다: mediaType `application/vnd.oci.empty.v1+json`, digest `sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`, size 2.
- `config.mediaType`은 "MUST be set to a value specific to the artifact type or the empty value"이며, 알려진 OCI image config 미디어 타입이면 안 된다 ([artifacts-guidance.md](https://github.com/opencontainers/image-spec/blob/main/artifacts-guidance.md)).
- 선택적 `subject` 필드로 다른 manifest에 매니페스트를 연결하고, referrers API(`GET /v2/<name>/referrers/<digest>`)로 조회한다 → **서명·SBOM·심사 결과를 아티팩트에 붙이는 표준 경로**.

단, 허용 스택에 OCI 레지스트리(Harbor/Distribution)가 없다. MinIO만 있으므로 OCI 경로를 쓰려면 컴포넌트를 추가해야 한다. 이건 열린 질문으로 아래에 옮긴다.

### 5. 인증 / 인가

#### MCP Authorization (에이전트가 사용자 대신 서비스를 호출할 때)

규범 요구사항 ([MCP Authorization spec 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)):

- 기반 규격: OAuth 2.1 (draft-ietf-oauth-v2-1-13), **RFC 8414**(AS metadata), **RFC 7591**(DCR), **RFC 9728**(Protected Resource Metadata).
- MCP 서버는 RFC 9728을 **MUST** 구현하고, PRM 문서에 `authorization_servers` 필드를 **MUST** 포함. 401 응답 시 `WWW-Authenticate` 헤더로 resource metadata URL을 **MUST** 알린다.
- 클라이언트는 **RFC 8707 Resource Indicators**의 `resource` 파라미터를 authorization/token 요청 **양쪽에 MUST 포함**하고, MCP 서버의 canonical URI를 지정해야 한다. AS가 지원하든 안 하든 **MUST** 보낸다.
- 서버는 토큰이 **자신을 audience로 발급된 것인지 MUST 검증**한다. **token passthrough는 명시적으로 금지** — 상류 API 호출 시 받은 토큰을 그대로 전달하면 안 되고(MUST NOT), 별도 토큰을 써야 한다. 근거로 RFC 9068의 audience claim을 인용한다.
- PKCE **MUST**, redirect URI 사전 등록 및 정확 일치 검증 **MUST**, 모든 AS 엔드포인트 HTTPS **MUST**, 단명 액세스 토큰 **SHOULD**, public client는 refresh token rotation **MUST**.
- STDIO 전송은 이 스펙을 따르지 **SHOULD NOT** — 환경변수에서 자격증명을 가져온다.
- 에러 코드: 401(인증 필요/토큰 무효), 403(스코프 부족), 400(요청 오류).

레지스트리 자신의 인가에 대해서도 MCP가 권고안을 낸다 ([registry-authorization.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/registry-authorization.md)). 레지스트리는 OAuth 2.1 Resource Server로 동작하고 스코프는 `mcp-registry:read` / `mcp-registry:write`를 쓸 수 있다. 결정적인 문장은 이것이다: *"scopes only control what types of operations a user can perform. Registries should still apply user-level authorization to control which specific resources a user can access. For example, a user with `mcp-registry:write` might only be able to publish servers to namespaces they own."*

스코프(RBAC)만으로는 부족하다. **리소스 단위 인가가 별도로 필요하다는 것이 규격의 명시적 입장이다.**

#### M2M 토큰

RFC 9068 (JWT Profile for OAuth 2.0 Access Tokens) — 헤더 `typ`는 `at+jwt`(권장) 또는 `application/at+jwt`. 필수 클레임은 `iss`, `exp`, `aud`, `sub`, `client_id`, `iat`, `jti`. 선택: `auth_time`, `acr`, `amr`, `scope`, `groups`, `roles`, `entitlements` ([RFC 9068](https://www.rfc-editor.org/rfc/rfc9068.html)).

`groups`/`roles`/`entitlements`가 표준 선택 클레임으로 이미 있으므로 사내 SSO IdP에서 불변 team_id를 이 클레임에 실어 보내는 것이 규격 정합적이다.

#### RBAC vs ReBAC — 권고

코어는 RBAC(역할 배정)으로, 소유권·공유는 ReBAC(관계 튜플)로 권고한다.

근거:
- RBAC 쪽: PyPI가 조직 롤 4종 + 프로젝트 롤 2종으로 충분히 운영된다 ([PyPI roles & entities](https://docs.pypi.org/organization-accounts/roles-entities/)). 대부분의 레지스트리 동작(read/publish/deprecate)은 역할 몇 개로 표현된다.
- ReBAC 쪽: Zanzibar는 "a uniform data model and configuration language for expressing a wide range of access control policies"를 제공하며, 결정이 "respect causal ordering of user actions and thus provide external consistency amid changes to access control lists"하고, 조 단위 ACL과 초당 수백만 요청을 p95 <10ms, 가용성 >99.999%로 처리한다 ([Zanzibar, USENIX ATC '19](https://research.google/pubs/zanzibar-googles-consistent-global-authorization-system/)).
- MCP registry-authorization이 요구하는 "네임스페이스 소유자만 발행 가능"은 정확히 **관계** 판정이다.

우리 규모(한 Biz)에서 Zanzibar급 시스템을 세우는 건 과하다. 그러나 **`(subject, relation, object)` 튜플 형태로 소유권을 저장하는 데이터 모델 자체는 채택할 가치가 있다.** `team:T-0417 #owner agent:foo`처럼 저장하면 팀명이 바뀌어도 튜플이 그대로 유지된다.

### 6. 텔레메트리 — OTel GenAI semantic conventions

GenAI semconv는 `opentelemetry.io/docs/specs/semconv/gen-ai/`에서 별도 저장소로 이전되었다 ([이전 안내](https://opentelemetry.io/docs/specs/semconv/gen-ai/)). 현행 원본은 [open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai).

대시보드에 실제로 필요한 속성 ([registry/attributes/gen-ai.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md)):

| 축 | 속성 |
|---|---|
| 에이전트 신원 | `gen_ai.agent.id` (unique and stable identifier), `gen_ai.agent.name`, `gen_ai.agent.version`, `gen_ai.agent.description` |
| 작업 | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.conversation.id`, `gen_ai.workflow.name` |
| 모델 | `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.request.temperature`, `gen_ai.request.max_tokens`, `gen_ai.request.stream` |
| 사용량(=비용) | `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.reasoning.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens`, `gen_ai.token.type` |
| 도구 | `gen_ai.tool.name`, `gen_ai.tool.type`, `gen_ai.tool.call.id`, `gen_ai.tool.definitions` |
| 결과/에러 | `gen_ai.response.finish_reasons`, `gen_ai.response.status`, `gen_ai.response.id` |
| 지연 | `gen_ai.response.time_to_first_chunk` |
| 평가 | `gen_ai.evaluation.name`, `gen_ai.evaluation.score.value`, `gen_ai.evaluation.score.label` |

메트릭 ([gen-ai-metrics.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md)):

| 메트릭 | 계기 | 단위 | 필수 속성 |
|---|---|---|---|
| `gen_ai.client.token.usage` | Histogram | `{token}` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.token.type` |
| `gen_ai.client.operation.duration` | Histogram | `s` | `gen_ai.operation.name` |
| `gen_ai.client.operation.time_to_first_chunk` | Histogram | `s` | 스트리밍 시에만 보고 |
| `gen_ai.client.operation.time_per_output_chunk` | Histogram | `s` | 스트리밍 시에만 보고 |
| `gen_ai.server.request.duration` / `time_to_first_token` / `time_per_output_token` | Histogram | `s` | 서버 측 |
| `gen_ai.invoke_agent.duration` | Histogram | `s` | 에이전트 |
| `gen_ai.invoke_agent.inference_calls` / `.tool_calls` | Counter | | 에이전트 |
| `gen_ai.execute_tool.duration` | Histogram | `s` | |
| `gen_ai.invoke_workflow.duration` | Histogram | `s` | |

스팬 이름 규칙: 추론은 `{gen_ai.operation.name} {gen_ai.request.model}`, 검색은 `{gen_ai.operation.name} {gen_ai.data_source.id}` ([gen-ai-spans.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md), 검색 결과 요약을 통해 확인).

레지스트리 관점에서 보면 `gen_ai.agent.id`가 "unique and stable identifier"로 규정돼 있다. 레지스트리가 발급하는 불변 agent ID를 그대로 이 속성에 넣으면 텔레메트리와 카탈로그가 자연 조인된다. 팀 소유권은 `gen_ai.*`에 표준 속성이 없으므로 사내 커스텀 속성(예: `skynix.team.id`)이 필요하다. **표준에 팀/소유자 속성은 없다(미확인이 아니라 부재 확인).**

#### OpenSearch가 텔레메트리 저장소로 맞는가 — 예

OpenSearch Data Prepper가 OTLP를 직접 수집한다 ([OpenSearch Trace Analytics](https://docs.opensearch.org/latest/data-prepper/common-use-cases/trace-analytics/)):
- 소스 플러그인: `otel_trace_source`(OpenTelemetry Collector로부터 trace 수신, gRPC + TLS/HTTPS 지원), `otel-logs-source`, `otel-metrics-source`
- 프로세서: `otel_traces`(span 레코드 처리, trace-group 필드 stateful 추출), `otel_traces_group`(누락 필드를 OpenSearch 질의로 보충), `service_map`(서비스 맵 메타데이터)
- 인덱스: `otel-v1-apm-span`, `otel-v1-apm-service-map`
- Data Prepper 2.0에서 레거시 `otel_traces_prepper`/`otel_traces_group_prepper`는 제거됨

보존 관리: Index State Management(ISM)로 정책 기반 rollover / delete / index_priority 전이를 구성한다 ([OpenSearch ISM](https://docs.opensearch.org/latest/im-plugin/ism/index/)).

### 7. 공급망 안전

Skill은 실행 가능한 프롬프트 + 스크립트다. Anthropic 자신의 경고 (원문): *"Use Skills only from trusted sources... a malicious Skill can direct Claude to invoke tools or execute code in ways that don't match the Skill's stated purpose"*, 그리고 *"Treat like installing software"* ([Agent Skills overview, Security considerations](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)).

엔터프라이즈 심사 체크리스트가 이미 문서화돼 있고 사내 심사 게이트 설계에 그대로 쓸 수 있다 ([Skills for enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise)):

리스크 지표 (High): 코드 실행(`*.py`/`*.sh`/`*.js`), 지시 조작(안전 규칙 무시·행동 은폐 지시), MCP 서버 참조, 네트워크 접근 패턴(URL/`fetch`/`curl`/`requests`), 하드코딩된 자격증명. (Medium): 파일시스템 접근 범위(`../`, 광범위 glob), 도구 호출.

프로세스 요구사항 (원문 요지):
- **"Establish separation of duties: Skill authors should not be their own reviewers."**
- Skill당 대표 질의 3–5개의 evaluation suite 제출 요구, 사용 모델 전반에서 테스트
- 프로덕션은 **버전 핀 고정**. `version`을 생략하면 최신을 쓰므로 누군가의 업로드가 즉시 프로덕션을 바꾼다
- **"Integrity verification: Compute checksums of reviewed Skills and verify them at deployment time. Use signed commits in your Skill repository to ensure provenance."**
- 사내 registry에 기록할 필드: Purpose / Owner / Version / Dependencies / Evaluation status
- **"Usage analytics are not currently available through the Skills API. Implement application-level logging."** → 우리가 텔레메트리 계층을 직접 만들어야 하는 근거

서명/증명 기술 (1차 자료 확인분):
- **cosign 자체 키 서명 가능** — `cosign generate-key-pair`(RSA/ECDSA/ED25519 지원), `cosign sign --key cosign.key <ref>`. 레지스트리에 서명을 올리지 않으려면 `cosign sign --bundle=bundle.sigstore.json --upload=false $IMAGE` ([Sigstore: Signing with Self-Managed Keys](https://docs.sigstore.dev/cosign/key_management/signing_with_self-managed_keys/), [Key Management](https://docs.sigstore.dev/cosign/key_management/)). → **외부 Fulcio/Rekor 없이도 폐쇄망에서 쓸 수 있다.**
- **in-toto Attestation Statement v1**: `_type` = `https://in-toto.io/Statement/v1`(필수), `subject`(필수, ResourceDescriptor 배열 — `name` + `digest`), `predicateType`(필수, predicate 타입 URI), `predicate`(선택) ([in-toto statement spec](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md)).
- **SLSA Provenance v1**: `predicateType` = `https://slsa.dev/provenance/v1`. `buildDefinition{buildType, externalParameters(SLSA Build L3에서 완전해야 함), internalParameters, resolvedDependencies}` + `runDetails{builder.id, metadata{invocationId, startedOn, finishedOn}, byproducts}` ([SLSA v1.0 Provenance](https://slsa.dev/spec/v1.0/provenance)).
- **미확인:** DSSE envelope의 `payloadType` 값은 위 statement 문서에서 확인하지 못했다 (별도 DSSE 규격 문서 필요).

---

### 8. N개 독립 인스턴스 + 상위 인스턴스 — 실제 메커니즘 ★

배포 전제가 "Biz마다 별도 인스턴스"이므로 핵심 질문 3개를 1차 자료로 확인했다: (a) 아티팩트를 복사하는지 참조만 하는지, (b) 네임스페이스 충돌을 어떻게 막는지, (c) 승격(publish upstream)이 push인가 pull인가.

#### 8.1 네 가지 선행 사례의 메커니즘 비교

| | **MCP Registry (subregistry)** | **OCI distribution** | **Artifactory (remote/virtual)** | **Nexus (proxy/group)** |
|---|---|---|---|---|
| 하위 인스턴스가 상위 데이터를 얻는 방법 | **하위가 상위를 폴링**해 스크래핑. `GET /v0.1/servers?updated_since=` | 레지스트리가 다른 레지스트리로 **pull 위임(proxy)** | **remote repository** = 캐싱 프록시 | **proxy repository** = "a substitute access point and managed cache for remote repositories" |
| 아티팩트 복사 여부 | **복사 안 함.** 레지스트리는 메타데이터만 호스팅 — "The MCP Registry only hosts metadata, not artifacts" | 프록시 구현에 달림(스펙 미정의) | **on-demand 복사.** "Artifacts are **not** pre-fetched... They are only fetched (pulled) and stored (cached) *on demand* when requested by a client" | **on-demand 복사.** 없으면 "the request is forwarded to the remote repository, downloaded, and cached to the local storage" |
| 통합 URL | subregistry가 동일 OpenAPI를 구현 → 클라이언트는 한 엔드포인트만 봄 | 없음 | **virtual repository** = local+remote를 한 URL로 집계 | **group repository** = hosted+proxy를 한 URL로 병합 |
| 충돌 해소 | 문서화된 규칙 없음(reverse-DNS 네임스페이스로 애초에 충돌 방지) | 스펙 미정의 | 저장소 순서 | **순서 기반.** "Keep the local hosted repositories first. Searching the hosted repositories is far quicker and keeps you from looking for your internal components in external proxies" |
| 오프라인/단절 | 하위가 자체 데이터스토어에 영속화 — 상위는 "does not provide uptime or data durability guarantees" | — | **Offline 체크박스**: "only artifacts from this repository that are already present in the cache are used. No further attempt will be made to fetch remote artifacts". 전역 offline mode도 존재 | 캐시가 외부 장애에 대한 복원력 제공 |

출처: [MCP registry-aggregators](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/registry-aggregators.mdx), [MCP quickstart](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx), [OCI distribution-spec](https://github.com/opencontainers/distribution-spec/blob/main/spec.md), [Artifactory remote repositories](https://docs.jfrog.com/artifactory/docs/remote-repositories), [Sonatype Nexus repository concepts](https://help.sonatype.com/en/repository-manager-concepts.html).

#### 8.2 세 질문에 대한 답

**(a) 아티팩트는 복사되는가?**

두 갈래가 뚜렷하다.

- **메타데이터 전용 연합 (MCP)**: 아티팩트는 어디에도 복사되지 않는다. downstream은 `server.json`만 복제하고 실제 패키지는 원 레지스트리(npm/PyPI/OCI)에서 클라이언트가 직접 받는다. 무결성은 `packages[].fileSha256`으로 보장. 장점: 저장소 비용 0, 동기화가 JSON 복사. **단점: 원본 인스턴스가 죽으면 설치가 깨진다.**
- **아티팩트 캐싱 연합 (Artifactory/Nexus)**: **lazy copy** — 사전 미러링이 아니라 **최초 요청 시점에 당겨서 캐시**한다. 이후 요청은 캐시에서 나간다. Artifactory는 캐시 전용 URL(`.../<remote-repo-name>-cache/<path>`)까지 별도로 노출한다.

우리 상황에서는 Biz 인스턴스가 각각 자체 MinIO를 가지므로 전사(2-tier)로의 승격은 *어느 쪽이든 될 수 있다*. 그러나 두 선행 사례 모두 **"eager full mirror"를 하지 않는다**는 점이 일치한다. Artifactory는 명시적으로 pre-fetch를 부정하고 MCP는 아예 복사하지 않는다. 사전 전량 복제는 어느 1차 자료도 지지하지 않는다.

**(b) 네임스페이스 충돌 방지**

- MCP의 답은 **reverse-DNS 네임스페이스 + 소유권 증명**이다. `name`이 `^[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+$`(슬래시 1개)로 강제되고, 네임스페이스는 DNS TXT 또는 GitHub org 소유 증명으로만 얻는다 ([authentication.mdx](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/authentication.mdx)). **인스턴스가 N개라도 도메인이 다르면 이름은 구조적으로 충돌할 수 없다.** → `kr.co.skhynix.<biz>.<team_id>/<name>` 형태면 Biz 접두가 충돌을 원천 차단한다.
- Nexus/Artifactory의 답은 **순서 기반 해소(shadowing)** 다 — 충돌을 막는 게 아니라 우선순위로 가린다. 사내에서는 "로컬 hosted 먼저"가 권고 순서다.
- Claude Code marketplace의 답은 **설치 시점 네임 한정**이다: `/plugin install plugin-name@marketplace-name`. plugin 이름이 같아도 마켓플레이스 이름으로 구분된다 ([Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)). 또한 `renames` 필드로 구 plugin 이름 → 신 이름 매핑을 카탈로그가 제공한다.

reverse-DNS(구조적 차단)를 1차로 하고 클라이언트 표면에서는 `@marketplace` 한정자(순서 무관)를 2차로 두기를 권고한다. Nexus식 순서 기반 shadowing은 **채택하지 말 것.** 어느 인스턴스의 것이 실행됐는지 사후 추적이 불가능해져 §6 텔레메트리와 충돌한다.

(c) 승격 방향

1차 자료가 표준화한 것은 pull뿐이다. MCP는 aggregator가 "expected to scrape data on a regular but infrequent basis (e.g., once per hour), and persist the data in their own data store"라고만 규정한다. Artifactory/Nexus의 remote/proxy도 전부 pull이다. OCI는 프록시 pull 위임만 언급하고 push federation은 명시적으로 범위 밖이다.

**즉 "downstream이 upstream으로 밀어 올린다(publish upstream)"는 동작을 표준화한 1차 자료는 이번 조사에서 찾지 못했다.** 존재하는 것은:
- MCP의 `POST /v0.1/publish` — 그러나 이건 "발행자가 레지스트리에 올린다"이지 "레지스트리가 레지스트리에 올린다"가 아니다. 다만 **Biz 인스턴스를 전사 레지스트리 관점의 '발행자 클라이언트'로 취급하면 그대로 재사용 가능**하다. 이때 Biz 인스턴스가 전사에 대해 네임스페이스 소유권을 증명하는 주체가 된다(§9 참조).
- MCP 확장 규약 `/v0.1/x/<namespace>/<extension>` — 승격 API를 규격 정합적으로 얹을 자리 ([extensions.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/extensions.md)).

현실적인 조합은 이렇다. 승격 *결정*은 push(Biz가 신청 → 전사가 승인), 승격 *후 데이터 동기화*는 pull(전사가 `updated_since`로 폴링)이다. 전자만 우리가 새로 만들면 되고 후자는 MCP aggregator 패턴을 그대로 쓴다.

#### 8.3 Claude Code marketplace가 이미 제공하는 다중 인스턴스 신뢰 모델 ★

여기에 우리 구조와 정확히 대응되는 1차 자료가 있다 ([Constrain plugin dependency versions](https://code.claude.com/docs/en/plugin-dependencies)):

- 기본적으로 **"Claude Code refuses to auto-install a dependency that lives in a different marketplace than the plugin declaring it. This prevents one marketplace from silently pulling in plugins from a source you have not reviewed."** → 인스턴스 간 기본값은 **거부**.
- 허용하려면 root marketplace의 `marketplace.json`에 `allowCrossMarketplaceDependenciesOn: ["acme-shared"]`를 명시한다. 그리고 결정적인 문장: **"only its allowlist is consulted, so trust does not chain through intermediate marketplaces."** → **신뢰가 전이(transitive)되지 않는다.** Biz A가 Biz B를 신뢰해도, B가 신뢰하는 C까지 자동으로 신뢰되지 않는다.
- 의존성 엔트리에 `marketplace` 필드로 다른 인스턴스를 명시적으로 지목한다: `{ "name": "audit-logger", "marketplace": "acme-shared" }`.
- 사용자가 그 의존성을 **수동으로 먼저 설치하면** allowlist 없이도 제약이 충족된다 — 관리자 정책과 사용자 명시 동의가 별개 경로다.

이건 우리가 설계할 "Biz A 사용자가 Biz B 에이전트를 쓴다"의 클라이언트 측 정책 모델 그대로다. 명시적 allowlist와 비전이 신뢰(non-transitive trust)를 채택하도록 권고한다.

---

### 9. 인스턴스 경계를 넘는 인증 (cross-instance / cross-registry trust) ★

Biz A 사용자가 Biz B 인스턴스의 에이전트를 다운로드/실행한다는 것이 문제다. 인스턴스가 독립 배포이므로 각각이 별도의 OAuth Resource Server다.

#### 9.1 표준 경로 — 공통 IdP를 두고 각 인스턴스를 별개 RP/RS로

OIDC Core가 보장하는 것 ([OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html)):

- `sub`는 **"A locally unique and never reassigned identifier within the Issuer for the End-User"** — 즉 **발급자(Issuer) 범위에서만** 유일하고 재사용되지 않는다. 전역 유일이 아니다. 따라서 사용자를 인스턴스 간에 동일인으로 식별하는 표준 키는 **`iss` + `sub` 쌍**이다.
- `sub`는 255 ASCII 문자를 초과할 수 없고 case-sensitive 문자열이다.
- **Public vs Pairwise(PPID)** 식별자 타입이 있다. PPID는 "an Identifier that identifies the Entity to a Relying Party that cannot be correlated with the Entity's PPID at another Relying Party" — **RP마다 다른 `sub`를 준다.** → **주의: 사내 IdP가 pairwise로 설정돼 있으면 Biz A 인스턴스와 Biz B 인스턴스가 같은 사용자를 다른 `sub`로 보게 되어, 인스턴스 간 사용자 동일성 판정이 깨진다.** 이 구조에서는 **public subject identifier로 설정해야 한다** (또는 별도의 안정적 사번 클레임을 합의).
- ID Token의 `aud`는 RP의 `client_id`를 반드시 포함하고 다른 audience를 추가로 담을 수 있다. 다중 audience일 때 `aud`는 문자열 배열이다. `azp`는 확장 사용 시에만 나타나며, 쓰지 않는 구현은 무시하도록 권고된다.

RFC 9068이 보장하는 것 ([RFC 9068](https://www.rfc-editor.org/rfc/rfc9068.html)): 액세스 토큰 JWT는 `typ: at+jwt`, 필수 클레임에 `iss`와 `aud`가 포함된다. `aud`는 resource indicator다. 각 Biz 인스턴스는 자기 자신을 audience로 하는 토큰만 받아들이면 된다. 이는 §5의 MCP 요구사항("서버는 자신을 audience로 발급된 토큰인지 MUST 검증")과 동일하다.

**즉 표준 경로는 이렇다:**
1. 사내 공통 IdP(OIDC AS) 하나. `iss`가 하나이므로 `iss`+`sub`로 사용자 동일성이 성립.
2. 각 Biz 인스턴스 = 별개의 **Resource Server**, 각자 고유한 resource identifier(canonical URI).
3. 클라이언트(CLI)는 RFC 8707 `resource` 파라미터로 **접근하려는 인스턴스를 지정**해 토큰을 받는다. MCP 스펙은 이걸 **MUST**로 규정한다 ([MCP Authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)).
4. Biz B 인스턴스는 `aud`가 자신인 토큰만 수락. **A용 토큰을 B에 넘기는 passthrough는 MCP 스펙상 금지(MUST NOT).**

**"레지스트리 간 신뢰"를 레지스트리끼리 세울 필요가 없다.** 공통 IdP가 신뢰의 단일 지점이고 각 인스턴스는 audience 검증만 하면 된다. MCP registry-authorization도 레지스트리가 MCP 서버와 동일한 Resource Server 패턴을 쓰라고 권고한다: "MCP clients can reuse their existing MCP authorization implementation without any changes" ([registry-authorization.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/registry-authorization.md)).

#### 9.2 인스턴스가 서로를 대신해 호출할 때 — RFC 8693 Token Exchange

전사 인스턴스가 Biz 인스턴스를 사용자 대신 호출하거나(승격 심사 시 원본 조회), Biz A 인스턴스가 Biz B 인스턴스에서 아티팩트를 pull할 때 필요한 표준이 RFC 8693 OAuth 2.0 Token Exchange다 ([RFC 8693](https://www.rfc-editor.org/rfc/rfc8693.html)):

- grant type URN: `urn:ietf:params:oauth:grant-type:token-exchange` — "indicates that a token exchange is being performed"
- 요청 파라미터:
  - `resource` — "A URI that indicates the target service or resource where the client intends to use the requested security token" (= 대상 인스턴스의 canonical URI)
  - `audience` — "The logical name of the target service where the client intends to use the requested security token"
  - `subject_token` / `subject_token_type` — "A security token that represents the identity of the party **on behalf of whom** the request is being made" (= 원 사용자)
  - `actor_token` — "A security token that represents the identity of the **acting party**" (= 호출하는 인스턴스)
  - `requested_token_type`, `scope`
- 토큰 타입 식별자: `urn:ietf:params:oauth:token-type:access_token`, `...:jwt` 등
- **위임(delegation) vs 사칭(impersonation)**: 사칭은 한 주체가 다른 주체와 "indistinguishable from" 상태가 되는 것, 위임은 두 정체성이 유지되며 "an agent for" 관계가 명시적으로 이해되는 것.
- **`act` 클레임**: "provides a means within a JWT to express that delegation has occurred and identify the acting party." 중첩 가능하며 "the outermost act claim represents the current actor."

**감사(audit) 요구가 있는 사내 플랫폼에서는 impersonation이 아니라 delegation을 써야 한다.** `act` 클레임이 "Biz A 인스턴스가 사용자 X를 대신해 Biz B를 호출했다"를 토큰 자체에 남긴다. 이게 §6 텔레메트리의 cross-Biz 사용량 귀속(attribution)과 직결된다.

#### 9.3 이 구조에서 검증되지 않은 지점

- **미확인:** MCP 인가 스펙은 클라이언트↔서버 인가만 다루고, **레지스트리↔레지스트리 신뢰 수립 절차**는 규정하지 않는다. RFC 8693이 메커니즘을 주지만, "어느 인스턴스가 어느 인스턴스를 신뢰하는가"의 정책 표현 방식은 표준이 없다. 이번 조사에서 찾은 가장 가까운 1차 자료는 §8.3의 Claude Code `allowCrossMarketplaceDependenciesOn`(비전이 allowlist)이며, 이건 클라이언트 측 정책이지 서버 간 프로토콜이 아니다.
- **미확인:** 각 인스턴스가 자체 AS를 두는 구성(IdP가 N개)에서의 cross-issuer 신뢰(예: 다중 `iss` 수용, JWKS 교차 신뢰)에 대한 1차 규격은 이번 조사 범위에서 확인하지 않았다. **공통 IdP 단일 `iss` 전제가 훨씬 단순하며, 사내 SSO가 이미 있다면 그 전제가 성립할 가능성이 높다.**

---

### 10. 인스턴스가 N개일 때의 스키마·설정 버전 파편화 관리

인스턴스마다 배포 시점이 다르면 스키마·API 버전이 갈린다. 실제 레지스트리 제품들이 쓰는 기법을 1차 자료에서 확인했다.

(1) 날짜 스탬프된 버전 URL로 스키마를 고정한다.
`server.json`의 첫 필드가 `$schema`이고 값이 버전이 박힌 URL이다: `https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json` ([versioning.mdx](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/versioning.mdx)). draft는 저장소 안에만 있고, 릴리스될 때 "changes in this section will be moved to a dated version section... and the schema will be published to a versioned URL" ([server-json CHANGELOG](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/CHANGELOG.md)). **문서 자체가 자기 스키마 버전을 들고 다니므로 구버전 인스턴스가 쓴 문서를 신버전 인스턴스가 해석할 수 있다.**

(2) 안정 API 버전과 개발 API 버전을 URL 경로로 분리한다.
2025-10-17에 `/v0.1/`을 도입하면서 ([API CHANGELOG](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/CHANGELOG.md)):
- 모든 `/v0/` 엔드포인트를 `/v0.1/`에도 노출, 현재는 동작 동일
- `/v0/`는 **additive change로 계속 진화**(새 optional 필드, 새 엔드포인트)
- `/v0.1/`은 **additive·하위호환 변경만** 하며 안정 유지
- 둘 다 v1.0까지 유지
- 마이그레이션 지침: "Production applications should consider using `/v0.1/` for stability"

N개 인스턴스가 서로 다른 시점에 배포되므로 인스턴스 간 통신(승격·동기화)은 반드시 안정 버전 경로를 쓴다. 사내 실험은 개발 경로에서 한다.

(3) 확장은 reverse-DNS 네임스페이스로 격리해 코어를 오염시키지 않는다.
`/v0.1/x/<namespace>/<extension>` 경로와 `_meta["<reverse-dns>/..."]` 키. 그리고 소비 측 규범: "Clients consuming extensions MUST gracefully handle missing extensions" ([extensions.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/extensions.md)). **상위 인스턴스가 하위의 확장 필드를 이해 못해도 깨지지 않아야 한다.** 이게 파편화 내성의 실제 규칙이다.

(4) 발행된 데이터를 불변으로 만들어 마이그레이션 자체를 없앤다.
"The version string **MUST** be unique for each publication of the server. Once published, the version string (and other metadata) cannot be changed" ([versioning.mdx](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/versioning.mdx)). 버전 범위 표기(`^1.2.3`, `1.x`, `1 - 2` 등)는 발행 시 금지되고 정확한 버전만 허용된다. `_meta`의 publisher 확장은 4KB 제한.

(5) 클라이언트 측 의존성은 semver range로 고정하고 충돌은 교집합으로 해소한다.
Claude Code plugin `dependencies`: `{ "name": "secrets-vault", "version": "~2.1.0" }`. 여러 플러그인이 같은 의존성을 제약하면 **"Claude Code intersects their ranges and resolves the dependency to the highest version that satisfies all of them"**, 교집합이 없으면 `range-conflict`로 설치 실패한다. 버전 해소는 git 태그 `{plugin-name}--v{version}`을 기준으로 한다. 태그가 강제 이동되면 캐시 디렉터리 이름에 12자 커밋 SHA 접미가 붙어 stale 캐시 재사용을 막는다 ([Plugin dependencies](https://code.claude.com/docs/en/plugin-dependencies)). pre-release는 range가 명시적으로 opt-in(`^2.0.0-0`)하지 않는 한 제외된다.

(6) 조직 정책은 managed settings로 중앙에서 고정한다.
`extraKnownMarketplaces`(자동 등록), `strictKnownMarketplaces`(소스 제한, `hostPattern` 정규식), `enabledPlugins`(기본 활성) ([Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)). 비-Anthropic 마켓플레이스는 auto-update가 기본 off다 ([Plugin dependencies](https://code.claude.com/docs/en/plugin-dependencies)). 인스턴스가 N개일 때 의도치 않은 전파를 막는 기본값이다.

---

## 어떻게 사용하는가? (How)

### A. 패키징 포맷 비교 — 무엇을 저장할 것인가

| 포맷 | 단위 | 매니페스트 | 필수 필드 | 아티팩트 호스팅 | 무결성 | 우리 적합도 |
|---|---|---|---|---|---|---|
| **Agent Skill** | 디렉터리 | `SKILL.md` frontmatter | `name`(≤64, `[a-z0-9-]`), `description`(≤1024) | 없음(파일시스템/zip/API) | 없음(스펙상) | **필수.** 최소 저장 단위 |
| **Claude Code Plugin** | 디렉터리 | `.claude-plugin/plugin.json` | `name` | marketplace source가 결정 | `archive` source의 `sha256`, `github` source의 `sha` | **필수.** Skill+Agent+MCP+hooks 묶음 |
| **Marketplace 카탈로그** | 저장소 | `.claude-plugin/marketplace.json` | `name`, `owner`, `plugins[]` | — | plugin별 | **필수.** 배포/설치 표면 |
| **MCP `server.json`** | 서버 1개 | `server.json` | `name`, `description`(≤100), `version` | 외부 패키지 레지스트리 참조 | `packages[].fileSha256` | **카탈로그 스키마의 기반으로 채택 권장** |
| **MCPB 번들** | 단일 파일 | `server.json`의 `registryType: "mcpb"` | `identifier`(URL), `fileSha256`(필수) | GitHub/GitLab releases만 (공식 레지스트리 제약) | SHA-256 필수 | 사내에선 MinIO presigned URL로 대체 가능 |
| **OCI artifact** | manifest+layers | OCI image manifest | `artifactType`, config는 empty descriptor 허용 | OCI 레지스트리 | digest 기반 전면 | 강력하나 **허용 스택에 레지스트리 없음** |

저장 대상으로 (1) 원본 디렉터리를 tar.gz로 압축한 번들 아티팩트(MinIO), (2) `SKILL.md`/`plugin.json`/`marketplace.json`에서 추출·정규화한 카탈로그 문서(MongoDB), (3) 검색용 역인덱스(OpenSearch)를 권고한다. 카탈로그 문서 스키마는 MCP `server.json`을 확장한다(`name` reverse-DNS 불변, `title` 가변, `_meta.<사내 네임스페이스>`에 팀/심사/승격 정보).

### B. MCP Registry 데이터 모델 + API 요약 (채택/확장 대상)

위 §3.1~3.4에 정리. 채택 시 사내 확장 지점:

1. `name` 네임스페이스를 **`kr.co.skhynix.<biz>.<불변 team_id>/<agent-name>`** 형태로. 팀명이 아니라 team_id를 넣는다 (npm의 실패를 피하는 지점).
2. `title`에 팀 표시명·에이전트 표시명 — 자유롭게 변경 가능.
3. `_meta["kr.co.skhynix.registry/official"]`에 `status`/`publishedAt`/`isLatest` (MCP와 동형) + `ownerTeamId`/`reviewStatus`/`promotionTier`.
4. 승격 API는 확장 규약대로 `POST /v0.1/x/kr.co.skhynix/promote`.
5. 전사(2-tier)는 각 Biz를 `GET /v0.1/servers?updated_since=...`로 폴링 — MCP aggregator 패턴 그대로.

### C. 소프트웨어 계층 분해

| # | 계층 | 책임 | 허용 스택 후보 | 1차 근거 |
|---|---|---|---|---|
| 1 | **Storage (아티팩트)** | 번들 tar.gz, 서명, SBOM 저장. 발행된 버전은 불변 | **MinIO** — 버전 관리 + Object Lock. GOVERNANCE 모드는 `s3:BypassGovernanceRetention` 권한자만 우회 가능, COMPLIANCE 모드는 **root 포함 모든 사용자로부터 write 차단**. **Object locking은 versioning을 요구**하며 버킷 생성 시 활성화하면 versioning이 자동 켜짐 | [MinIO Object Locking and Immutability](https://docs.min.io/aistor/administration/object-locking-and-immutability/) |
| 2 | **Catalog (메타데이터)** | `server.json` 유사 문서, 버전 이력, 팀 소유권 튜플 | **MongoDB** — `validator` + `$jsonSchema`로 스키마 강제, `validationLevel`/`validationAction`으로 기존 문서 처리·위반 시 동작 제어. 기본값은 invalid 문서 거부 | [MongoDB Schema Validation](https://www.mongodb.com/docs/manual/core/schema-validation/) |
| 3 | **API** | REST(발행/조회/상태변경/승격) | **FastAPI** — OpenAPI/JSON Schema 자동 생성, Pydantic 검증, DI 시스템, OAuth2+JWT 보안 유틸리티 내장, async. **Flask보다 FastAPI 권장**: 레지스트리 API의 계약(contract)이 곧 OpenAPI이고, MCP 레지스트리 자체가 openapi.yaml을 규격으로 배포한다 — 스펙 파일을 손으로 유지하지 않아도 되는 쪽이 맞다 | [FastAPI Features](https://fastapi.tiangolo.com/features/), [MCP openapi.yaml](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/openapi.yaml) |
| 4 | **Auth** | 사내 SSO(OIDC) 로그인, M2M 토큰, 네임스페이스 소유권 판정 | **FastAPI 보안 유틸 + 사내 IdP**. 토큰은 RFC 9068 `at+jwt`(필수 클레임 `iss,exp,aud,sub,client_id,iat,jti`), 팀 소유권은 `groups`/`roles` 선택 클레임에 **불변 team_id**로. 레지스트리는 OAuth 2.1 Resource Server, 스코프 `registry:read`/`registry:write`. **스코프만으로 부족 — 리소스 단위 소유권 판정 별도 필요** | [RFC 9068](https://www.rfc-editor.org/rfc/rfc9068.html), [MCP registry-authorization](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/registry-authorization.md) |
| 5 | **Distribution / CLI** | 업로드·다운로드·설치 | **기성 경로 재사용**: `.claude-plugin/marketplace.json`을 API가 생성해 서빙하고, 사원은 `/plugin marketplace add <사내 URL>` → `/plugin install x@biz-market`. 관리자는 `extraKnownMarketplaces` + `strictKnownMarketplaces`(hostPattern)로 자동 등록·소스 제한. 아티팩트는 `source: "archive"` + `sha256` | [Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces) |
| 6 | **Search** | 카탈로그 전문 검색 | **OpenSearch**. MCP 공식 API가 `search`를 "intentionally simple"로 두고 고급 검색을 subregistry에 위임한다고 명시 — 우리가 채워야 할 자리 | [official-registry-api.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/official-registry-api.md) |
| 7 | **Cache / Rate limit / Queue** | 카탈로그 캐시, API 레이트리밋, 스캔·인덱싱 잡 | **Redis**. 레이트리밋은 공식 문서화된 INCR+EXPIRE 패턴(`MULTI/EXEC` 또는 Lua `EVAL`로 race 제거) | [Redis INCR — Pattern: rate limiter](https://redis.io/docs/latest/commands/incr/) |
| 8 | **Observability** | 에이전트 사용량/비용/지연/에러 | **OTel SDK → OTel Collector → Data Prepper(`otel_trace_source`/`otel-metrics-source`/`otel-logs-source`) → OpenSearch**. 인덱스 `otel-v1-apm-span`, `otel-v1-apm-service-map`. 보존은 ISM 정책. 속성은 `gen_ai.*` 표준 사용, 팀 소유권은 사내 커스텀 속성 | [OpenSearch Trace Analytics](https://docs.opensearch.org/latest/data-prepper/common-use-cases/trace-analytics/), [OTel GenAI metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) |
| 9 | **Federation (인스턴스 간)** | Biz 인스턴스 N개 ↔ 전사 인스턴스 동기화, 타 Biz 카탈로그 조회 | **동기화는 pull**: 상위가 `GET /v0.1/servers?updated_since=`로 증분 폴링(`include_deleted` 자동 true). **아티팩트는 lazy copy** — 사전 미러링 금지(Artifactory 명시). 인스턴스 간 호출 토큰은 RFC 8693 delegation(`act` 클레임). 클라이언트 측 신뢰는 명시적 allowlist + **비전이** | [MCP aggregators](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/registry-aggregators.mdx), [Artifactory remote repositories](https://docs.jfrog.com/artifactory/docs/remote-repositories), [RFC 8693](https://www.rfc-editor.org/rfc/rfc8693.html), [Plugin dependencies](https://code.claude.com/docs/en/plugin-dependencies) |
| 10 | **Governance / Promotion** | 심사 게이트, 서명·검증, 1→2 tier 승격 | Anthropic 엔터프라이즈 체크리스트를 발행 파이프라인의 게이트로. 서명은 **cosign 자체 키**(폐쇄망 가능), 증명은 **in-toto Statement + SLSA Provenance predicate**. 승격 *결정*은 `/v0.1/x/<ns>/promote` 확장(push), 승격 후 *데이터*는 pull | [Skills for enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise), [Sigstore self-managed keys](https://docs.sigstore.dev/cosign/key_management/signing_with_self-managed_keys/), [in-toto statement](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md), [SLSA provenance](https://slsa.dev/spec/v1.0/provenance) |
| 11 | **Schema / version discipline** | N개 인스턴스의 배포 시점 차이 흡수 | 문서마다 **날짜 스탬프 `$schema` URL**, API는 안정(`/v0.1`)/개발(`/v0`) 경로 분리 + additive-only, 확장은 reverse-DNS 격리 + "MUST gracefully handle missing extensions", 발행 데이터 불변, 버전 범위 표기 금지 | [MCP versioning](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/versioning.mdx), [API CHANGELOG](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/CHANGELOG.md), [extensions.md](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/extensions.md) |

### D. 팀 rename 문제 — 자료에서 도출되는 구체적 처방

| 원칙 | 근거 |
|---|---|
| 네임스페이스에 **team_id(불변)** 를 넣고 팀명은 절대 넣지 않는다 | npm은 스코프=조직명이라 rename 시 전 패키지 재발행 필요 ([npm](https://docs.npmjs.com/renaming-an-organization/)) |
| 식별자 `name`과 표시명 `title`을 스키마에서 분리한다 | MCP `server.json`이 `name`(reverse-DNS, 패턴 강제)과 `title`("Optional human-readable title or display name")을 분리 ([schema](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/draft/server.schema.json)) |
| 소유권은 이름이 아니라 **롤/관계 배정**으로 표현한다 | PyPI 조직/프로젝트 롤 분리 ([PyPI](https://docs.pypi.org/organization-accounts/roles-entities/)); Zanzibar 관계 튜플 ([Zanzibar](https://research.google/pubs/zanzibar-googles-consistent-global-authorization-system/)) |
| 발행된 버전 메타데이터는 **불변**, 변하는 건 `status`뿐 | MCP: "Server metadata is generally immutable, except for the `status` field" ([aggregators](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/registry-aggregators.mdx)) |
| 아티팩트도 불변으로 강제한다 | MinIO Object Lock COMPLIANCE 모드는 root도 write 불가 ([MinIO](https://docs.min.io/aistor/administration/object-locking-and-immutability/)) |
| 이름 변경 경로가 필요하면 **rename map**을 카탈로그가 제공한다 | Claude Code marketplace에 `renames` 필드가 존재("Map old plugin names to new names or null") ([Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)) |

---

## 설계서 작성 전 사용자가 답해야 할 열린 질문

1. **1차 대상 런타임은 무엇인가?** Claude Code(파일시스템 Skill/plugin)인가, 사내 자체 에이전트 런타임인가, 둘 다인가. Anthropic 문서상 Skill은 surface 간 동기화되지 않으므로, 대상마다 별도 배포 경로가 필요하다.
2. **`team_id`의 발급 주체는?** 사내 HR/조직 시스템에 매년 변하지 않는 조직 코드가 이미 있는가, 아니면 레지스트리가 자체 ID를 발급하고 조직 코드를 매핑 테이블로 관리하는가. 후자면 조직 개편 시 매핑 갱신 프로세스가 필요하다.
3. **팀이 해체/병합되면 자산 소유권은 어디로 가는가?** (rename보다 어려운 케이스. MCP registry에는 이전 기능이 문서상 없음 — 우리가 정의해야 함.)
4. **승격 시 아티팩트를 복사하는가, 참조만 하는가?** 1차 자료는 둘 다 지지하지 않는 것이 하나 있다 — **사전 전량 미러링**. 전사 인스턴스가 자체 MinIO에 lazy copy할지(Artifactory 방식, Biz 인스턴스 장애에 강함), `packages[].fileSha256` 참조만 둘지(MCP 방식, 저장 비용 0이지만 Biz 인스턴스가 죽으면 설치 실패) 결정 필요.
5. **승격된 에이전트의 소유권은 누구인가?** Biz 팀이 계속 유지보수하는가, 전사 팀이 인수하는가. 원본이 deprecated 되면 승격본은? (MCP는 `status`만 가변으로 두므로 상태 전파 규칙을 정의해야 함.)
5b. Biz 인스턴스가 몇 개까지 늘어나며 서로 직접 통신하는가? N-to-1(모두 전사만 바라봄)이면 신뢰 관계가 N개, N-to-N(Biz끼리 직접)이면 N² 로 늘어난다. Claude Code의 "trust does not chain" 원칙상 전이 신뢰를 쓸 수 없으므로 이 숫자가 그대로 운영 부담이 된다.
5c. 인스턴스 배포 버전을 강제 동기화할 수 있는가? 아니라면 §10의 안정 API 경로(`/v0.1`)와 날짜 스탬프 스키마를 처음부터 도입해야 한다. 나중에 넣으면 이미 파편화된 뒤다.
6. **OCI 레지스트리를 스택에 추가할 것인가?** 추가하면 digest 기반 불변성·referrers API를 통한 서명/증명 부착이 공짜로 따라온다. 안 하면 MinIO + 자체 매니페스트로 같은 것을 직접 만들어야 한다.
7. **사내 IdP가 OIDC를 지원하는가, SAML만인가?** MCP 인가 스펙(OAuth 2.1/RFC 9728)을 따르려면 OIDC/OAuth AS가 필요하다. SAML만이면 브릿지가 필요하다.
7b. 사내 IdP의 subject identifier 유형(public / pairwise(PPID))은 무엇인가? ★ pairwise면 Biz A 인스턴스와 Biz B 인스턴스가 같은 사용자를 다른 `sub`로 보게 되어 인스턴스 경계를 넘는 사용자 동일성 판정이 깨진다. public으로 설정하거나 안정적인 사번 클레임을 별도 합의해야 한다 ([OIDC Core](https://openid.net/specs/openid-connect-core-1_0.html)).
7c. IdP가 RFC 8693 token exchange를 지원하는가? 인스턴스가 사용자를 대신해 다른 인스턴스를 호출하는 경로(승격 심사 시 원본 조회, 타 Biz 아티팩트 pull)에 필요하다. 미지원이면 인스턴스 간 호출은 client_credentials(서비스 계정)로 떨어진다. 누구를 대신한 호출인지 토큰에 남지 않아 감사·사용량 귀속이 약해진다.
8. **에이전트가 사내 서비스를 호출할 때 토큰 모델은?** MCP 스펙은 token passthrough를 금지(MUST NOT)하므로, 에이전트→상류 API는 별도 토큰 발급이 필요하다. 사내 API 게이트웨이가 이걸 지원하는가.
9. **텔레메트리를 누가 계측하는가?** Anthropic은 "Usage analytics are not currently available through the Skills API"라고 명시한다. 에이전트 실행 지점(사용자 로컬 Claude Code 포함)에서 OTLP를 내보낼 방법이 있는가 — 없으면 사용량 대시보드는 게이트웨이/프록시 경유 트래픽만 볼 수 있다.
10. **심사(review) 인력과 SLA는?** Anthropic 체크리스트는 "authors should not be their own reviewers"를 요구한다. Biz별 리뷰어 풀이 실재하는가, 아니면 자동 스캔 + 사후 감사로 갈 것인가.
11. **폐쇄망에서 서명 키를 어디에 두는가?** cosign 자체 키는 폐쇄망에서 동작하나, 키 보관·회전 주체(사내 KMS/HSM 유무)가 정해져야 한다.
12. **에이전트 정의에 프롬프트·모델 파라미터가 포함되는가?** 사내 LLM(Kimi-K2.5, Qwen3-VL, BGE-M3)의 엔드포인트/모델명을 매니페스트에 박을지, 런타임 설정으로 주입할지. `plugin.json`의 `userConfig`가 후자를 위한 기성 메커니즘이다.

---

## 이번 조사에서 **1차 자료로 확인하지 못한** 항목 (설계 전 별도 확인 필요)

- PyPI 프로젝트명 불변성 및 소유권 이전 공식 절차
- **레지스트리↔레지스트리 신뢰 수립 프로토콜** — 표준 없음을 확인. RFC 8693이 토큰 메커니즘을, Claude Code `allowCrossMarketplaceDependenciesOn`이 클라이언트 측 비전이 allowlist를 주지만, 서버 간 정책 표현 표준은 부재
- **downstream → upstream push federation** — 어느 1차 자료도 표준화하지 않음(전부 pull). `POST /v0.1/publish`를 인스턴스가 발행자 클라이언트로서 호출하는 것이 가장 가까운 경로
- 다중 issuer(인스턴스별 자체 AS) 구성에서의 cross-issuer 신뢰(JWKS 교차 신뢰) 규격
- Artifactory virtual repository의 정확한 충돌 해소 규칙 (remote repository 문서에서는 확인, virtual 전용 문서는 미확인. Nexus group의 "hosted first" 권고는 확인)
- MCP Registry의 네임스페이스 소유권 **이전(transfer)** 기능 존재 여부 (발행 시 소유권 *증명*만 문서화 확인)
- DSSE envelope의 `payloadType` 값
- Claude Code marketplace `source: "github"`에서 GitHub Enterprise 호스트를 지정하는 방법
- `gen_ai.*` semconv에 팀/소유자/조직 속성 — **표준에 존재하지 않음을 확인**(부재), 커스텀 속성 필요
- MinIO Object Lock의 커뮤니티(AGPL) 에디션 지원 범위 (확인한 문서는 AIStor 기준)

---

## 참고 자료 (References)

**Agent Skills / Plugins**
- [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [Skills for enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise)
- [Plugins reference](https://code.claude.com/docs/en/plugins-reference)
- [Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)
- [Constrain plugin dependency versions](https://code.claude.com/docs/en/plugin-dependencies)

**MCP**
- [MCP Authorization spec 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
- [MCP Registry — generic server.json](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/generic-server-json.md)
- [MCP Registry — server.schema.json](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/draft/server.schema.json)
- [MCP Registry — official registry requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Registry — generic registry API](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/generic-registry-api.md)
- [MCP Registry — official registry API](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/official-registry-api.md)
- [MCP Registry — registry authorization](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/registry-authorization.md)
- [MCP Registry — extensions](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/extensions.md)
- [MCP Registry — aggregators / subregistries](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/registry-aggregators.mdx)
- [MCP Registry — authentication / namespaces](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/authentication.mdx)
- [MCP Registry — publishing quickstart](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx)
- [MCP Registry — openapi.yaml](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/openapi.yaml)
- [MCP Registry — versioning](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/versioning.mdx)
- [MCP Registry — server.json CHANGELOG](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/CHANGELOG.md)
- [MCP Registry — API CHANGELOG](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/api/CHANGELOG.md)

**연합 / 다중 인스턴스**
- [JFrog Artifactory — Remote repositories](https://docs.jfrog.com/artifactory/docs/remote-repositories)
- [Sonatype Nexus — Repository manager concepts (hosted/proxy/group)](https://help.sonatype.com/en/repository-manager-concepts.html)

**OCI**
- [OCI image-spec — manifest.md (artifact usage)](https://github.com/opencontainers/image-spec/blob/main/manifest.md)
- [OCI image-spec — artifacts-guidance.md](https://github.com/opencontainers/image-spec/blob/main/artifacts-guidance.md)
- [OCI distribution-spec](https://github.com/opencontainers/distribution-spec/blob/main/spec.md)

**네임스페이스 / 소유권 선행 사례**
- [npm — About scopes](https://docs.npmjs.com/about-scopes/)
- [npm — Renaming an organization](https://docs.npmjs.com/renaming-an-organization/)
- [PyPI — Organization accounts](https://docs.pypi.org/organization-accounts/)
- [PyPI — Roles and entities](https://docs.pypi.org/organization-accounts/roles-entities/)
- [Zanzibar: Google's Consistent, Global Authorization System (USENIX ATC '19)](https://research.google/pubs/zanzibar-googles-consistent-global-authorization-system/)

**AuthN/AuthZ**
- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html) — `iss`+`sub`, public vs pairwise subject identifier, `aud`/`azp`
- [RFC 9068 — JWT Profile for OAuth 2.0 Access Tokens](https://www.rfc-editor.org/rfc/rfc9068.html)
- [RFC 8693 — OAuth 2.0 Token Exchange](https://www.rfc-editor.org/rfc/rfc8693.html) — 인스턴스 간 위임, `act` 클레임
- RFC 8414 / RFC 7591 / RFC 9728 / RFC 8707 — MCP 인가 스펙이 인용하는 기반 규격

**텔레메트리**
- [OTel — GenAI semconv 이전 안내](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OTel GenAI — attribute registry](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md)
- [OTel GenAI — metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md)
- [OTel GenAI — spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [OpenSearch — Trace Analytics (Data Prepper)](https://docs.opensearch.org/latest/data-prepper/common-use-cases/trace-analytics/)
- [OpenSearch — Index State Management](https://docs.opensearch.org/latest/im-plugin/ism/index/)

**스택 검증**
- [MinIO — Object Locking and Immutability](https://docs.min.io/aistor/administration/object-locking-and-immutability/)
- [MongoDB — Schema Validation](https://www.mongodb.com/docs/manual/core/schema-validation/)
- [Redis — INCR (rate limiter pattern)](https://redis.io/docs/latest/commands/incr/)
- [FastAPI — Features](https://fastapi.tiangolo.com/features/)

**공급망**
- [Sigstore — Signing with self-managed keys](https://docs.sigstore.dev/cosign/key_management/signing_with_self-managed_keys/)
- [Sigstore — Key management overview](https://docs.sigstore.dev/cosign/key_management/)
- [in-toto — Attestation Statement v1](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md)
- [SLSA v1.0 — Provenance](https://slsa.dev/spec/v1.0/provenance)
