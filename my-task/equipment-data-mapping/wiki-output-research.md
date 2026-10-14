---
tags: [equipment-data-mapping, wiki, research]
note_kind: "조사·제안"
last_updated: 2026-10-15
review_status: source_review_only
---

# 장비 데이터 지도 위키 — LLM Wiki와 OpenWiki 적용 검토

> 제안: 기존 파일 종류별 위키를 유지하고, LLM Wiki의 작성 규칙과 OpenWiki의 주장별 근거·갱신 방식을 가져온다. 두 앱의 직접 도입은 보류한다.

## 왜 필요한가? (Why)

위키를 읽는 엔지니어는 “어디에 무엇이 있고, 무슨 뜻이며, 무엇을 믿어도 되는가”를 알아야 한다. 파일 목록만 있으면 의미를 알 수 없고, LLM 설명만 있으면 실제 파일로 돌아가 확인할 수 없다.

현재 [P1 실행 계획](./04-phased-plan.md)의 “낮과 밤의 분업”에는 파일 종류마다 한 장, 장비·local 경로 병기, 필드별 근거와 미검증 표시, 중간 결과 저장, 아침 보고서가 이미 있다. [2026-10-13 결정](./decision-log.md)도 P1에서 비교용 추출 계약과 전체 YAML 스키마를 미루도록 정했다. 이번 조사는 이를 대체하는 새 설계가 아니라 위키 출력의 빈칸을 보완하는 제안이다.

## 핵심 개념 (What)

### 두 프로젝트에서 확인한 것

| 대상 | 확인한 방식 | 장비 매핑에서의 판단 |
|---|---|---|
| LLM Wiki | 원본, 생성 위키, 작성 규칙을 분리하고 색인·로그를 둔다 | 장비 표본을 보존하고 그 위에 설명 문서를 누적하기 좋다 |
| LLM Wiki | purpose와 schema를 구분하고 사용자 정의 페이지 유형을 허용한다 | 장비용 목적·페이지 틀을 정의하는 데 활용한다 |
| LLM Wiki | 생성 페이지에 `sources[]`를 넣도록 요구한다 | 페이지 단위 출처다. 필드 뜻의 개별 근거는 별도로 필요하다 |
| OpenWiki | Claim을 단일 주장과 버전이 있는 근거로 저장한다 | 필드 의미·파일 관계를 근거와 함께 관리하는 데 참고한다 |
| OpenWiki | 근거의 현재 버전을 비교해 stale 또는 unresolved를 찾는다 | 다시 읽을 페이지와 재검토할 설명을 좁히는 데 참고한다 |
| OpenWiki | 페이지 본문·근거·검증·manifest를 저장한 뒤 완료 처리한다 | 밤 실행의 종류별 완료·재시작 규칙을 구체화하는 데 참고한다 |

LLM Wiki의 원 패턴은 도메인에 따라 폴더·페이지 형식을 정하도록 한다. 앱 구현도 프로젝트 schema를 페이지 배치의 기준으로 사용한다. 파일 목록과 색인을 코드로 만들고 LLM이 의미 설명을 작성하는 기존 방향과 잘 맞는다. 현재 생성 프롬프트는 `index.md`와 `overview.md`를 LLM이 출력하지 않도록 명시한다. [원 패턴](https://github.com/nashsu/llm_wiki/blob/48fd970e206a02a6d2028d1dbfc41b7a0345bf0b/llm-wiki.md), [템플릿](https://github.com/nashsu/llm_wiki/blob/48fd970e206a02a6d2028d1dbfc41b7a0345bf0b/src/lib/templates.ts#L62-L122), [생성 프롬프트](https://github.com/nashsu/llm_wiki/blob/48fd970e206a02a6d2028d1dbfc41b7a0345bf0b/src/lib/ingest.ts#L2390-L2490)

OpenWiki는 LangChain의 코드 저장소 문서화 도구로 출발했다. 현재 README에는 personal 모드도 있지만, Grounded Claims는 저장소 코드 위키와 저장소 근거에 적용된다고 명시한다. 장비 파일 폴더를 넣으면 같은 근거 추적이 자동으로 된다고 가정할 수 없다. [공식 소개](https://www.langchain.com/blog/introducing-openwiki-an-open-source-agent-for-repo-documentation), [현재 README](https://github.com/langchain-ai/openwiki/blob/7c469e59584bfac3f97f5c32a67604de46805999/README.md)

구현에서는 Claim에 `id`, `statement`, `evidence[]`를 두고, 근거는 `resource`와 `version`을 가진다. 사전 검사에서 근거를 다시 해석해 버전이 다르면 stale, 찾지 못하면 unresolved로 기록한다. 이 방식은 의미의 진실성을 증명하는 절차와 구분해야 한다. [Claim 타입](https://github.com/langchain-ai/openwiki/blob/7c469e59584bfac3f97f5c32a67604de46805999/src/claims/core/types.ts#L1-L32), [근거 변경 검사](https://github.com/langchain-ai/openwiki/blob/7c469e59584bfac3f97f5c32a67604de46805999/src/claims/brains/code/preflight.ts#L55-L90)

### 위키에서 나눠 보여줄 정보

| 정보 | 출력 단위 | 작성 주체 |
|---|---|---|
| 장비 데이터 구조 | 모델·S/W 버전의 색인과 폴더 트리 | 코드가 관측 구조 생성, LLM이 설명 초안 |
| 실제 파일 목록 | 장비·수집 회차별 목록 파일 | 코드 |
| 파일의 의미 | 파일 종류별 페이지 | 코드 프로파일 + LLM 해석 + 현업 판정 |
| 필드 구조와 뜻 | 파일 종류 페이지 안의 필드 표 | 코드가 구조, LLM이 뜻 후보 |
| 파일 사이의 관계 | 관련 종류를 잇는 링크와 근거 표 | 코드 관측 또는 LLM 후보, 상태 명시 |
| 모르는 것·모순 | 검토 목록과 아침 보고서 | 코드 집계, LLM 제안, 사람 판정 |

수만 개의 파일을 전부 위키 페이지로 만들지 않는다. **실제 파일 목록은 빠짐없이 기록하고, 설명은 파일 종류로 묶는다.** 단독 설정 파일만 예외로 한 장을 둔다. 필드별 독립 페이지와 벤더 공통 개념 페이지는 재사용 필요가 생길 때 추가한다.

## 어떻게 사용하는가? (How)

### 최소 출력 구조 제안

아래는 사내 산출물 루트의 예시이며 이 저장소에 실데이터를 만들자는 뜻이 아니다. 파일명과 계층은 제안이고 아직 구현하지 않았다.

```text
mapping-output/
  purpose.md                         # 위키가 답해야 하는 질문과 범위
  schema.md                          # 페이지 틀, 용어, 근거·승인·갱신 규칙
  raw/<equipment-id>/<run-id>/        # 받은 표본. 회차별 보존
  runs/<run-id>/
    inventory.jsonl                  # 조회된 파일, 종류 ID, 수집/제외/실패 상태
    profiles/                        # 코드가 뽑은 구조·통계와 LLM 해석 중간 결과
    report.md                        # 아침 보고서
  wiki/<model>/<sw-version>/
    index.md                         # 종류별 설명·개수·검증 상태와 링크
    stores/                          # 의미 있는 저장소·폴더의 구조와 종류 목록
    file-classes/<class-id>.md        # 파일 종류별 설명과 필드 표
    review.md                        # 모르는 뜻, 모순, 근거 변경으로 재검토할 항목
    log.md                           # 어떤 회차가 무엇을 바꿨는지
```

`purpose.md`에는 “교정 데이터는 어디에 있는가”, “어떤 파일에서 어떤 종류의 데이터를 얻는가”, “설명이 확인된 범위는 어디까지인가”처럼 사용자 질문을 적는다. `schema.md`에는 허용하는 페이지 유형과 아래 틀을 적는다. 둘을 처음부터 큰 설정 체계로 만들 필요는 없다.

모델 공통 설명과 장비별 관측을 구분한다. 모델 위키에는 여러 장비에 공통으로 쓸 설명을 두고, 파일 개수·값 예시·수집 시각에는 어느 장비·회차의 관측인지 표시한다. 첫 장비만 봤다면 “이 모델 전체에서 확인됨”이라고 쓰지 않는다.

### 파일 종류 한 장에 담을 것

| 절 | 필요한 내용 |
|---|---|
| 식별·출처 | 종류 ID, 모델, S/W 버전, 관측 장비·회차, 장비 경로 패턴, local 표본 경로 |
| 한 줄 설명 | 무엇이 들어 있는지, 역할 후보, 의미 검증 상태 |
| 파일 목록 | 해당 회차에서 발견한 개수, 받은 표본 개수, 원본 목록 링크, 대표 파일명 |
| 데이터 구조 | 인코딩·구분자·섹션·열·반복 단위. 표본에서 관측한 범위 명시 |
| 필드 표 | 이름·자료형·단위·값 예시·뜻 후보·근거·현업 판정 |
| 관련 파일 | 어떤 키나 기록을 통해 연결되는지, 관계의 근거와 검증 상태 |
| 모르는 것 | 미해석 필드, 표본 사이의 차이, 추가로 필요한 표본·현업 질문 |
| 갱신 기록 | 근거 표본·프로파일의 버전과 변경 이유, 승인 이력 |

예를 들어 아래 행은 전부 가상의 장비·필드이며, 원문이 존재한다는 확인과 뜻의 승인을 나누어 보여준다.

| 필드 | 코드 관측 | 뜻 후보 | 근거 | 현업 판정 |
|---|---|---|---|---|
| `XOffset` | 실수, `[Stage]` 섹션 | 스테이지 X축 보정값 | `sample:s0012`, 48행, `XOffset=0.42` | 미검증 |

원문에 단위가 없으면 `um`처럼 추측해서 관측 칸에 넣지 않는다. 필요하면 뜻 후보와 함께 “단위 추정, 미검증”으로 적는다. 파일 이름이 비슷하다는 것만으로 “A가 B를 생성한다”는 관계를 확정하지 않는다. 관계도 필드 의미와 같은 근거·승인 규칙을 적용한다.

P1에서는 이 표와 근거 참조로 시작한다. OpenWiki의 전체 Claim 저장 형식이나 P2의 `locator`, `expect_count`를 지금 도입하지 않는다. 개별 주장을 여러 페이지에서 재사용하거나 변경 영향 추적이 복잡해지면 주장 ID와 별도 근거 파일로 확장한다.

### 생성·검토·갱신 규칙

1. **목록과 구조는 코드로 만든다.** 파일 목록의 각 항목에 종류 ID 또는 미분류·제외·실패 사유를 남긴다. 종류별 파일 수와 표본 수를 구분하고, 일부 읽기 범위를 기록한다.
2. **LLM에는 해석만 맡긴다.** 코드 프로파일과 표본을 보고 뜻·관계 후보를 제안한다. 근거의 위치와 원문을 검사한 뒤 정해진 틀로 렌더링한다. “근거 있음”은 “뜻이 맞음”의 승인이 아니다.
3. **현업 판정은 별도로 보존한다.** 페이지를 다시 생성해도 승인자·승인 시각·승인 대상의 근거 버전이 사라지지 않아야 한다. 근거가 바뀌면 지난 승인은 이력으로 남기고 현재 주장을 재검토 대상으로 표시한다.
4. **완료 표시는 마지막에 한다.** 표본 참조, 프로파일·해석 결과, 위키 페이지가 저장되고 검사가 끝난 종류만 완료로 기록한다. 페이지별 저장 후 완료 처리하는 OpenWiki의 manifest 방식이 참고가 된다. [페이지 완료 검사](https://github.com/langchain-ai/openwiki/blob/7c469e59584bfac3f97f5c32a67604de46805999/src/generation/page-manifest.ts#L175-L193)
5. **바뀐 근거에 연결된 설명을 재검토한다.** 받은 표본의 해시 외에도 parser·프롬프트·페이지 규칙의 개정을 기록한다. 목록의 크기·수정 시각만 같다는 이유로 원격 파일 내용이 불변이라고 보장하지 않는다.
6. **미관측을 삭제로 단정하지 않는다.** 이번 회차에서 못 읽었거나 수집 범위에서 빠진 파일은 과거 설명을 지우지 않고 미관측으로 표시한다. 원본 삭제, 접근 실패, 표본 미선택을 구분한다.

LLM Wiki에는 텍스트 해시 캐시와 source 삭제 시 공유 페이지를 보존하는 로직이 있다. 다만 공유 페이지 본문에서 그 source만 뒷받침하던 주장을 골라 폐기하는 것까지 해당 삭제 결정 함수가 수행하지는 않는다. 장비 위키는 위의 근거별 재검토 규칙을 따르는 편이 맞다. [캐시](https://github.com/nashsu/llm_wiki/blob/48fd970e206a02a6d2028d1dbfc41b7a0345bf0b/src/lib/ingest-cache.ts), [삭제 결정](https://github.com/nashsu/llm_wiki/blob/48fd970e206a02a6d2028d1dbfc41b7a0345bf0b/src/lib/source-delete-decision.ts)

위키 검사는 링크·필수 칸·목록 개수·근거 위치 같은 기계 검사부터 한다. LLM Wiki의 구조 lint는 고립 페이지·깨진 링크·외부로 나가는 링크가 없는 페이지를 찾는다. 의미 lint는 LLM 판단이므로 검토 후보로만 쓴다. 링크 검사를 통과했다고 파일 뜻이 맞는 것은 아니다. [구조 lint](https://github.com/nashsu/llm_wiki/blob/48fd970e206a02a6d2028d1dbfc41b7a0345bf0b/src/lib/lint-structural-core.ts), [의미 lint](https://github.com/nashsu/llm_wiki/blob/48fd970e206a02a6d2028d1dbfc41b7a0345bf0b/src/lib/lint.ts#L205)

### 직접 도입 판단과 다음 단계

**현재는 기존 수집·프로파일 파이프라인에 작은 Markdown 출력기를 붙이는 방향을 권한다.** LLM Wiki는 목적·페이지 틀의 참고, OpenWiki는 근거·완료·갱신 계약의 참고로 쓴다. 장비 접근·대표 표본·전용 포맷 해석·현업 승인은 어느 쪽을 쓰더라도 별도로 해결해야 한다.

OpenWiki README는 local OpenAI-compatible endpoint 지원을 설명한다. 이것만으로 사내 GLM-5.3의 tool calling과 전체 실행이 호환된다고 판단할 수는 없다. 또한 현재 visualizer는 public CDN을 쓰고 telemetry는 기본 활성화이며 끄는 설정을 제공한다. 직접 시험한다면 외부망 없는 환경에서의 화면과 통신 설정을 별도로 확인해야 한다. [현재 README](https://github.com/langchain-ai/openwiki/blob/7c469e59584bfac3f97f5c32a67604de46805999/README.md)

다음에는 가짜 장비의 파일 종류 하나로 목록 → 프로파일 → 위키 한 장 → 색인 → 검토 목록을 만들어 본다. “이 파일은 어디에 있고, 무슨 구조이며, 뜻의 근거와 미확인 부분은 무엇인가”를 그 출력만으로 찾을 수 있는지 확인하면 된다. 그래프 화면·벡터 검색·필드별 페이지는 그 뒤 필요에 따라 판단한다.

## 참고 자료 (References)

- [LLM Wiki 저장소](https://github.com/nashsu/llm_wiki) — 확인 revision `48fd970e206a02a6d2028d1dbfc41b7a0345bf0b`.
- [LangChain OpenWiki 저장소](https://github.com/langchain-ai/openwiki) — 확인 revision `7c469e59584bfac3f97f5c32a67604de46805999`.
- [기존 지도 설계](./02-map-design.md), [P1 실행 계획](./04-phased-plan.md), [결정 기록](./decision-log.md).

확인 범위는 공개 문서와 관련 소스다. 설치·실행, 사내 LLM 호출, 실제 장비 파일의 파싱·의미 정확도는 검증하지 않았다. 이 문서는 제안이며 기존 설계의 확정 결정으로 취급하지 않는다.
