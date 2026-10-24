---
tags: [orchestration, agentic-coding, pi, pi-subagents, multi-agent, hcp, llm]
level: intermediate
last_updated: 2026-10-24
reviewed_on: 2026-10-24
review_status: partial
document_type: learning
category_major: "에이전트 오케스트레이션"
category_middle: "주제 안내"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-05"
---

# Pi 오케스트레이션

> pi 코딩 에이전트로 역할을 나누고, 역할마다 모델 크기를 맞추고, 검증은 다른 세션의 다른 모델에 맡기는 사내용 구조. 실행은 `pi-subagents`가 맡는다. 이 폴더에는 사내 조건에 맞춘 설정만 있다.

## 목차와 읽기 순서

| 순서 | 문서 | 성격 | 언제 읽나 |
|---|---|---|---|
| 1 | [에이전틱 코딩: 2026년에 정리된 것들](./agentic-coding.md) | 설명 | 처음 읽을 때. 이 폴더의 설정이 왜 이렇게 생겼는지 |
| 2 | 이 개요 | 설명 | 역할, 모델 배정, 설정 파일, 대화형과 무인 모드 |
| 3 | [검토 보충](./review-notes.md) | 참조 | 실행 전에. 기준 버전, 알려진 한계, 미확인 항목 |
| 4 | [속도 제한 아래에서 오케스트레이션하기](./rate-limits.md) | 설명·사용법 | 자식 수와 동시성을 정할 때. 500K TPM·50 RPM 계산, 측정, 집의 구독 요금제 |
| 5 | [사내 PC 세팅](./office-setup.md) | 템플릿 | 회사 PC에 처음 설정할 때. `PROMPT-1`~`7` |
| 6 | [작업 저장소 AGENTS.md 규칙](./agents-md.snippet.md) | 템플릿 | 에이전트가 일할 저장소마다 |
| 7 | [결정 정책](./decisions.example.md) | 템플릿 | 무인 실행 전에 |
| 8 | [퇴근 원샷](./oneshot.md) | 템플릿 | 한 번의 무인 실행 |
| 9 | [야간 세팅](./night-setup.md) | 템플릿 | 매시간 반복하는 무인 실행. 아직 운영 검증 전 |
| - | [분류 목차](./taxonomy-index.md) · [정리 기록](./organization-log.md) · [분류 검증 기록](./classification-review.md) | 관리 기록 | 문서를 정리하거나 Obsidian에서 찾을 때 |

설정값의 원본은 JSON 스니펫 두 개(`settings.snippet.json`, `subagent-config.snippet.json`)다. `office-setup.md`의 JSON은 같은 값을 붙여넣기 좋게 옮긴 것이다. 원샷과 야간 세팅은 한 번 실행과 반복 실행으로 용도가 달라서 둘 다 둔다.

## 휴대폰, 태블릿에서 읽기

이 폴더의 문서 전체를 한 페이지로 묶은 HTML이 claude.ai에 있다: <https://claude.ai/artifact/GYzo5pa8TtCdEVKwWj4e3a>

- 비공개 링크다. 본인 claude.ai 계정으로 로그인해야 열린다.
- `.md`를 고쳐도 자동으로 반영되지 않는다. 저장소 루트에서 `python scripts/build_folder_html.py orchestration`으로 `reading.html`을 다시 만든 뒤, Claude Code에 같은 링크로 다시 게시해 달라고 요청한다.

## 왜 필요한가 (Why)

에이전트 하나에 큰 일을 통째로 맡기면 두 곳에서 무너진다. 먼저 **컨텍스트**가 넘친다. 그다음 **검증**이 사라진다. 자기가 쓴 코드를 자기가 통과시키기 때문이다.

일을 쪼개면 두 문제가 함께 풀린다. 대신 새 문제가 셋 생긴다. 누가 무엇을 하는지, 결과를 어떻게 모으는지, 둘이 같은 파일을 동시에 고치면 어떻게 되는지다.

이 셋을 처리하는 장치는 사내에 이미 설치된 `pi-subagents`에 있다. pi 본체는 서브에이전트 기능을 일부러 넣지 않았고, 확장 패키지가 이 일을 맡는다. 자식 세션 격리, 역할 정의, 도구 allowlist, 병렬 실행, 결과 수집, 자식이 부모에게 결정을 올리는 경로가 모두 들어 있다. 새로 만들 이유가 없다.

그래서 이 폴더가 더하는 것은 사내 조건 세 가지뿐이다.

1. 외부 LLM API를 못 쓴다. 그래서 사내 HCP 게이트웨이를 pi에 등록한다.
2. 크기가 다른 모델이 세 종류 있다. 그래서 어느 역할에 어느 크기를 맡길지 정한다.
3. 데이터가 밖으로 나가면 안 된다. 그래서 웹 검색 역할과, 외부 CLI(Claude Code, Codex, Cursor)를 부르는 역할을 끈다.

이 구조가 업계에서 확인된 패턴 중 어디에 해당하는지는 [에이전틱 코딩 문서](./agentic-coding.md)에 정리했다. 요약하면 **쓰기는 한 줄기로, 읽기와 검토는 나눠서**다.

## 핵심 개념 (What)

### 네 가지 설계 결정

**1. 모델 티어링이 핵심이다.** 판단에는 큰 모델, 구현에는 중간 모델, 넓게 읽고 추리는 일에는 작은 모델을 쓴다. 비용이 얼마나 줄지는 모델 가격, 재시도, 작업 성공률에 달려 있고, 이 구성에서는 재 보지 않았다. `pi-subagents`에서는 `subagents.agentOverrides.<역할>.model`로 역할마다 모델을 고정한다.

**2. 검증은 다른 세션의 다른 모델이 한다.** 리뷰어는 워커와 다른 모델로, 새 컨텍스트(fresh context)에서 띄운다. 한 에이전트의 환각이 다음 단계에서 사실로 굳는 일을 여기서 끊는다.

모델 ID가 다르다고 오류가 독립적이지는 않다. Big과 Medium은 둘 다 GLM 계열이라 같은 착각을 할 수 있다. 별칭(`HCP-*-Latest`)만 보면 계열이 보이지 않으니 실제 모델을 확인한다. 리뷰의 독립성은 모델 이름이 아니라 다음 세 가지에서 나온다.

- 새 컨텍스트에서 시작한다.
- 워커의 보고가 아니라 코드를 직접 읽는다.
- 검증 명령을 직접 다시 돌린다.

리뷰어 프롬프트가 이 셋을 요구해야 리뷰가 의미를 가진다.

빌트인 `reviewer`에는 `bash`가 없고, 프롬프트도 셸 사용을 금지한다. 0.81 기준 빌트인 프롬프트는 "셸 명령을 쓰지 말고, 돌려야 할 테스트 명령은 보고만 하라"고 지시한다. 이대로면 리뷰어는 워커의 보고를 글로만 대조하고, 검증의 절반이 빠진다. 그래서 이 설정은 `reviewer`의 `tools`에 `bash`를 되돌려 주고, `systemPrompt`도 덮어써서 검증 명령을 직접 다시 돌리게 한다. 도구만 주고 프롬프트를 그대로 두면 모델은 금지 지시를 따른다. 이 폴더가 내용을 고쳐 쓰는 역할 정의는 `reviewer` 하나다.

**3. 동시 쓰기는 파일 경계로 막는다.** `oracle`이 작업을 파일이 겹치지 않게 쪼갠다. 겹침을 피할 수 없으면 워커마다 워크트리를 따로 준다.

**4. 사내 경로를 벗어나는 역할은 끈다.** 빌트인 `researcher`와 `evidence-auditor`는 `web_search`와 `fetch_content`로 밖에 나간다. 0.75 이후 들어온 `claude-code`, `codex-exec`, `cursor-agent`(각각 `-writer` 포함)는 pi가 아니라 외부 CLI를 실행하므로, 사내 게이트웨이 설정이 적용되지 않는다. `delegate`는 `edit`을 가진 다섯 번째 쓰기 역할인데 역할별 모델 제한이 없다. 사내에서는 이 아홉 개를 `disabled: true`로 끈다. 역할을 끄는 것은 라우팅 정책이지 네트워크 차단이 아니다.

### 역할과 모델 배정

빌트인 역할을 그대로 쓰고, 모델과 thinking 수준만 사내 티어에 맞춰 배정한다.

`agentOverrides`는 기본 배정일 뿐 경계가 아니다. 실행마다 주는 override(`/run reviewer[model=...]`)가 더 우선한다. 티어를 실제로 강제하려면 `modelScope`의 `enforce`와 `strict`가 필요하다. 둘 다 설정 스니펫에 넣어 두었다.

| 빌트인 역할 | 모델 | thinking | 도구 | 하는 일 |
|------|------|----------|------|---------|
| `oracle` | `my-local-provider/HCP-Big-Latest` | high | 읽기 + bash | 판단, 분해, 방향 이탈 감시 |
| `worker` | `my-local-provider/HCP-Medium-Latest` | high | 전체 (edit 포함) | 구현, 검증 명령 실행 |
| `reviewer` | `my-local-provider/HCP-Big-Latest` | high | 읽기 + **bash**(덮어씀) + write | diff 검토, 검증 재실행, 판정 |
| `scout` | `itc-vlm/qwen3.8-27b` | low | 읽기 + bash + write | 전수 검색, 선별, 목록 작성 |
| `researcher`, `evidence-auditor` | - | - | - | **끈다** (웹 검색) |
| `claude-code`·`codex-exec`·`cursor-agent` 계열 6개 | - | - | - | **끈다** (외부 CLI) |
| `delegate` | - | - | - | **끈다** (모델 제한 없는 쓰기 역할) |

별칭은 다섯 개지만 실제 모델은 네 개다. 계열은 별칭 뒤의 실제 모델이 정하므로 함께 적어 둔다.

| 모델 | 실제 모델 | context / maxTokens | 이 구성에서의 자리 |
|------|----------|--------------------|------------------|
| `my-local-provider/HCP-Big-Latest` | GLM-5.3 | 1,048,574 / 1,048,574 | 판단(oracle), 검증(reviewer) |
| `my-local-provider/HCP-Medium-Latest` | GLM-5.3-flash | 1,048,574 / 1,048,574 | 구현(worker), watchdog |
| `my-local-provider/HCP-Small-Latest` | Qwen3.6-35B-A3B | 262,144 / 262,144 | 배정 없음. scout의 빠른 대안 |
| `my-local-provider/HCP-Vision-Latest` | Qwen3.6-35B-A3B (이미지 입력) | 262,144 / 262,144 | 배정 없음. Small과 같은 모델에 이미지 입력을 더한 것 |
| `itc-vlm/qwen3.8-27b` | Qwen3.8 27B | 262,144 / 32,768 | 정찰(scout), reviewer 교차 검증 |

#### scout에 `qwen3.8-27b`를 쓰는 이유

`A3B`는 전체 35B 중 토큰당 활성 파라미터가 3B라는 뜻이다. 활성 파라미터가 적으니 계산량은 줄 것으로 기대할 수 있지만, 실제 처리량은 엔진, 하드웨어, 배치 크기에 따라 다르므로 재 봐야 한다.

scout의 실패는 속도가 아니라 호출부 누락에서 온다. scout이 호출 지점 하나를 빠뜨리면 그 뒤의 모든 워커가 잘못된 범위로 일하고, 무인 야간 실행이면 그렇게 밤 하나가 날아간다. scout은 전체 작업의 한 단계이고 작업량은 worker가 진다. 그래서 정찰이 한 번 느린 비용보다 한 번 놓친 비용이 크다.

작업량이 많아 처리량이 급하면 실행마다 모델을 바꾼다. `modelScope`에 두 모델을 모두 열어 두었다.

```text
/run scout[model=my-local-provider/HCP-Small-Latest] "..."
```

#### `qwen3.8-27b`를 reviewer 교차 검증에도 쓰는 이유

Big과 Medium은 둘 다 GLM 계열이라 같은 착각을 할 수 있다. Qwen은 계열이 다르므로, 영향이 큰 변경에서 2차 의견으로 가치가 있다.

```text
/run reviewer[model=itc-vlm/qwen3.8-27b] "이 diff 를 다시 검증해라. 앞선 리뷰 결과는 보지 마라."
```

다만 27B는 Big보다 판단이 약하므로 주 검증이 아니라 2차 의견으로만 쓴다. 출력 상한이 32,768로 다른 모델보다 낮은 점도 감안한다. scout의 긴 목록에는 충분하지만 긴 문서를 쓰기에는 좁다.

#### `HCP-Small-Latest`와 `HCP-Vision-Latest`는 같은 모델이다

당시 구성 설명으로는 Vision 쪽이 이미지 입력을 받는다는 차이만 있다. 가중치, 템플릿, 샘플링 설정까지 같은지는 확인하지 않았으므로 텍스트 품질이 같다고 보장하지는 않는다. 스크린샷을 읽혀야 할 때는 모델을 바꾸는 것이 아니라 같은 모델의 이미지 입력 경로를 쓰는 것이다.

코딩 오케스트레이션에서 늘 쓸 일은 없어서 역할에 배정하지 않았다. UI 버그 스크린샷이나 캡처한 문서를 읽혀야 할 때만 실행마다 지정한다. scout의 `allow`에 함께 넣어 두었다.

```text
/run scout[model=my-local-provider/HCP-Vision-Latest] "이 스크린샷의 에러 메시지를 읽고 해당 코드를 찾아라"
```

활성 파라미터가 3B라는 점은 이미지 입력에서도 같다. 캡처 문서 추출처럼 정확도가 중요한 작업에 이 모델이 맞는지는 따로 판단한다.

### 도구 allowlist는 경계가 아니다

`edit`은 `worker`만 가진다. 다른 역할은 allowlist에서 `edit`을 뺐다. 그래도 다른 역할이 읽기 전용이 되는 것은 아니다.

- `write`는 기존 파일도 덮어쓸 수 있다. `edit`을 뺀다고 파일이 보호되지 않는다.
- `bash`를 가진 역할(`oracle`, `reviewer`, `scout`)은 셸로 수정, 삭제, 네트워크 호출을 모두 할 수 있다.
- `reviewer`가 워커의 검증 명령을 다시 돌리면, 워커가 방금 바꾼 테스트와 스크립트가 실행된다. `pytest`나 `npm test`를 허용하는 것은 그 안의 임의 코드를 허용하는 것이다.
- `researcher`와 `evidence-auditor`를 꺼도 외부 전송이 막히지는 않는다. `bash`가 있으면 밖으로 나갈 수 있다.

allowlist는 기본값을 정하는 장치다. 실제 경계가 필요하면 컨테이너나 VM으로 격리한다. 자세한 내용은 [원샷 문서의 "워크트리는 격리가 아니다"](./oneshot.md#워크트리는-격리가-아니다)에 있다.

## 어떻게 사용하는가 (How)

### 1. 사내 모델 등록을 확인한다

`~/.pi/agent/models.json`에 `my-local-provider`와 `itc-vlm`이 이미 등록돼 있으면 `smoke.sh -l 0`으로 확인만 하고 넘어간다.

새 모델을 추가할 때는 세 곳에서 자주 막힌다.

- 최상위는 `{"providers": {...}}`여야 한다. provider를 최상위에 바로 두면 `must have required properties providers` 오류와 함께 **파일 전체가 무시되고** 모델이 하나도 보이지 않는다.
- `apiKey`와 `headers`에서는 `$ENV_VAR` 치환과 `!command`가 된다. `baseUrl`은 치환되지 않으므로 실제 URL 문자열을 넣는다.
- `contextWindow`, `maxTokens`, `thinkingLevelMap`은 게이트웨이가 실제로 받아 주는 값과 맞춘다. 모델이 목록에 보이는 것과 긴 출력이나 thinking 요청이 성공하는 것은 별개다. `smoke.sh -l 2`로 확인한다.

### 2. `office-setup.md`의 프롬프트로 설정한다

사내 PC 설정은 [사내 PC 세팅](./office-setup.md)의 프롬프트를 pi에게 시켜서 한다. 프롬프트마다 번호가 있으므로 "office-setup.md의 PROMPT-4를 실행해라"처럼 지시한다.

| 번호 | 하는 일 |
|------|---------|
| `PROMPT-1` | 현재 설정 상태 확인 (읽기만) |
| `PROMPT-2` | `settings.json` 병합 |
| `PROMPT-3` | `extensions/subagent/config.json` 생성 |
| `PROMPT-4` | 설정 검증 15항목 |
| `PROMPT-5` | 역할 배선 확인 |
| `PROMPT-6` | 서브에이전트 왕복 |
| `PROMPT-7` | 실제 작업 한 바퀴 |

붙여넣을 JSON이 문서에 그대로 들어 있어서 모델이 값을 지어낼 여지가 없고, 단계마다 성공 기준과 보고 항목이 정해져 있다. 가장 중요한 것은 `PROMPT-4`다. 실제로 겪은 함정을 15개 항목으로 검사한다. 모두 **오류 없이 조용히 무시되는** 종류라 눈으로는 보이지 않는다.

`office-setup.md`의 JSON과 이 폴더의 스니펫은 같은 값이어야 한다. 손으로 양쪽을 고치면 어긋나므로, 스니펫을 고쳤으면 문서의 JSON 블록을 스니펫에서 다시 복사한다.

설정은 두 파일로 나뉜다. 키를 다른 파일에 넣으면 오류 없이 무시된다.

| 파일 | 넣을 것 |
|------|---------|
| `~/.pi/agent/settings.json` | `settings.snippet.json`의 내용. pi 코어의 `defaultModel`(부모 세션 모델)과 `httpIdleTimeoutMs`, 그리고 `subagents.defaultModel`(자식 기본값), `agentOverrides`, `modelScope`, `watchdog` |
| `~/.pi/agent/extensions/subagent/config.json` | `subagent-config.snippet.json`의 내용. `timeoutMs`, `toolTimeoutMs`, `asyncByDefault`와 동시성 상한 |

`agentOverrides`의 `description`은 영어로 쓴다. 취향 문제가 아니다. 이 문자열은 `subagent({ action: "list" })` 출력에 실려서, 부모가 어떤 역할을 쓸지 판단할 때 읽는다. 빌트인 역할 설명이 전부 영어라 한글을 섞으면 같은 목록 안에서 언어가 갈리고, Windows 콘솔에서 글자가 깨질 수도 있다.

같은 이유로 운영자용 메모를 `description`에 쓰지 않는다. "작업량이 많으면 다른 모델로 바꿔라" 같은 문장은 사람이 읽을 내용이다. 모델 컨텍스트에 들어가면 모델이 스스로 모델을 바꿔야 한다고 읽을 수 있다. 역할을 구분하는 데 필요한 내용만 남긴다.

### 3. 스크립트를 돌릴 수 있으면 `merge-settings.py`와 `smoke.sh`를 쓴다

bash와 python을 돌릴 수 있는 환경이면 설정을 손으로 합치지 말고 `merge-settings.py`를 쓴다. 기존 키를 보존하면서 깊은 병합을 하고, 바뀔 값을 먼저 전부 보여 준 뒤 `--apply`를 줄 때만 파일에 쓴다. 대상 파일은 `.bak`으로 백업된다.

```bash
# 1) 미리보기 - 무엇이 추가되고 무엇이 덮어써지는지만 본다
python merge-settings.py settings.snippet.json ~/.pi/agent/settings.json
python merge-settings.py subagent-config.snippet.json ~/.pi/agent/extensions/subagent/config.json

# 2) 확인했으면 적용
python merge-settings.py settings.snippet.json ~/.pi/agent/settings.json --apply
python merge-settings.py subagent-config.snippet.json ~/.pi/agent/extensions/subagent/config.json --apply
```

PowerShell에서는 `~` 대신 `$HOME`을 쓴다.

```powershell
python .\orchestration\merge-settings.py .\orchestration\settings.snippet.json $HOME\.pi\agent\settings.json
python .\orchestration\merge-settings.py .\orchestration\subagent-config.snippet.json $HOME\.pi\agent\extensions\subagent\config.json
```

미리보기에서 **덮어씀**으로 표시된 항목은 기존 값이 사라진다. 특히 `defaultModel`을 이미 다른 모델로 쓰고 있었다면 그 값이 바뀌니 확인한다. 스크립트의 동작 범위는 다음과 같다.

- 없는 디렉터리와 파일은 만든다.
- 생략한 키는 지우지 않는다.
- 같은 스니펫을 두 번 병합하면 아무것도 바뀌지 않을 수 있다.
- 대상 JSON이 깨져 있으면 덮어쓰기 전에 멈춘다.
- 동시 편집, 쓰는 중의 장애, 기존 백업 덮어쓰기에 대해서는 안전을 보장하지 않는다.

저장소 하나에만 적용하려면 settings 쪽 대상 경로를 그 저장소의 `.pi/settings.json`으로 바꾼다. 프로젝트 설정이 사용자 설정보다 우선한다.

배선이 맞는지는 눈으로 보지 말고 `smoke.sh`로 돌려서 확인한다. 계층이 올라갈수록 비용이 들므로 아래 계층부터 쓴다.

```bash
./orchestration/smoke.sh -l 0     # 설정만. LLM 호출 0회, 비용 0
./orchestration/smoke.sh          # L0 + 모델 3종 연결 (기본)
./orchestration/smoke.sh -l 2     # + 도구 호출 검증
./orchestration/smoke.sh -l 3     # + 서브에이전트 왕복. 첫 세팅 때 1회
./orchestration/smoke.sh -l 0 -v  # 실패 원인 상세
```

| 계층 | 보는 것 | 비용 |
|------|---------|------|
| L0 | `models.json` 구조, baseUrl, apiKey, 역할 4종 배정, thinking 레벨 유효성, `modelScope`와 역할 배정의 교차 검증, 꺼야 할 역할 9개, reviewer `systemPrompt`, watchdog, 런타임 상한의 파일 위치, 모델 3종 등록 | 0 |
| L1 | 역할과 대안 모델 목록의 게이트웨이 연결, 인증, 호환 설정 (모델당 1회) | 목록 크기에 따라 다름 |
| L2 | 출력에 숫자가 나오는지. 실제 도구 호출 여부는 실행 기록에서 따로 확인 | 목록 크기에 따라 다름 |
| L3 | 부모가 자식을 띄우고 결과가 돌아오는지 | 몇 회 |

L0이 실패하면 위 계층은 돌리지 않는다. 설정 오류 때문에 모델 호출을 낭비하지 않는다.

L0이 잡아내는 것 중 눈으로 보이지 않는 것은 다음과 같다.

- `models.json` 최상위에 `providers`가 없다. 파일 전체가 조용히 무시된다.
- `thinking` 레벨을 그 모델이 지원하지 않는다. 오류 없이 다른 레벨로 바뀐다. 판정 규칙은 셋이다.
  - `reasoning`이 없거나 false면 `off` 하나만 지원한다. 다른 값을 주면 thinking이 꺼진 채로 돈다.
  - `reasoning: true`인데 `thinkingLevelMap`이 없으면 `off`부터 `high`까지 지원하고 `xhigh`와 `max`만 지원하지 않는다. 정상이다.
  - `thinkingLevelMap`이 있으면 값이 `null`인 레벨을 지원하지 않는다.
- `modelScope.agents.<역할>.allow`가 그 역할의 배정 모델을 허용하지 않는다. 그 역할은 항상 실패한다.
- `agentOverrides`가 가리키는 모델이 `models.json`에 없다. 오타나 provider 혼동이 원인이다.
- `tools`를 `"read, grep, bash"` 같은 쉼표 문자열로 썼다. 빌트인 frontmatter는 이 표기를 받지만 `settings.json`은 배열만 받는다. 필드 이름이 같아도 담는 파일이 다르면 형식이 다르다.
- 역할별 `allow`에 넣은 대안 모델이 전역 `allow`에 없다. 역할 규칙은 전역 규칙을 완화하지 못하므로, 그 모델을 쓰는 순간 거부된다.
- `timeoutMs`를 `settings.json`에 넣었다. 오류 없이 무시된다.
- `apiKey`를 평문으로 적었다.

### 4. Windows에서 스크립트를 돌린다

`smoke.sh`는 bash 스크립트라 PowerShell에서 바로 실행되지 않는다. 다음 셋 중 하나를 쓴다.

```powershell
# 1) Git for Windows 가 있으면 (가장 흔함)
& "C:\Program Files\Git\bin\bash.exe" ./smoke.sh -l 0

# 2) WSL 이 있으면
wsl ./smoke.sh -l 0

# 3) Git Bash 를 열어서 평소처럼
./smoke.sh -l 0
```

한글이 깨져 보이면 콘솔 코드페이지 문제다. 스크립트 출력은 UTF-8인데 Windows 콘솔의 기본값은 949(한국어)다. 스크립트가 Git Bash나 MSYS에서 실행되면 `chcp 65001`을 알아서 건다. PowerShell에서 직접 돌리거나 그래도 깨지면 아래를 먼저 실행한다.

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

매번 치기 싫으면 PowerShell 프로필에 넣는다.

```powershell
notepad $PROFILE      # 없으면: New-Item -Path $PROFILE -ItemType File -Force
# 위 세 줄을 붙여넣고 저장, 새 창에서 적용
```

Git Bash에서도 깨지면 터미널 글꼴이 한글을 표시하지 못하는 것이다. `D2Coding`, `Malgun Gothic`, `Cascadia Mono` 중 하나로 바꾼다. Git Bash에서는 창 우클릭 → Options → Text → Font에서 바꾼다.

`python3` 없이 `python`만 있는 환경이 흔하다. 그래서 스크립트는 `python3`, `python`, `py` 순서로 찾고 Python 3인 것만 쓴다. 셋 다 없으면 멈춘다.

Windows의 `timeout.exe`는 GNU `timeout`과 다른 대기 명령이다. 무인 실행의 전체 시간 상한은 아래처럼 PowerShell로 건다. 이 예제는 여러 줄 프롬프트 전달과 자식 프로세스 종료를 검증하지 않았다. 적용하기 전에 [검토 보충](./review-notes.md)의 한계를 확인한다.

```powershell
$prompt = Get-Content .\oneshot-prompt.txt -Raw
$p = Start-Process pi -ArgumentList '-p', $prompt -PassThru -NoNewWindow
Wait-Process -Id $p.Id -Timeout 28800 -ErrorAction SilentlyContinue   # 8시간
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
```

Windows에서 무인 실행을 하려면 `smoke.sh -l 3`을 한 번은 꼭 돌린다. `pi-subagents`의 백그라운드 자식 실행은 공식 검증 대상이 Linux x64이고, Windows는 실험 단계로 표기돼 있다. npm으로 설치한 pi는 Node 러너를 쓰므로 일반적인 경로이긴 하지만, 검증 목록에 없다는 사실은 그대로다. 포그라운드(`asyncByDefault: false`)로 쓰는 편이 안전하다.

### 5. 대화형으로 쓴다 (업무 중)

부모 pi 세션이 오케스트레이터다. 자연어로 지시하면 부모가 `subagent` 도구를 호출한다.

```bash
cd /path/to/repo
pi                     # 부모 세션 = 오케스트레이터 (HCP-Big)
```

부모는 직접 일할지 자식을 띄울지 스스로 판단한다. 작은 일에는 오케스트레이션을 쓰지 않는다. 자식 하나를 띄우는 비용이 그냥 고치는 비용보다 크다.

| 일의 크기 | 이렇게 한다 |
|---|---|
| 파일 1~2개 수정, 원인이 분명함 | 부모에게 바로 시킨다. 서브에이전트 없음 |
| 어디를 고쳐야 할지 모름 | `scout`으로 먼저 찾고, 부모가 고친다 |
| 파일 여러 개, 범위가 분명함 | `worker` 하나 → `reviewer` |
| 모듈 여러 개, 쪼개야 함 | `oracle`로 분해 → `worker` 병렬 → `reviewer` |
| 이미 짠 코드 검토만 | `/parallel-review` |

기본 루프는 패키지가 권장하는 순서이고, 이 설계와 모양이 같다.

```text
clarify → scout → worker → fresh reviewer → worker
```

하루 동안 반복하는 지시는 대개 이 세 가지다.

```text
scout 으로 <경로> 아래에서 <증상/대상> 관련 파일과 호출 지점을 찾아줘. 그 밖은 뒤지지 말고, 고치지도 말고.
```
```text
worker 로 <작업>을 구현해줘. 범위는 <경로>로 한정하고, 끝나면 <검증 명령>을 돌려줘.
```
```text
reviewer 를 fresh context 로 띄워서 방금 diff 를 검증해줘.
보고를 믿지 말고 git diff 를 직접 읽고 검증 명령도 직접 다시 돌리게 해줘.
```

리뷰어가 지적하면 **같은 worker에게 되돌린다.** 부모가 직접 고치기 시작하면 역할 분리가 무너진다.

자주 쓰는 슬래시 명령은 다음과 같다.

| 명령 | 쓸 때 |
|---|---|
| `/run <역할> "<작업>"` | 역할 하나를 직접 지목. `--bg`는 백그라운드, `--fork`는 현재 대화를 물려줌 |
| `/parallel-review` | 정확성, 테스트, 복잡도를 각각 보는 리뷰어를 동시에 띄우고 결과를 합침 |
| `/review-loop` | worker → reviewer → 수정 사이클을 상한까지 자동으로 반복 |
| `/parallel-cleanup` | 구현이 끝난 뒤 검토 전용으로 한 번 더 봄 |
| `/gather-context-and-clarify` | 먼저 파악하고, 물어볼 것을 사람에게 질문 |

`/parallel-research`는 웹 검색을 하는 `researcher`를 쓰므로 사내에서는 쓸 수 없다. 꺼 두었다.

순서를 부모 모델의 판단에 맡기지 않고 고정하려면 워크플로 스크립트를 쓴다. `runs.all([{ key, agent, task }, ...])`이 여러 자식을 함께 띄우고, 결과는 넣은 순서대로 배열로 돌아온다. 파일 겹침을 피할 수 없을 때만 워커마다 워크트리를 따로 준다.

사람이 옆에 있으니 막히면 묻는다. 자식이 `contact_supervisor`로 올린 결정을 부모가 사람에게 그대로 전달한다. 결정 정책 파일도 결정 로그도 필요 없다. 그 자리에서 답하면 된다.

`asyncByDefault: false`로 둔 이유도 여기 있다. 자식이 포그라운드로 돌아야 진행을 보면서 끊을 수 있다. 오래 걸리는 것만 `--bg`로 떼어 내고 `subagent({ action: "status" })`로 확인한다.

### 6. 무인으로 쓴다 (퇴근 원샷, 야간 반복)

설정은 대화형과 같고, 프롬프트에 담는 규칙만 다르다. 달라지는 것은 세 가지다.

1. **결정을 미리 위임한다.** `.orch/decisions.md`에 A(미리 승인), B(기본값 규칙), C(스스로 판단 금지)의 3단 정책을 써 둔다. 자식이 막히면 부모가 이 파일을 근거로 답한다. 스스로 정한 것은 되돌리는 방법까지 `.orch/결정로그.md`에 남긴다.

   판정은 **하드 가드 → D → C → A → B** 순서로만 한다. D는 사람이 미리 구체적으로 답한 것이다. C가 "사람에게 물어라"라는 뜻이라면 D는 이미 물어서 받은 답이므로 C보다 앞에 둔다. A를 먼저 보면, "기존 패턴 따르기(A)"이면서 "인증 로직 변경(C)"인 변경이 승인되어 버린다.

   세 모드가 실제로 다른 곳은 C 목록을 어디까지 적용하느냐다. `보수`는 C를 만나면 전체를 멈춘다. `균형`은 그 단위만 멈춘다. `자율`은 C를 하드 가드까지만 좁히고, 나머지는 B 규칙으로 스스로 정하고 계속 간다. `자율`의 "전부 스스로 판단"은 "전부 허용"이 아니라 "멈추지 않음"이다.

   정책은 프롬프트 본문에 그대로 붙여넣는다. 파일을 작업 트리 밖에 두고 `chmod 444`를 걸어도, 같은 사용자 권한의 `bash`가 되돌릴 수 있어서 실수 방지 수준에 그친다. 컨텍스트에 들어온 원문이 기준이어야 파일을 바꿔치기해도 의미가 없어진다.
2. **워크트리를 나누고 단위마다 커밋한다.** 단위 하나를 통째로 revert할 수 있어야 반쯤 고친 호출부가 남지 않는다. 커밋할 때는 `git add -A`가 아니라 그 단위의 경로만 올린다. 전체를 올리면 다른 워커의 미완성 변경이 섞여서, revert가 남의 작업까지 지운다. 워커 둘 이상을 동시에 돌릴 때는 워커마다 워크트리를 따로 준다. push만 금지한다. 워크트리는 되돌리기 수단이지 격리가 아니다. 홈 디렉터리, SSH 키, 네트워크에는 그대로 닿는다.
3. **상한을 건다.** `timeoutMs`는 subagent 실행 하나의 제한이지 밤 작업 전체의 상한이 아니다. 부모가 단위를 계속 만들면 전체 시간은 계속 늘어난다. 전체 시간 상한은 `timeout 8h pi`처럼 OS 수준에서 건다.

프롬프트 템플릿은 [퇴근 원샷](./oneshot.md)에, 정책 템플릿은 [결정 정책](./decisions.example.md)에 있다.

작업 스케줄러로 매시간 돌릴 때는 한 번에 전부 끝내는 프롬프트를 쓰지 않는다. 회차가 겹치면 같은 워크트리를 두 에이전트가 고친다. 대신 [야간 세팅](./night-setup.md)의 `NIGHT-1`~`NIGHT-5`를 쓴다. 회차당 한 단위만 하고, 이전 회차가 살아 있으면 건너뛰고, `.orch\STOP` 파일로 끝낸다.

무인이든 아니든 `subagents.watchdog`은 켜 둔다. 턴이 끝날 때마다 다른 모델이 방금 한 일을 되짚어서 범위 이탈, 반복 위험, 위험한 변경을 찾는다. 단 watchdog은 사후 검토이지 실행 전 차단이 아니다. 전송이나 삭제가 일어난 뒤에 지적하므로 하드 가드를 대신하지 못한다. `watchdog.main.model`을 생략하면 세션 모델을 물려받으니, 부모와 다른 모델을 명시한다. 스니펫에 넣어 두었다. 중간 점검이 필요하면 `cadence.everyNTools`를 함께 켠다. 이 옵션은 호출 수를 크게 늘리므로 [속도 제한 문서](./rate-limits.md)를 먼저 본다.

원샷에 `pi-subagents`를 쓰는 이유는 분해 방법을 미리 알 수 없어서다. 원샷 프롬프트는 "이거 해 둬"에 가깝고, 워커를 몇 개 띄울지는 저장소를 읽어 본 뒤에 정해진다. 부모 세션이 그 판단을 하고 자식을 띄우는 것이 Supervisor 패턴의 핵심이고, 순서가 고정된 파이프라인으로는 할 수 없다.

### 7. 사내 규칙은 작업 저장소의 `AGENTS.md`에 둔다

빌트인 역할 4종은 모두 `systemPromptMode: replace`다. 자기 프롬프트로 시스템 프롬프트를 교체하므로 별도 역할 파일을 끼워 넣을 자리가 없다. 대신 넷 다 `inheritProjectContext: true`라서, 작업 대상 저장소의 `AGENTS.md`는 모든 자식에게 항상 전달된다.

"배정된 파일 밖은 수정하지 않는다", "`git add -A`를 쓰지 않는다", "검증 명령을 지어내지 않는다" 같은 규칙은 그곳에 둔다. 원샷 프롬프트에만 적으면 대화형으로 쓸 때 자식에게 전달되지 않는다. 넣을 내용은 [작업 저장소 AGENTS.md 규칙](./agents-md.snippet.md)에 있다.

자식은 스킬을 물려받지 않는다(`inheritSkills: false`). 스킬에 의존하는 규칙은 쓰지 않는다.

### 8. 지키는 것

- 리뷰어가 지적하면 워커에게 되돌린다.
- 커밋은 사람이 한다.
- 워커를 늘리기 전에, 작업이 정말 파일 단위로 나뉘는지 먼저 확인한다.

## 운영 가드레일

| 항목 | 규칙 |
|------|------|
| 외부 전송 | `researcher`와 `evidence-auditor`를 끈다. `bash`가 있으면 우회할 수 있으므로 실제 차단은 네트워크 정책으로 한다. |
| 자격 증명 | API 키는 환경 변수(`$HCP_API_KEY`, `$ITC_VLM_API_KEY`)로만 준다. `models.json`과 저장소에 키를 넣지 않는다. |
| 샌드박스 | pi에는 샌드박스도 권한 확인 창도 없다. 자식은 실행한 사용자의 권한을 그대로 가진다. 하드 가드는 프롬프트 문구일 뿐 강제되지 않는다. 실제 경계가 필요하면 컨테이너나 VM을 쓴다. |
| 비밀 값 | 환경 변수에 둔 API 키는 `bash`로 읽힌다. 에이전트에게서 숨겨야 하면 격리 경계 밖의 프록시가 키를 가진다. |
| 비용 | 큰 모델은 `oracle`과 `reviewer`에만 쓴다. 자식 기본 모델(`subagents.defaultModel`)은 Medium으로 둔다. |
| 우회 금지 | 자식 실행이 깨지면 외부 CLI나 다른 실행 경로로 우회하지 않는다. 실패를 그대로 보고한다. |

## 속도 제한 (RPM / TPM)

사내 게이트웨이는 **분당 500,000토큰, 50요청**으로 묶여 있다. 에이전트는 매 요청에 지금까지의 대화를 압축 후 남은 만큼 다시 보낸다. 그래서 요청 하나가 10,000토큰을 넘는 순간부터는 요청 수가 아니라 토큰 수가 먼저 찬다. 코딩 에이전트는 거의 항상 그렇다.

계산 방법, 자식 수를 정하는 공식, 측정 스크립트, 재시도 설정, 운영팀 확인 목록은 [속도 제한 아래에서 오케스트레이션하기](./rate-limits.md)에 있다. 집에서 구독 요금제(5시간 창)로 쓸 때의 차이도 같은 문서에 있다. 요점은 네 가지다.

- **pi에는 분당 토큰·요청 제한기가 없다.** 아래 값은 자식을 몇 개 띄울지만 정한다. 다른 pi 세션과 다른 harness는 서로의 사용량을 모른다.
- **동시성보다 자식의 컨텍스트 크기와 요청 주기가 먼저다.** 요청당 60K 토큰이면 분당 8번이 한계다. 자식 하나가 30초마다 요청하면 둘까지 돌릴 수 있고, 10초마다 요청하면 하나도 벅차다.
- **게이트웨이가 캐시 적중 토큰을 세는지 먼저 확인한다.** 에이전트 요청은 대부분 캐시 적중이라, 이 답에 따라 같은 작업의 집계량이 몇 배씩 달라진다.
- **재시도는 두 층이 곱해진다.** 현재 값이면 생성 하나에 최악의 경우 35번 요청이 나간다.

| 설정 | 파일 | 기본값 | 이 구성 | 이유 |
|------|------|--------|---------|-----|
| `globalConcurrencyLimit` | config.json | 20 | **2** | 실행 하나 안에서 동시에 도는 자식 수 |
| `parallel.concurrency` | config.json | 4 | **2** | 병렬 그룹의 동시 실행 수 |
| `maxActiveAsyncRunsPerSession` | config.json | 무제한 | **1** | 동시에 도는 최상위 비동기 실행 수 |
| `maxSubagentSpawnsPerRun` | config.json | 64 | **12** | 실행 하나에서 띄울 수 있는 자식 총수. 폭주 방지 |
| `retry.maxRetries` | settings.json | 3 | **6** | 429는 정상적으로 발생하므로 재시도로 넘긴다 |
| `retry.baseDelayMs` | settings.json | 2000 | **8000** | 2초 뒤 재시도는 분당 한도에서 다시 429를 부른다 |
| `retry.provider.maxRetries` | settings.json | 0 | **4** | HTTP 계층 재시도. `Retry-After` 헤더를 따른다 |

스니펫은 업무 중(대화형) 기준이다. 무인 실행 값과 프로필별 예산은 [속도 제한 문서의 예산 절](./rate-limits.md#3-예산을-나눠서-정한다)에 있다.

watchdog은 눈에 잘 띄지 않는 소비자다. 턴이 끝날 때마다 별도 모델 호출이 들어가고, 이 호출의 사용량은 세션 로그에 남지 않는다. 그래서 cadence는 끄고 턴 경계 검토만 남겼고, 자식 watchdog은 아예 껐다. 자식의 품질은 watchdog이 아니라 `reviewer`가 맡는다.

## 알아둘 동작

- 모델은 이 순서로 정해진다: 실행마다 준 override → provider별 역할 override → `agentOverrides.<역할>.model` → 에이전트 frontmatter → `subagents.defaultModel` → 부모 세션 모델.
- `agentOverrides`로 덮어쓸 수 있는 필드에는 `tools`, `thinking`, `model`, `description`, `disabled`, `systemPrompt`가 있다. 역할 파일을 통째로 복사할 필요가 없다.
- 빌트인 역할을 통째로 고쳐야 하면 `subagent({ action: "eject", agent: "reviewer" })`로 사용자 디렉터리에 복사본을 꺼낸다.
- 사용자 역할은 `~/.pi/agent/agents/**/*.md`, 프로젝트 역할은 `.pi/agents/**/*.md`에 둔다.
- 0.79부터는 같은 Pi 프로세스에서 세션을 넘겨받아도 이전 세션의 백그라운드 실행을 보고, 기다리고, 멈출 수 있다. 끝난 워커를 한 번 더 돌리는 재개도 된다.
- `/subagents-doctor`는 설치, 설정, 비동기 용량을 한 번에 점검한다. `smoke.sh`와 겹치지만 패키지가 직접 보는 값이라 함께 본다.
- 새 세션에서 `subagent` 도구 대신 `subagents_enable`만 보일 수 있다(`toolActivation: "auto"`). 대화 중에 도구를 추가할 수 있는 모델이면 필요할 때 켜는 방식이다. 배선 오류가 아니다. `subagents_enable`을 먼저 부르면 된다.
- `checkpointBeforeDeadlineMs`는 시간 상한 직전에 "정리하고 멈춰라" 신호를 준다. 비동기 단일 실행에만 걸리고, 도구 호출 사이에만 전달되며, 전달을 보장하지 않는다. 기본 포그라운드 실행은 보호하지 않는다.
- 자식 도구 권한(`permissions`의 `allow`, `ask`, `deny`)은 선택 기능이다. `ask`는 자식 watchdog 모델이 판정하는데, 이 설정은 자식 watchdog을 꺼 두었으므로 `ask`는 전부 거부된다. `bash`는 이 기능으로 막을 수 없다.
- `pi --list-models <이름>`은 못 찾으면 `No models matching "이름"`이라고 답한다. 안내문에 검색어가 그대로 들어 있어서, 단순 grep으로 존재 여부를 확인하면 항상 통과한다. 출력 표의 provider와 model 열로 대조한다.

## 검증 상태

2026-10-24 기준이다.

- 이번 갱신의 기준은 **pi 1.4.0 + pi-subagents 0.81.0**이다. 로컬 설치의 문서와 빌트인 역할 파일과 대조했다. 사내 PC에서 0.81로 다시 돌려 보지는 않았다.
- 0.81 빌트인 `reviewer`의 프롬프트가 셸 사용을 금지한다는 것을 로컬 파일(`agents/reviewer.md`)에서 확인했다. 그래서 `systemPrompt` 덮어쓰기를 추가했고, `smoke.sh -l 0`이 이것을 검사한다.
- 아래 2026-09-22 사내 통과 기록은 그 당시 버전의 기록이다. 0.81로 갱신한 뒤에는 `PROMPT-4`~`7`을 다시 돌린다.
- `models.json` 스키마(최상위 `providers` 필수)와 `--list-models` 동작은 pi 0.86.1로 직접 확인했다.
- thinking 레벨이 조정되는 방식은 pi 소스(`models.js`의 `getSupportedThinkingLevels`, `clampThinkingLevel`)로 확인했다.
- `smoke.sh`는 `PI_CODING_AGENT_DIR`로 provider 2개와 모델 5개 구성을 흉내 내서, 통과(ok 43)와 잘못된 값을 넣었을 때의 검출을 둘 다 확인했다.
- 사내 환경에서 `office-setup.md`의 `PROMPT-1`~`PROMPT-7`이 전부 통과했다(2026-09-22). 설정 검증 15항목, 역할 배선, 서브에이전트 왕복, 실제 저장소에서의 작업 한 바퀴까지 확인했다. 대화형 경로는 검증된 상태다.
- 야간 스케줄러 경로(`night-setup.md`, `night-run.ps1`)는 아직 실행 검증 전이다. 작성 환경에 pwsh가 없어서 `night-run.ps1`은 문법 검사도 하지 못했다. `NIGHT-4`의 수동 실행으로 꼭 확인한다.
- 설계는 Codex(gpt-6-astra)와 4라운드 검토했고, 지적받은 P1 20건을 반영했다.
- 2026-10-24 갱신은 Codex와 웹 조사, 계획 검토를 나눠 진행했다. 계획 검토의 P1 4건(reviewer 프롬프트, `delegate`, 외부 CLI 역할, 내장 스케줄에 대한 과장)을 반영했다.

## 참고 자료

- [에이전틱 코딩: 2026년에 정리된 것들](./agentic-coding.md): 이 설계의 근거가 된 연구와 업체 자료 목록
- [Pi 공식 문서](https://pi.dev/docs/latest) · [Providers](https://pi.dev/docs/latest/providers) · [Security](https://pi.dev/docs/latest/security)
- `pi-subagents` 문서: 설치 경로의 `docs/` (`models.md`, `agents.md`, `workflows.md`, `configuration.md`)
- [6 Multi-Agent Orchestration Patterns for Production (2026)](https://beam.ai/agentic-insights/multi-agent-orchestration-patterns-production)

## 관련 문서

- [에이전틱 코딩 배경](./agentic-coding.md) · [속도 제한 아래에서 오케스트레이션하기](./rate-limits.md) · [사용량 집계 스크립트](./rate-usage.py)
- [사내 PC 세팅 프롬프트 (PROMPT-1~7)](./office-setup.md): 주 경로
- [세팅 점검 스크립트](./smoke.sh) · [설정 병합 스크립트](./merge-settings.py)
- [역할별 티어 배선](./settings.snippet.json) · [런타임 상한](./subagent-config.snippet.json)
- [퇴근 원샷 프롬프트](./oneshot.md) · [결정 정책 템플릿](./decisions.example.md)
- [야간 무인 세팅 프롬프트 (NIGHT-1~5)](./night-setup.md) · [스케줄러 래퍼](./night-run.ps1)
- [작업 저장소 AGENTS.md 규칙](./agents-md.snippet.md)
