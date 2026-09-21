---
tags: [orchestration, pi, pi-subagents, multi-agent, hcp, llm]
level: intermediate
last_updated: 2026-09-21
---

# Pi 오케스트레이션

> pi 코딩 에이전트로 역할을 나누고, 모델 크기를 역할에 맞춰 배분하고, 검증을 다른 모델에 맡기는 사내용 구조. 실행은 `pi-subagents` 가 맡고, 이 폴더는 **사내 조건에 맞춘 배선**만 담는다.

## 왜 필요한가 (Why)

에이전트 하나에 큰 일을 통째로 맡기면 두 군데서 무너진다. **컨텍스트**가 먼저 넘치고, 그 다음 **검증**이 사라진다. 자기가 쓴 코드를 자기가 통과시키기 때문이다.

쪼개면 두 문제가 같이 풀린다. 다만 쪼갠 뒤에 새 문제가 생긴다 — 누가 무엇을 하는지, 결과를 어떻게 걷는지, 동시에 같은 파일을 고치면 어떻게 되는지.

이 셋을 처리하는 기계 장치는 **이미 사내에 설치된 `pi-subagents`** 가 갖고 있다. 자식 세션 격리, 역할 정의, 도구 allowlist, 병렬 팬아웃, 결과 수집, 자식→부모 에스컬레이션이 전부 들어 있다. 새로 만들 이유가 없다.

그래서 이 폴더가 실제로 더하는 것은 **사내 조건 세 가지**뿐이다.

1. 외부 LLM API를 못 쓴다 → 사내 HCP 게이트웨이를 pi에 등록해야 한다.
2. 크기가 다른 모델 세 개가 있다 → **어느 역할에 어느 크기를 물릴지** 정해야 한다.
3. 외부로 나가면 안 된다 → 웹 검색을 쓰는 역할을 꺼야 한다.

## 핵심 개념 (What)

### 설계 결정

**1. 모델 티어링이 전부다.** 판단에는 큰 모델, 양산에는 중간 모델, 읽고 추리는 일에는 작은 모델. 전부 큰 모델로 돌릴 때 대비 비용이 절반 아래로 떨어진다는 것이 2026년 프로덕션 사례들의 공통 보고다. `pi-subagents` 는 `subagents.agentOverrides.<역할>.model` 로 역할마다 모델을 못박게 해준다.

**2. 검증은 다른 세션·다른 모델이 한다.** 리뷰어를 워커와 다른 모델로, fresh context 로 띄운다. 한 에이전트의 환각이 다음 단계에서 사실로 굳는 연쇄를 여기서 끊는다.

**모델 ID가 다르다고 오류가 독립적이지는 않다.** Big·Medium 은 둘 다 GLM 계열이라 같은 착각을 공유할 수 있다. 별칭(`HCP-*-Latest`)만 보면 계열이 안 보이므로 실제 모델을 확인해야 한다. 독립성의 실질은 모델 이름이 아니라 ① 새 컨텍스트 ② 워커 보고가 아닌 **코드를 직접 읽는 것** ③ 검증 명령을 **직접 다시 돌리는 것** 에서 나온다. 리뷰어 프롬프트가 이 셋을 요구해야 의미가 있다.

빌트인 `reviewer` 에는 **`bash` 가 없다.** 읽기만 가능해서 검증 명령을 다시 돌리지 못한다. 그러면 워커 보고를 글로만 대조하게 되고, 게이트의 절반이 사라진다. 그래서 이 설정은 `reviewer` 의 `tools` 를 덮어써 `bash` 를 되돌려준다 — 이 폴더가 손대는 유일한 역할 정의다.

**3. 동시 쓰기는 파일 경계로 막는다.** `oracle` 이 **파일이 겹치지 않게** 쪼갠다. 겹침을 피할 수 없으면 워커마다 별도 워크트리를 준다.

**4. 외부로 나가는 역할은 끈다.** 빌트인 `researcher` 와 `evidence-auditor` 는 `web_search` / `fetch_content` 로 바깥에 나간다. 사내에서는 `disabled: true` 로 막는다.

### 역할과 모델 배정

빌트인 역할을 그대로 쓰고, 모델과 thinking 을 사내 티어로 배정한다.

`agentOverrides` 는 **기본 배정일 뿐 경계가 아니다.** 런당 override(`/run reviewer[model=...]`)가 더 우선한다. 티어를 실제로 강제하려면 `modelScope`(`enforce`/`strict`)가 필요하고, 설정 스니펫에 함께 넣어 뒀다.

| 빌트인 역할 | 모델 | thinking | 도구 | 하는 일 |
|------|------|----------|------|---------|
| `oracle` | `my-local-provider/HCP-Big-Latest` | high | 읽기 + bash | 판단·분해·드리프트 감시 |
| `worker` | `my-local-provider/HCP-Medium-Latest` | high | 전체 (edit 포함) | 구현, 검증 명령 실행 |
| `reviewer` | `my-local-provider/HCP-Big-Latest` | high | 읽기 + **bash**(덮어씀) + write | diff 검토, 검증 재현, 판정 |
| `scout` | `itc-vlm/qwen3.8-27b` | low | 읽기 + bash + write | 전수 검색, 선별, 목록화 |
| `researcher` / `evidence-auditor` | — | — | — | **꺼둔다** (외부 네트워크) |

별칭은 다섯이지만 **실제 모델은 넷**이다. 별칭 뒤의 모델이 계열을 결정하므로 같이 적어 둔다.

| 모델 | 실제 모델 | context / maxTokens | 이 구성에서의 자리 |
|------|----------|--------------------|------------------|
| `my-local-provider/HCP-Big-Latest` | GLM-5.3 | 1,048,574 / 1,048,574 | 판단(oracle) · 검증(reviewer) |
| `my-local-provider/HCP-Medium-Latest` | GLM-5.3-flash | 1,048,574 / 1,048,574 | 구현(worker) · watchdog |
| `my-local-provider/HCP-Small-Latest` | Qwen3.6-35B-A3B | 262,144 / 262,144 | 배정 없음. scout 의 **빠른 대안** |
| `my-local-provider/HCP-Vision-Latest` | Qwen3.6-35B-A3B (이미지 입력) | 262,144 / 262,144 | 배정 없음. Small 과 **같은 모델** + 이미지 |
| `itc-vlm/qwen3.8-27b` | Qwen3.8 27B | 262,144 / 32,768 | 정찰(scout) · reviewer 교차 검증 |

**scout 에 `HCP-Small-Latest` 가 아니라 `qwen3.8-27b` 를 쓰는 이유.** `A3B` 는 총 35B 중
**토큰당 활성 파라미터가 3B** 라는 뜻이다. 처리량은 이쪽이 훨씬 좋지만, scout 의 실패 모드는
속도가 아니라 **호출부 누락**이다. scout 이 호출 지점을 하나 빠뜨리면 그 뒤 모든 워커가
잘못된 범위를 공유하고, 무인 야간 실행이면 그대로 밤 하나를 날린다. scout 은 전체의 한 단계일
뿐이고 물량은 worker 가 지므로, 정찰이 한 번 느린 비용보다 한 번 놓친 비용이 크다.

물량이 많아 처리량이 급하면 런당 교체한다. `modelScope` 에 둘 다 열어 뒀다.

```text
/run scout[model=my-local-provider/HCP-Small-Latest] "..."
```

**`qwen3.8-27b` 를 reviewer 교차 검증에도 쓰는 이유.** Big·Medium 은 둘 다 GLM 계열이라 같은
착각을 공유할 수 있다. qwen 은 계열이 다르므로 영향이 큰 변경에 2차 의견으로 값이 있다.

```text
/run reviewer[model=itc-vlm/qwen3.8-27b] "이 diff 를 다시 검증해라. 앞선 리뷰 결과는 보지 마라."
```

다만 27B 는 Big 보다 판단이 약하므로 **주 게이트가 아니라 2차 의견으로만** 쓴다. 출력 상한이
32,768 로 다른 모델보다 낮은 것도 감안한다 — scout 의 긴 목록에는 충분하지만 장문 산출에는 좁다.

**`HCP-Small-Latest` 와 `HCP-Vision-Latest` 는 같은 모델이다.** Vision 쪽이 이미지 입력을 받는
것뿐이라 텍스트 품질 차이는 없다. 그래서 스크린샷을 읽혀야 하면 **모델을 바꾸는 것이 아니라
같은 모델의 이미지 입구를 쓰는 것**이고, 판단 품질이 달라지지 않는다.

코딩 오케스트레이션에서 상시로 쓸 일은 없어 역할에 배정하지 않았다. UI 버그 스크린샷이나
캡처한 문서를 읽혀야 할 때만 런당 지정한다. scout 의 `allow` 에 함께 넣어 뒀다.

```text
/run scout[model=my-local-provider/HCP-Vision-Latest] "이 스크린샷의 에러 메시지를 읽고 해당 코드를 찾아라"
```

다만 활성 파라미터가 3B 라는 점은 이미지에서도 같다. 캡처 문서 추출처럼 정확도가 중요한
작업에 이 모델을 그대로 쓰는 것이 맞는지는 별도로 판단할 일이다.

`worker` 만 `edit` 을 가진다. 나머지 역할은 allowlist 에서 `edit` 을 뺐다.

**다만 이것은 읽기 전용이 아니다.** 정확히 하자면:

- `write` 는 **기존 파일도 덮어쓸 수 있다.** `edit` 을 뺀다고 파일이 보호되지 않는다.
- `bash` 를 가진 역할(oracle·reviewer·scout)은 셸로 수정·삭제·네트워크 호출이 모두 가능하다.
- 특히 `reviewer` 가 워커의 검증 명령을 재실행하면 **워커가 방금 바꾼 테스트·스크립트가 실행된다.**
  `pytest` 나 `npm test` 를 허용한다는 것은 그 안의 임의 코드를 허용한다는 뜻이다.
- `researcher`·`evidence-auditor` 를 끄는 것도 외부 전송 차단이 아니다. `bash` 가 있으면 나갈 수 있다.

allowlist 는 **기본값을 정하는 장치**이지 경계가 아니다. 진짜 경계가 필요하면 컨테이너/VM 격리가
필요하다 — [oneshot.md 의 "워크트리는 격리가 아니다"](./oneshot.md) 참고.

## 어떻게 사용하는가 (How)

### 1. 사내 모델 등록 (이미 돼 있으면 확인만)

`~/.pi/agent/models.json` 에 `my-local-provider` 와 `itc-vlm` 두 provider 가 이미 등록돼 있다면 이 단계는 건너뛰고 `smoke.sh -l 0` 으로 확인만 한다.

새 모델을 추가할 때 걸리는 곳은 셋이다.

- **최상위는 `{"providers": {...}}` 다.** provider 를 바로 올리면 `must have required properties providers` 로 **파일 전체가 무시되고** 모델이 하나도 안 보인다.
- `apiKey` 와 `headers` 는 `$ENV_VAR` 치환이 되고 `!command` 도 된다. **`baseUrl` 은 치환이 안 된다** — 실제 URL 문자열을 넣어야 한다.
- `contextWindow` / `maxTokens` / `thinkingLevelMap` 은 게이트웨이가 실제로 받아주는 값과 맞춰야 한다. 목록에 보이는 것과 긴 출력·thinking 요청이 성공하는 것은 별개다. `smoke.sh -l 2` 로 확인한다.

### 2. 역할별 티어 배선

설정이 **두 파일로 나뉜다.** 키를 엉뚱한 파일에 넣으면 오류 없이 조용히 무시되니 주의한다.

| 파일 | 넣을 것 |
|------|---------|
| `~/.pi/agent/settings.json` | `settings.snippet.json` 내용 — pi 코어의 `defaultModel`(**부모 세션 모델**), `httpIdleTimeoutMs`, 그리고 `subagents.defaultModel`(**자식 기본값**), `agentOverrides`, `modelScope`, `watchdog` |
| `~/.pi/agent/extensions/subagent/config.json` | `subagent-config.snippet.json` 내용 — `timeoutMs`, `toolTimeoutMs`, `asyncByDefault` |

`agentOverrides` 의 `description` 은 영어로 쓴다. 취향 문제가 아니라, 이 문자열이
`subagent({ action: "list" })` 출력에 실려 **부모가 어떤 역할을 쓸지 판단할 때 읽는 텍스트**이기
때문이다. 빌트인 역할 설명이 전부 영어라 한글을 섞으면 같은 목록에서 언어가 갈리고, Windows
콘솔에서 깨질 여지도 생긴다. 같은 이유로 **운영자용 메모를 여기 쓰지 않는다** — "물량이 급하면
다른 모델로 바꿔라" 같은 문장은 사람이 읽을 내용인데, 모델 컨텍스트에 들어가면 모델이 스스로
모델을 교체해야 한다고 읽을 수 있다. 역할 구분에 필요한 것만 남긴다.

bash·python 을 돌릴 수 있으면 손으로 합치지 말고 `merge-settings.py` 를 쓴다.
(사내 PC 처럼 스크립트 실행이 번거로우면 `office-setup.md` 의 `PROMPT-2` 로 한다.) 기존 키를 보존하면서 깊은 병합을 하고,
**바뀔 값을 먼저 전부 보여준 뒤** `--apply` 를 줄 때만 기록한다. 대상 파일은 `.bak` 으로 백업된다.

```bash
# 1) 미리보기 - 무엇이 추가되고 무엇이 덮어써지는지만 본다
python merge-settings.py settings.snippet.json ~/.pi/agent/settings.json
python merge-settings.py subagent-config.snippet.json ~/.pi/agent/extensions/subagent/config.json

# 2) 확인했으면 적용
python merge-settings.py settings.snippet.json ~/.pi/agent/settings.json --apply
python merge-settings.py subagent-config.snippet.json ~/.pi/agent/extensions/subagent/config.json --apply
```

PowerShell 에서도 그대로 쓴다 (`~` 대신 `$HOME`).

```powershell
python .\orchestration\merge-settings.py .\orchestration\settings.snippet.json $HOME\.pi\agent\settings.json
python .\orchestration\merge-settings.py .\orchestration\subagent-config.snippet.json $HOME\.pi\agent\extensions\subagent\config.json
```

**덮어씀** 항목은 기존 값이 사라진다는 뜻이니 반드시 확인한다. 특히 `defaultModel` 을 이미
다른 모델로 쓰고 있었다면 그 값이 바뀐다. 없는 디렉터리·파일은 알아서 만든다. 두 번 돌려도
안전하고, 대상 파일이 깨진 JSON 이면 덮어쓰지 않고 멈춘다.

저장소 단위로만 적용하려면 settings 쪽 대상 경로를 그 저장소의 `.pi/settings.json` 으로 바꾼다
(프로젝트 설정이 사용자 설정을 이긴다).

### 2-0. 세팅은 `office-setup.md` 로 진행한다 (주 경로)

사내 세팅은 [office-setup.md](./office-setup.md) 의 프롬프트를 pi 에게 시켜서 한다.
각 프롬프트에 `PROMPT-1` ~ `PROMPT-7` 번호가 붙어 있어 **"office-setup.md 의 PROMPT-4 를
실행해라"** 처럼 지시하면 된다.

| 번호 | 하는 일 |
|------|---------|
| `PROMPT-1` | 현재 설정 상태 확인 (읽기만) |
| `PROMPT-2` | `settings.json` 병합 |
| `PROMPT-3` | `extensions/subagent/config.json` 생성 |
| `PROMPT-4` | **설정 검증 12항목** |
| `PROMPT-5` | 역할 배선 확인 |
| `PROMPT-6` | 서브에이전트 왕복 |
| `PROMPT-7` | 실제 작업 한 바퀴 |

붙여넣을 JSON 이 문서에 그대로 박혀 있어 모델이 값을 지어낼 여지가 없고, 단계마다 성공
판정과 보고 항목이 정해져 있다. `PROMPT-4` 가 핵심이다 — 지금까지 실제로 발목을 잡은
함정 12가지를 항목별로 검사한다. 전부 **오류 없이 조용히 무시되는** 종류라 눈으로는 안 보인다.

`office-setup.md` 의 프롬프트와 이 폴더의 스니펫은 **같은 원본에서 생성한다.** 손으로 양쪽을
고치면 반드시 어긋나므로, 스니펫을 고쳤으면 문서의 JSON 블록도 다시 생성한다.

### 2-1. 스크립트로 검증할 수 있으면 (`smoke.sh`)

배선이 맞았는지는 눈으로 보지 말고 돌려서 확인한다. 계층이 올라갈수록 비싸므로 아래에서부터 쓴다.

```bash
./orchestration/smoke.sh -l 0     # 설정만. LLM 호출 0회, 비용 0
./orchestration/smoke.sh          # L0 + 모델 3종 연결 (기본)
./orchestration/smoke.sh -l 2     # + 도구 호출 검증
./orchestration/smoke.sh -l 3     # + 서브에이전트 왕복. 첫 세팅 때 1회
./orchestration/smoke.sh -l 0 -v  # 실패 원인 상세
```

| 계층 | 보는 것 | 비용 |
|------|---------|------|
| L0 | `models.json` 구조·baseUrl·apiKey, 역할 4종 배정, **thinking 레벨 유효성**, `modelScope` ↔ 역할 배정 교차 검증, watchdog, 런타임 상한이 맞는 파일에 있는지, 모델 3종 등록 | 0 |
| L1 | 게이트웨이 연결·인증·compat (모델당 1회) | 3회 호출 |
| L2 | tool calling 이 게이트웨이에서 실제로 되는지 | 3회 호출 |
| L3 | 부모가 자식을 띄우고 결과가 돌아오는지 | 수 회 |

L0 이 실패하면 위 계층은 돌리지 않는다. 설정 오류로 낭비할 호출이 없다.

L0 이 잡아내는 것 중 눈으로는 안 보이는 것들:

- `models.json` 최상위 `providers` 래퍼 누락 → **파일 전체가 조용히 무시된다**
- `thinking` 레벨이 그 모델에서 미지원 → 오류 없이 다른 레벨로 바뀐다. 판정 규칙은 셋이다.
  - `reasoning` 이 없거나 false → **`off` 하나만 지원**. 다른 값을 주면 thinking 이 통째로 꺼진 채 돈다
  - `reasoning: true` 인데 `thinkingLevelMap` 이 **없으면** → `off`~`high` 지원, `xhigh`·`max` 만 미지원 (정상)
  - `thinkingLevelMap` 이 있으면 → 값이 `null` 인 레벨이 미지원
- `modelScope.agents.<역할>.allow` 가 그 역할의 배정 모델과 어긋남 → 그 역할이 **항상** 실패한다
- `agentOverrides` 가 가리키는 모델이 `models.json` 에 없음 (오타·provider 혼동)
- `tools` 를 `"read, grep, bash"` 같은 쉼표 문자열로 씀 → **빌트인 frontmatter 는 이 표기를 받지만 `settings.json` 은 배열만 받는다.** 같은 필드명이라도 담는 파일이 다르면 형식이 다르다
- 에이전트별 `allow` 에 넣은 대안 모델이 **전역 `allow` 에 없음** → 에이전트 규칙은 전역을 완화하지 못하므로 쓰는 순간 거부된다
- `timeoutMs` 를 `settings.json` 에 넣음 → 오류 없이 무시된다
- `apiKey` 평문 하드코딩

#### Windows PowerShell 에서 돌릴 때

`smoke.sh` 는 bash 스크립트라 PowerShell 에서 바로 실행되지 않는다. 셋 중 하나를 쓴다.

```powershell
# 1) Git for Windows 가 있으면 (가장 흔함)
& "C:\Program Files\Git\bin\bash.exe" ./smoke.sh -l 0

# 2) WSL 이 있으면
wsl ./smoke.sh -l 0

# 3) Git Bash 를 열어서 평소처럼
./smoke.sh -l 0
```

**한글이 깨져 보이면 콘솔 코드페이지 문제다.** 스크립트 출력은 UTF-8 인데 Windows 콘솔은
기본이 949(한국어)라 그대로 읽으면 깨진다. 스크립트가 Git Bash/MSYS 에서 실행되면 알아서
`chcp 65001` 을 걸지만, PowerShell 에서 직접 돌리거나 그래도 깨지면 아래를 먼저 실행한다.

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

Git Bash 를 쓰는데도 깨지면 터미널 글꼴이 한글을 못 그리는 경우다. `D2Coding`, `Malgun Gothic`,
`Cascadia Mono` 중 하나로 바꾼다 (Git Bash: 창 우클릭 → Options → Text → Font).

`python3` 가 아니라 `python` 만 있는 환경이 흔해서, 스크립트가 `python3` → `python` → `py`
순으로 찾아 **Python 3 인 것만** 쓰도록 해 뒀다. 셋 다 없으면 거기서 멈춘다.

전체 벽시계 상한도 Windows 에서는 `timeout` 이 없으므로 PowerShell 로 건다.

```powershell
$prompt = Get-Content .\oneshot-prompt.txt -Raw
$p = Start-Process pi -ArgumentList '-p', $prompt -PassThru -NoNewWindow
Wait-Process -Id $p.Id -Timeout 28800 -ErrorAction SilentlyContinue   # 8시간
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
```

그리고 Windows 에서 무인 실행을 할 거면 **`smoke.sh -l 3` 을 반드시 한 번 돌린다.**
`pi-subagents` 의 백그라운드 자식 실행은 공식 검증 대상이 Linux x64 이고 Windows 는
실험 단계로 표기돼 있다. npm 으로 설치한 pi 는 Node 러너를 쓰므로 일반 경로지만,
검증 매트릭스에 없다는 사실은 그대로다. 포그라운드(`asyncByDefault: false`)로 쓰는 편이 안전하다.

### 3. 런북

`pi-subagents` 는 부모 pi 세션이 오케스트레이터다. 자연어로 지시하면 부모가 `subagent` 도구를 호출한다.

**기본 루프** — 패키지가 권장하는 순서이자 이 설계와 같은 모양이다.

```text
clarify → scout → worker → fresh reviewer → worker
```

```text
scout 으로 이 저장소에서 <대상> 관련 파일을 찾아줘.
worker 로 <작업>을 구현해줘. 끝나면 reviewer 를 띄워서 검증 명령을 직접 다시 돌리게 하고, 지적이 있으면 worker 에게 되돌려줘.
```

**병렬 리뷰**

```text
/parallel-review
```

정확성·테스트·불필요한 복잡도를 각각 보는 리뷰어를 동시에 띄우고 결과를 합친다.

**결정적 파이프라인이 필요할 때** — 부모 모델의 판단에 맡기지 않고 순서를 고정하려면 `workflowScript` 를 쓴다. `runs.all([{ key, agent, task }, ...])` 가 팬아웃이고, 결과는 순서대로 배열로 돌아온다.

**워크트리 격리** — 파일 겹침을 피할 수 없을 때만. 워커마다 별도 체크아웃을 준다.

### 3-1. 사내 규칙은 작업 저장소의 `AGENTS.md` 에 둔다

빌트인 역할 4종은 모두 `systemPromptMode: replace` 다 — 자기 프롬프트로 교체하므로 별도 역할 파일을 끼워 넣을 자리가 없다. 대신 넷 다 `inheritProjectContext: true` 라서 **작업 대상 저장소의 `AGENTS.md` 는 모든 자식에게 항상 전달된다.**

"배정된 파일 밖 수정 금지", "git add -A 쓰지 말 것", "검증 명령을 지어내지 말 것" 같은 규칙의 제자리가 거기다. 원샷 프롬프트에만 적으면 **대화형으로 쓸 때는 자식에게 전달되지 않는다.** 넣을 내용은 [agents-md.snippet.md](./agents-md.snippet.md) 에 있다.

자식은 스킬을 상속하지 않으므로(`inheritSkills: false`) 스킬에 의존하는 규칙은 쓰지 않는다.

### 4. 지키는 것

- 리뷰어가 지적하면 **워커에게 되돌린다.** 부모가 직접 고치기 시작하면 구조가 무너진다.
- 커밋은 사람이 한다.
- 워커를 늘리기 전에, 작업이 정말 파일 단위로 나뉘는지 먼저 확인한다.

## 운영 가드레일

| 항목 | 규칙 |
|------|------|
| 외부 전송 | `researcher` / `evidence-auditor` 를 끈다. 단 `bash` 가 있으면 우회 가능하다 — 실제 차단은 네트워크 정책으로 한다. |
| 자격증명 | API 키는 환경변수로만 (`$HCP_API_KEY`, `$ITC_VLM_API_KEY`). `models.json` 과 저장소에 키를 넣지 않는다. |
| 샌드박스 | pi에는 샌드박스도 권한 팝업도 없다. 자식은 실행 사용자 권한을 그대로 가진다. **하드 가드는 프롬프트 문구일 뿐 강제가 아니다.** 실제 경계가 필요하면 컨테이너/VM. |
| 시크릿 | API 키를 환경변수에 두면 `bash` 로 읽힌다. 에이전트에게서 숨겨야 한다면 격리 경계 밖 프록시가 키를 들어야 한다. |
| 비용 | 큰 모델은 `oracle` · `reviewer` 에만. `defaultModel` 은 Medium으로 둔다. |
| 폴백 금지 | 자식 실행이 깨졌을 때 외부 CLI나 다른 실행 경로로 우회하지 않는다. 실패를 그대로 보고한다. |

## 속도 제한 (RPM / TPM)

사내 게이트웨이에 호출 한도가 걸려 있으면 **동시 실행이 곧바로 한도를 넘는다.** 예로
`my-local-provider` 가 RPM 50 / TPM 500,000 이면 기본값으로는 감당이 안 된다.

| 설정 | 파일 | 기본값 | 이 구성 | 왜 |
|------|------|--------|---------|-----|
| `globalConcurrencyLimit` | config.json | 20 | **2** | 한 실행 안에서 동시에 도는 자식 수. 20 이면 RPM 50 을 즉시 넘는다 |
| `parallel.concurrency` | config.json | 4 | **2** | 병렬 그룹의 동시 실행 수 |
| `maxActiveAsyncRunsPerSession` | config.json | 4 | **1** | 동시에 도는 최상위 비동기 실행 수 |
| `maxSubagentSpawnsPerRun` | config.json | 64 | **12** | 한 실행에서 띄울 수 있는 자식 총량. 폭주 방지 |
| `retry.maxRetries` | settings.json | 3 | **6** | 429 는 정상적으로 발생한다. 재시도로 넘긴다 |
| `retry.baseDelayMs` | settings.json | 2000 | **8000** | 2초 후 재시도는 RPM 한도에서 다시 429 를 부른다 |
| `retry.provider.maxRetries` | settings.json | — | **4** | SDK 계층 재시도. 에이전트 재시도와 별개로 먼저 동작한다 |

**429 는 재시도하지 않으면 그대로 실패다.** `pi-subagents` 문서가 명시한다 — 속도 제한을
포함한 provider 오류는 그 시도에서 그대로 반환되고, 다른 모델로 넘어가지 않는다. 자식이
죽으면 그 작업 단위가 통째로 날아간다. 무인 야간 실행에서는 치명적이다.

**TPM 이 더 빡빡할 수 있다.** 컨텍스트 100만짜리 모델에 긴 세션을 물리면 요청 하나가
수십만 토큰이 된다. TPM 500,000 이면 그런 요청 몇 개로 분당 한도가 찬다. 대응은 설정이
아니라 구조다 — 자식마다 fresh context 를 주고, 부모가 자식 전사를 흡수하지 않게 하고,
보고를 파일로 받는 것이 전부 여기에 기여한다.

### 다른 harness 와 같은 provider 를 나눠 쓴다면

**RPM/TPM 은 provider 계정 단위다.** 여기 설정은 pi 세션 하나의 자식 수만 제한할 뿐,
Claude Code·Codex·다른 pi 세션이 같은 엔드포인트를 치면 **같은 버킷을 나눠 쓴다.**
pi 에는 프로세스 간 공유 토큰 버킷이 없다 — 서로의 존재를 모른다.

한 세션이 활발히 돌 때의 대략적인 요청 예산이다.

| 소비자 | 요청/분 |
|--------|---------|
| 부모 세션 | 5–10 |
| 자식 2개 | 10–20 |
| watchdog (경계 검토) | 3–8 |
| **합계** | **약 20–40** |

RPM 50 이면 **혼자서도 거의 채운다.** 그래서 두 프로필로 나눈다.

| | 단독 사용 | **다른 harness 와 공유** |
|---|---|---|
| `globalConcurrencyLimit` | 2 | **1** |
| `parallel.concurrency` | 2 | **1** |
| `watchdog.enabled` | true | true |
| `watchdog.children.enabled` | false | false |
| `watchdog.cadence` | 설정 안 함 | 설정 안 함 |
| `retry.maxRetries` | 6 | **8** |
| `retry.baseDelayMs` | 8000 | **15000** |

기본 스니펫은 **단독 사용** 기준이다. 공유한다면 위 값으로 바꾼다.

**watchdog 이 숨은 소비자다.** 턴 경계마다 별도 모델 호출이 들어간다. `cadence.everyNTools`
를 켜면 도구 N번마다 또 호출해서 요청이 배로 는다. 그래서 cadence 는 끄고 경계 검토만 남겼고,
자식 watchdog 은 아예 껐다 — 자식이 동시에 돌면 이게 RPM 을 가장 크게 먹는다.
자식의 품질 통제는 watchdog 이 아니라 `reviewer` 게이트가 맡는다.

### 결국 재시도가 안전망이다

동시성을 아무리 낮춰도 **다른 harness 를 제어할 수는 없다.** 그래서 429 를 막는 게 아니라
**429 를 견디는 쪽**이 현실적인 설계다. `retry` 가 설정돼 있으면 429 는 실패가 아니라
지연이 된다 — 느려지되 작업은 끝난다.

한도를 모르면 `globalConcurrencyLimit: 1` 로 시작해서 올린다. 병렬을 포기하는 것이
429 로 밤 작업을 날리는 것보다 싸다. 무인 야간 실행이라면 애초에 1 로 두는 편이 낫다 —
밤에는 급할 일이 없다.

## 알아둘 동작

- 모델 해석 우선순위: 런당 override → provider별 역할 override → `agentOverrides.<역할>.model` → 에이전트 frontmatter → `subagents.defaultModel` → 부모 세션 모델.
- `agentOverrides` 로 덮어쓸 수 있는 필드에 `tools`, `thinking`, `model`, `description`, `disabled`, `systemPrompt` 가 포함된다. 역할 파일을 통째로 복사할 필요가 없다.
- 빌트인을 통째로 고쳐야 하면 `subagent({ action: "eject", agent: "reviewer" })` 로 사용자 디렉터리에 복사본을 꺼낸다.
- 사용자 역할은 `~/.pi/agent/agents/**/*.md`, 프로젝트 역할은 `.pi/agents/**/*.md` 에 둔다.
- `pi --list-models <이름>` 은 못 찾으면 `No models matching "이름"` 이라고 답한다. **안내문에 검색어가 그대로 들어 있어서** 단순 grep으로 존재 확인을 하면 항상 통과한다. 표의 provider·model 열로 대조해야 한다.

검증 상태:

- `models.json` 스키마(최상위 `providers` 래퍼 필수)와 `--list-models` 동작은 pi 0.86.1 로 직접 확인했다.
- thinking 레벨 클램프 동작은 pi 소스(`models.js` 의 `getSupportedThinkingLevels` / `clampThinkingLevel`)로 확인했다.
- `smoke.sh` 는 `PI_CODING_AGENT_DIR` 로 실제 2-provider·5-모델 구성을 흉내 내 **통과(ok 37)** 와 **변형 주입 시 검출(8/8)** 양방향을 확인했다.
- `settings.snippet.json` / `subagent-config.snippet.json` 의 키는 설치된 `pi-subagents` 0.70.0 문서 기준이며, **사내 엔드포인트에 물려 실행 검증한 것은 아니다.** 첫 배선 때 `smoke.sh -l 3` 으로 확인한다.
- 설계는 Codex(gpt-6-astra)로 3라운드 검증을 거쳤다. 지적 P1 14건을 반영했다.

## 두 가지 사용 모드

설정은 하나를 공유하고, **프롬프트 규율만 다르다.**

### 업무 중 (대화형, 사람이 옆에 있음)

```bash
cd /path/to/repo
pi                     # 부모 세션 = 오케스트레이터 (HCP-Big)
```

부모가 직접 일할지 자식을 띄울지 스스로 판단한다. **작은 일에는 오케스트레이션을 쓰지 않는 것이 맞다** — 자식 하나 띄우는 비용이 그냥 고치는 비용보다 크다.

| 일의 크기 | 이렇게 한다 |
|---|---|
| 파일 1~2개 수정, 원인이 뻔함 | 부모에게 그냥 시킨다. 서브에이전트 없음 |
| 어디를 고쳐야 할지 모르겠음 | `scout` 으로 먼저 찾는다 → 부모가 고친다 |
| 파일 여러 개, 범위가 분명함 | `worker` 1명 → `reviewer` |
| 모듈 여러 개, 쪼개야 함 | `oracle` 로 분해 → `worker` 병렬 → `reviewer` |
| 이미 짠 코드 검토만 | `/parallel-review` |

**전형적인 하루 루프**

```text
scout 으로 <증상/대상> 관련 파일과 호출 지점을 찾아줘. 고치지는 말고.
```
```text
worker 로 <작업>을 구현해줘. 범위는 <경로>로 한정하고, 끝나면 <검증 명령>을 돌려줘.
```
```text
reviewer 를 fresh context 로 띄워서 방금 diff 를 검증해줘.
보고를 믿지 말고 git diff 를 직접 읽고 검증 명령도 직접 다시 돌리게 해줘.
```

지적이 나오면 **같은 worker 에게 되돌린다.** 부모가 직접 고치기 시작하면 구조가 무너진다.

**슬래시 단축**

| 명령 | 쓸 때 |
|---|---|
| `/run <역할> "<작업>"` | 역할 하나를 직접 지목. `--bg` 로 백그라운드, `--fork` 로 현재 맥락 물려주기 |
| `/parallel-review` | 정확성·테스트·복잡도를 각각 보는 리뷰어 동시 투입 |
| `/review-loop` | worker → reviewer → fix 사이클을 상한까지 자동 |
| `/parallel-cleanup` | 구현 끝난 뒤 검토 전용 패스 |
| `/gather-context-and-clarify` | 먼저 파악하고, 물어볼 것을 나에게 질문 |

`/parallel-research` 는 `researcher`(웹 검색)를 쓰므로 사내에서는 못 쓴다. 꺼놨다.

**사람이 있으므로 막히면 물어보는 게 맞다.** 자식이 `contact_supervisor` 로 올린 결정을 부모가 너에게 그대로 전달한다. 결정 정책 파일도 결정 로그도 필요 없다 — 그 자리에서 답하면 된다.

`asyncByDefault: false` 로 둔 이유가 이것이다. 포그라운드로 흘러야 진행을 보면서 끊을 수 있다. 오래 걸리는 것만 `--bg` 로 떼어 놓고 `subagent({ action: "status" })` 로 확인한다.

### 퇴근 원샷 (무인)

달라지는 것은 셋뿐이다.

1. **결정을 미리 위임한다** — `.orch/decisions.md` 에 A(미리 승인) / B(기본값 규칙) / C(자가 판단 금지) 3단 정책을 써둔다. 자식이 막히면 부모가 이 파일을 근거로 답한다. 스스로 정한 것은 전부 `.orch/결정로그.md` 에 되돌리는 법까지 남긴다.

   판정은 **하드가드 → D → C → A → B** 순서로만 한다. D(사람이 미리 구체적으로 답한 것)가 C보다 앞인 이유는, C는 "사람에게 물어라"는 뜻이고 D는 이미 물어서 받은 답이기 때문이다. A를 먼저 보면 "기존 패턴 따르기(A)"인 동시에 "인증 로직 변경(C)"인 변경이 승인되어 버린다.

   세 모드가 실제로 다른 지점은 **C 목록의 적용 범위**다. `보수`는 전체를 멈추고, `균형`은 그 단위만 멈추고, `자율`은 **C를 하드 가드까지만 좁혀** 나머지는 B 규칙으로 스스로 정하고 계속 간다. "전부 자가 판단"은 "전부 허용"이 아니라 "멈추지 않음"이다.

   정책은 **프롬프트 본문에 그대로 붙여넣는다.** 파일만 트리 밖에 두고 `chmod 444` 를 걸어도 같은 사용자 권한의 `bash` 가 되돌릴 수 있어 실수 방지 수준에 그친다. 컨텍스트에 들어온 원문이 기준이 되어야 파일 교체가 의미를 잃는다.
2. **워크트리를 분리하고 단위별로 커밋한다** — 단위 하나를 통째로 revert 할 수 있어야 반쯤 고친 호출부가 남지 않는다. 단 `git add -A` 가 아니라 **그 단위의 경로만** 올린다. 전체를 올리면 병렬 워커의 미완성 변경이 섞여 revert 가 남의 작업까지 지운다. 워커를 2명 이상 동시에 돌릴 때는 워커마다 워크트리를 따로 준다. push 만 금지한다. 단, 워크트리는 **되돌리기 수단이지 격리가 아니다** — 홈·SSH 키·네트워크에는 그대로 닿는다.
3. **상한을 건다** — `timeoutMs` 는 **개별 subagent 실행** 제한이지 밤 작업 전체의 상한이 아니다. 부모가 단위를 계속 만들면 전체 시간은 계속 늘어난다. 전체 벽시계 상한은 `timeout 8h pi` 처럼 **OS 레벨로** 걸어야 한다.

프롬프트 템플릿은 [oneshot.md](./oneshot.md), 정책 템플릿은 [decisions.example.md](./decisions.example.md).

그리고 무인이든 아니든 `subagents.watchdog` 은 켜둔다. 턴 경계에서 다른 모델이 방금 한 일을 되짚어 범위 이탈·루프 위험·위험한 변경을 찾는다.

**단 이것은 사후 검토이지 실행 전 차단이 아니다.** 전송이나 삭제가 일어난 *뒤에* 지적한다. 하드 가드를 대신하지 못한다. 그리고 `watchdog.main.model` 을 생략하면 세션 모델을 상속하므로, 부모와 다른 모델이 되도록 명시해야 한다 — 스니펫에 넣어 뒀다. 중간 점검이 필요하면 `cadence.everyNTools` 를 함께 켠다.

### 왜 원샷에 `pi-subagents` 인가

**분해를 미리 알 수 없기 때문이다.** 원샷 프롬프트는 "이거 해둬"에 가깝고, 워커를 몇 명 띄울지는 저장소를 읽어 본 뒤에 정해진다. 부모 세션이 그 판단을 하고 자식을 띄우는 것이 Supervisor 패턴의 핵심이며, 고정 파이프라인으로는 안 된다.

## 참고 자료

- [Pi 공식 문서](https://pi.dev/docs/latest) · [Providers](https://pi.dev/docs/latest/providers) · [Security](https://pi.dev/docs/latest/security)
- `pi-subagents` 문서 — 설치 경로의 `docs/` (`models.md`, `agents.md`, `workflows.md`, `configuration.md`)
- [6 Multi-Agent Orchestration Patterns for Production (2026)](https://beam.ai/agentic-insights/multi-agent-orchestration-patterns-production)

## 관련 문서

- [사내 PC 세팅 프롬프트 (PROMPT-1~7)](./office-setup.md) — 주 경로
- [세팅 점검 스크립트](./smoke.sh) · [설정 병합 스크립트](./merge-settings.py)
- [역할별 티어 배선](./settings.snippet.json) · [런타임 상한](./subagent-config.snippet.json)
- [퇴근 원샷 프롬프트](./oneshot.md) · [결정 정책 템플릿](./decisions.example.md)
- [작업 저장소 AGENTS.md 규칙](./agents-md.snippet.md)
