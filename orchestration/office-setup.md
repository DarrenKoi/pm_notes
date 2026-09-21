---
tags: [orchestration, setup, office, pi-subagents]
level: intermediate
last_updated: 2026-09-22
---

# 사내 PC 세팅 — 프롬프트 모음

> 각 프롬프트에 `PROMPT-N` 번호가 붙어 있다. pi 에게 **"office-setup.md 의 PROMPT-N 을 실행해라"**
> 라고 지시하면 된다. 스크립트를 돌리지 않고 전부 pi 로 진행하는 경로다.

## 이 문서를 읽는 에이전트에게

- **지시받은 번호의 프롬프트 하나만 실행한다.** 다른 번호의 프롬프트를 이어서 실행하지 마라.
- 프롬프트는 ` ```text ` 블록 안의 내용이다. 그 밖의 설명 문장은 사람이 읽는 것이니 실행하지 마라.
- 각 프롬프트 아래 **보고할 것** 에 적힌 항목을 반드시 출력한다. 요약하지 말고 실제 값을 낸다.
- 확인하지 않은 것을 확인했다고 쓰지 마라. 모르면 `확인불가` 라고 쓴다.

## 순서

| 번호 | 하는 일 | 선행 |
|------|---------|------|
| `PROMPT-1` | 현재 설정 상태 확인 (읽기만) | — |
| `PROMPT-2` | `settings.json` 병합 | 1 |
| `PROMPT-3` | `extensions/subagent/config.json` 생성 | 2 |
| `PROMPT-4` | **설정 검증 15항목** | 3 |
| `PROMPT-5` | 역할 배선 확인 (`subagent list`) | 4 |
| `PROMPT-6` | 서브에이전트 왕복 | 5 |
| `PROMPT-7` | 실제 작업 한 바퀴 | 6 |

앞 번호가 끝나야 뒤 번호가 의미 있다. 건너뛰지 않는다.

---

## PROMPT-1 — 현재 상태 확인

아직 아무것도 고치지 않는다. 지금 무엇이 있는지만 본다.

```text
설정 파일을 읽기만 해라. 아무것도 수정하지 마라.

1. ~/.pi/agent/settings.json 을 읽고 다음을 알려줘라.
   - defaultModel 값
   - subagents 블록이 있는가, 있으면 그 안의 키 목록
   - 그 외 최상위 키 이름들 (값은 필요 없다)
2. ~/.pi/agent/extensions/subagent/config.json 이 존재하는가.
3. `pi --list-models` 를 실행하고 provider 열과 model 열만 "provider/model" 형식으로 나열해라.

파일에 실제로 있는 값만 써라. 없으면 "없음" 이라고 해라.
```

**보고할 것** — 위 세 가지 전부.

`PROMPT-2` 의 JSON 은 모델 이름이 아래와 같다고 가정한다. 3번 목록과 다르면 **먼저 알려달라.**

```text
my-local-provider/HCP-Big-Latest
my-local-provider/HCP-Medium-Latest
my-local-provider/HCP-Small-Latest
my-local-provider/HCP-Vision-Latest
itc-vlm/qwen3.8-27b
```

---

## PROMPT-2 — settings.json 병합

`agentOverrides`·`modelScope`·`watchdog` 이 들어간다.

```text
~/.pi/agent/settings.json 에 아래 JSON 을 병합해라.

규칙:
- 먼저 settings.json 을 settings.json.bak 으로 복사해라.
- 아래에 있는 키만 손대라. 기존의 다른 키(theme, packages 등)는 한 글자도 바꾸지 마라.
- subagents 를 통째로 교체하지 마라. 그 안의 키만 합쳐라 (깊은 병합).
- agentOverrides 안에 이미 있는 역할은 통째로 교체해도 된다. 단 교체 전에 원래 내용을 출력해라.
- 값을 새로 지어내지 마라. 아래 있는 값을 그대로 써라.
- tools 는 반드시 JSON 배열로 써라. 쉼표로 이은 문자열은 pi 가 거부한다.
- 아래 JSON 에 subagents.watchdog.cadence 와 subagents.watchdog.children.cadence 가
  없는 것은 의도적이다. 기존 설정에 그 키가 있으면 "삭제" 해라. 생략은 삭제가 아니다.
  cadence 는 도구 N번마다 추가 모델 호출을 만들어 RPM 을 크게 먹는다.
- 기존 최상위 defaultModel 에 "provider/id" 형태가 들어 있으면 고쳐라. pi 는 최상위
  defaultProvider 와 defaultModel(id 만) 을 따로 받는다. 접두사가 붙으면 조회가 실패하고
  아무 모델로 폴백한다.
- 쓰기 전에 "추가한 키" 와 "덮어쓴 키(이전 -> 이후)" 를 전부 나열해서 나에게 보여줘라.
- 쓴 다음 파일을 다시 읽어서 JSON 으로 파싱되는지 확인하고, subagents.agentOverrides 의
  키 목록과 subagents.modelScope.enforce 값을 출력해라.

```json
{
  "defaultProvider": "my-local-provider",
  "defaultModel": "HCP-Big-Latest",
  "defaultThinkingLevel": "high",
  "httpIdleTimeoutMs": 900000,
  "retry": {
    "enabled": true,
    "maxRetries": 6,
    "baseDelayMs": 8000,
    "maxAgentDelayMs": 120000,
    "provider": {
      "maxRetries": 4,
      "maxRetryDelayMs": 60000
    }
  },
  "subagents": {
    "defaultModel": "my-local-provider/HCP-Medium-Latest",
    "defaultProvider": "my-local-provider",
    "agentOverrides": {
      "oracle": {
        "model": "my-local-provider/HCP-Big-Latest",
        "thinking": "high",
        "description": "Planning tier (Big). Decomposition and judgment only. Hands implementation to worker."
      },
      "reviewer": {
        "model": "my-local-provider/HCP-Big-Latest",
        "thinking": "high",
        "tools": [
          "read",
          "grep",
          "find",
          "ls",
          "bash",
          "write",
          "watchdog_diff",
          "contact_supervisor"
        ],
        "description": "Verification gate (Big). Runs in a separate session from worker. Reads the diff directly and re-runs the verification commands instead of trusting the worker's report."
      },
      "worker": {
        "model": "my-local-provider/HCP-Medium-Latest",
        "thinking": "high",
        "description": "Implementation tier (Medium). Edits code within an assigned, non-overlapping file scope."
      },
      "scout": {
        "model": "itc-vlm/qwen3.8-27b",
        "thinking": "low",
        "description": "Recon tier. Exhaustive search, filtering and enumeration. Read-only reconnaissance; does not edit source."
      },
      "researcher": {
        "disabled": true
      },
      "evidence-auditor": {
        "disabled": true
      }
    },
    "modelScope": {
      "enforce": true,
      "strict": true,
      "allow": [
        "my-local-provider/*",
        "itc-vlm/*"
      ],
      "agents": {
        "oracle": {
          "allow": [
            "my-local-provider/HCP-Big-Latest"
          ]
        },
        "reviewer": {
          "allow": [
            "my-local-provider/HCP-Big-Latest",
            "itc-vlm/qwen3.8-27b"
          ]
        },
        "worker": {
          "allow": [
            "my-local-provider/HCP-Medium-Latest"
          ]
        },
        "scout": {
          "allow": [
            "itc-vlm/qwen3.8-27b",
            "my-local-provider/HCP-Small-Latest",
            "my-local-provider/HCP-Vision-Latest"
          ]
        }
      }
    },
    "watchdog": {
      "enabled": true,
      "main": {
        "model": "my-local-provider/HCP-Medium-Latest"
      },
      "children": {
        "enabled": false,
        "model": "my-local-provider/HCP-Medium-Latest"
      }
    }
  }
}
```
```

**성공 판정** — `agentOverrides` 키가 **6개**로 나와야 한다.

```text
oracle, reviewer, worker, scout, researcher, evidence-auditor
```

하나만 남아 있으면 깊은 병합이 아니라 **교체**가 일어난 것이다. `settings.json.bak` 으로
되돌리고 다시 시도한다. `modelScope.enforce` 는 `true` 여야 한다.

**보고할 것** — 추가/덮어쓴 키 목록과 마지막 확인 출력.

---

## PROMPT-3 — subagent config.json 생성

`timeoutMs` 는 `settings.json` 이 아니라 **이 파일**에 있어야 한다. 자리를 틀리면 오류 없이 무시된다.

```text
~/.pi/agent/extensions/subagent/config.json 을 만들어라.
디렉터리가 없으면 만들고, 파일이 이미 있으면 .bak 으로 백업한 뒤 아래 키만 병합해라.

```json
{
  "timeoutMs": 10800000,
  "toolTimeoutMs": 900000,
  "asyncByDefault": false,
  "globalConcurrencyLimit": 2,
  "maxActiveAsyncRunsPerSession": 1,
  "maxSubagentSpawnsPerRun": 12,
  "parallel": {
    "maxTasks": 4,
    "concurrency": 2
  }
}
```

쓴 다음 파일 내용을 그대로 출력해라.
```

**보고할 것** — 최종 파일 내용.

---

## PROMPT-4 — 설정 검증 (15항목)

**여기가 제일 중요하다.** 지금까지 실제로 발목을 잡은 함정이 전부 들어 있다.
모두 **오류 없이 조용히 무시되거나 엉뚱하게 동작하는** 종류라 눈으로는 안 보인다.

```text
설정을 검증해라. 파일을 수정하지 마라. 아래 항목을 하나씩 확인하고
각각 OK / FAIL / 확인불가 로 판정한 뒤, FAIL 은 무엇을 어떻게 고쳐야 하는지 적어라.
추측하지 말고 파일에서 읽은 실제 값을 근거로 제시해라.

읽을 파일:
  A = ~/.pi/agent/models.json
  B = ~/.pi/agent/settings.json
  C = ~/.pi/agent/extensions/subagent/config.json

0. B 의 최상위 defaultProvider 와 defaultModel 이 둘 다 있고, defaultModel 에
   "/" 가 없는가. pi 는 getModel(provider, id) 로 조회하므로 defaultModel 에
   "provider/id" 를 넣으면 실패하고 아무 모델로 폴백한다. FAIL 이면 둘로 나눈 값을 제시해라.

1. A 의 최상위에 "providers" 키가 있는가.
   없으면 파일 전체가 무시된다. {"providers": {"<provider>": {...}}} 형태여야 한다.

2. B 의 subagents.agentOverrides 키가 정확히 6개인가.
   oracle, reviewer, worker, scout, researcher, evidence-auditor

3. agentOverrides 의 각 역할에서 tools 필드가 있으면 그것이 "배열"인가.
   "read, grep, bash" 같은 쉼표 문자열이면 FAIL 이다. 빌트인 frontmatter 는 그 표기를
   받지만 settings.json 은 배열만 받는다. FAIL 이면 배열로 바꾼 값을 제시해라.

4. agentOverrides 의 각 model 값이 A 에 실제로 정의된 provider/id 와 글자 그대로 일치하는가.
   대소문자와 provider 접두사까지 본다.

5. agentOverrides 의 각 thinking 값이 그 모델에서 지원되는가.
   A 에서 그 모델의 정의를 보고 아래 규칙으로 판정해라. 추측하지 말고 규칙만 적용해라.

   (a) reasoning 필드가 없거나 false 다  -> 지원 레벨은 off 하나뿐이다.
       이 경우 thinking 에 off 가 아닌 값을 준 것은 FAIL 이다. 조용히 off 로 떨어져
       thinking 이 아예 꺼진 채로 돈다.
   (b) reasoning 이 true 인데 thinkingLevelMap 필드가 아예 없다 -> 정상이다.
       off, minimal, low, medium, high 가 지원되고 xhigh 와 max 만 지원되지 않는다.
       thinking 이 이 다섯 중 하나면 OK 다. 이 경우 확인불가로 쓰지 마라.
   (c) reasoning 이 true 이고 thinkingLevelMap 이 있다 -> 그 레벨의 값이 null 이면 미지원,
       xhigh/max 는 맵에 키가 있어야만 지원이다. 미지원 레벨을 주면 pi 가 오류 없이
       다른 레벨로 바꿔버려 의도한 비용·품질이 안 나온다.

   FAIL 이면 그 모델이 실제로 지원하는 레벨 목록을 함께 알려줘라.

6. researcher 와 evidence-auditor 가 disabled: true 인가. 값이 문자열 "true" 가 아니라
   불리언 true 여야 한다.

7. subagents.modelScope 가 있는가. enforce 와 strict 가 true 인가.

8. modelScope.agents.<역할>.allow 의 각 항목이 modelScope.allow(전역) 에도 매치되는가.
   에이전트 규칙은 전역 규칙을 완화하지 못한다. 전역에 없는 항목은 쓰는 순간 거부되는 죽은 항목이다.

9. modelScope.agents.<역할>.allow 가 그 역할의 배정 모델을 허용하는가.
   허용하지 않으면 그 역할은 항상 실패한다.

10. B 의 subagents 안에 timeoutMs 가 있는가. 있으면 FAIL 이다.
    이 키는 C 에 있어야 하며 B 에 두면 오류 없이 무시된다.

11. C 에 timeoutMs 와 toolTimeoutMs 가 있는가. 값도 함께 보여줘라.

12. subagents.watchdog.enabled 가 true 이고 watchdog.main.model 이 명시돼 있는가.
    생략하면 부모 세션 모델을 상속해서 독립적인 검토가 되지 않는다.

13. B 에 retry 가 있는가. retry.enabled 가 false 면 FAIL 이다.
    사내 게이트웨이는 RPM 한도가 낮아 429 가 정상적으로 발생하는데, 재시도가 꺼져 있으면
    429 한 번에 워커가 죽는다. maxRetries 와 baseDelayMs 값도 함께 보여줘라.

14. C 의 globalConcurrencyLimit 값이 얼마인가. 없으면 기본 20 이다.
    RPM 한도가 50 이면 20 은 즉시 넘는다. 4 이하를 권장한다.
    parallel.concurrency 값도 함께 보여줘라(없으면 기본 4).
    B 의 subagents.watchdog 또는 watchdog.children 에 cadence 키가 남아 있으면 FAIL 이다.
    스니펫에서 생략한 것은 기존 설정에서 자동으로 지워지지 않는다. cadence 는 도구 N번마다
    추가 모델 호출을 만들어 RPM 을 크게 먹는다. RPM 한도가 낮으면 경계 검토만 남기는 편이 낫다.

마지막에 OK 개수 / FAIL 개수 / 확인불가 개수를 한 줄로 요약해라.
```

**성공 판정** — FAIL 0건. `확인불가` 가 있으면 어느 항목인지 알려달라.

**보고할 것** — 15개 항목의 판정과 마지막 요약 줄.

---

## PROMPT-5 — 역할 배선 확인

```text
subagent 도구로 사용 가능한 에이전트 목록을 조회해라 (action: "list").
각 에이전트의 이름과, 해석된 모델이 보이면 모델도 함께 보여줘라.

그다음 다음을 확인해서 알려줘라.
- oracle, worker, reviewer, scout 네 개가 모두 보이는가
- researcher 와 evidence-auditor 가 목록에서 빠졌는가 (빠져야 정상이다)
```

**성공 판정** — 네 역할이 보이고 `researcher`·`evidence-auditor` 는 안 보인다.

**보고할 것** — 목록 그대로.

---

## PROMPT-6 — 서브에이전트 왕복

부모가 자식을 실제로 띄우는지 본다. 여기까지 되면 배선은 끝이다.

```text
scout 서브에이전트를 하나 띄워서 현재 디렉터리의 파일 목록을 조사하게 해라.
네가 직접 하지 말고 반드시 subagent 도구를 써라.

자식이 끝나면 다음을 알려줘라.
- 자식이 실제로 떴는가
- 자식이 어떤 모델로 돌았는가 (알 수 있으면)
- 자식이 돌려준 결과
```

**성공 판정** — 자식이 뜨고 결과가 돌아온다.

**실패하면** — 부모가 직접 처리해버린 것일 수 있다(설정 문제 아님).
"직접 하지 말라" 를 한 번 더 강조해 재시도하고, 그래도 안 되면 오류를 그대로 보고한다.

---

## PROMPT-7 — 실제 작업 한 바퀴

버려도 되는 워크트리에서 작은 작업을 한 번 돌린다. 먼저 셸에서:

```powershell
cd <작업할 저장소>
git worktree add ../wt-test -b test/orch
cd ../wt-test
pi
```

```text
scout 으로 <작은 모듈 이름> 관련 파일과 호출 지점을 찾아줘. 고치지는 말고.
```

**여기서 볼 것** — `scout` 이 **호출부를 빠뜨리지 않았는지** 직접 확인한다.
이게 scout 모델을 고른 이유이고, 사내 코드베이스에서만 확인할 수 있다.
부족하면 알려달라 — 더 큰 모델로 올리겠다.

이어서 worker → reviewer 까지 한 바퀴 돌려보고, 끝나면 워크트리째 버린다.

```powershell
cd -
git worktree remove ../wt-test --force
```

---

## 막히면

오류 메시지를 **요약하지 말고 그대로** 가져온다.
pi 가 "했다" 고만 하고 근거를 안 보여주면 프롬프트 끝에 이 줄을 붙인다.

```text
실제 파일 내용과 실행한 명령의 출력을 그대로 보여줘라. 요약하지 마라.
```

## 스크립트로 검증하고 싶다면

bash 와 python 을 돌릴 수 있는 환경이면 `PROMPT-4` 대신 `smoke.sh` 를 쓸 수 있다.
같은 항목을 기계적으로 검사하고 판정이 흔들리지 않는다.

```bash
./smoke.sh -l 0      # 설정 검사, LLM 호출 0회
./smoke.sh -l 3      # 연결·도구·서브에이전트 왕복까지
```

설정 병합도 `merge-settings.py` 가 더 안전하다. JSON 병합은 결정적 변환이라
모델이 판단할 여지가 없는 편이 낫다.

## 관련 문서

- [오케스트레이션 개요](./README.md)
- [퇴근 원샷 프롬프트](./oneshot.md) · [결정 정책 템플릿](./decisions.example.md)
- [작업 저장소 AGENTS.md 규칙](./agents-md.snippet.md)
