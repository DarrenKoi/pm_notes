---
tags: [orchestration, setup, office, pi-subagents]
level: intermediate
last_updated: 2026-09-21
---

# 사내 PC 세팅 — 단계별 프롬프트

> 스크립트를 돌리는 대신 **pi 에게 시켜서** 세팅한다. 각 단계의 프롬프트를 그대로
> 붙여넣고, "보고할 것" 에 적힌 출력을 그대로 가져오면 다음 단계로 넘어간다.
> 모델은 이미 pi 에 연결돼 있다고 가정한다.

각 단계는 **앞 단계가 끝나야** 의미가 있다. 순서대로 한다.

---

## 0단계 — pi 시작

아무 저장소에서나 연다. 아직 코드를 고치지 않으므로 어디든 상관없다.

```powershell
pi
```

---

## 1단계 — 현재 상태 확인

아직 아무것도 고치지 않는다. 지금 무엇이 있는지만 본다.

**붙여넣을 프롬프트**

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

2단계의 JSON 은 모델 이름이 다음과 같다고 가정하고 쓰여 있다. 3번 목록과 다르면 **먼저 알려달라.**
고쳐서 다시 주겠다.

```text
my-local-provider/HCP-Big-Latest
my-local-provider/HCP-Medium-Latest
my-local-provider/HCP-Small-Latest
my-local-provider/HCP-Vision-Latest
itc-vlm/qwen3.8-27b
```

또 1번의 `defaultModel` 에 기존 값이 있으면 2단계에서 **덮어써진다.** 그 값을 유지하고 싶으면
알려달라 — 2단계 JSON 에서 그 줄을 빼고 주겠다.

---

## 2단계 — settings.json 병합

여기서 `agentOverrides`·`modelScope`·`watchdog` 이 들어간다. **1단계에서 문자열이 어긋난 게
있었다면 먼저 알려달라.** 아래 블록의 모델 이름을 실제 값으로 고쳐서 다시 주겠다.

**붙여넣을 프롬프트**

```text
~/.pi/agent/settings.json 에 아래 JSON 을 병합해라.

규칙:
- 먼저 settings.json 을 settings.json.bak 으로 복사해라.
- 아래에 있는 키만 손대라. 기존의 다른 키(theme, packages 등)는 한 글자도 바꾸지 마라.
- subagents 를 통째로 교체하지 마라. 그 안의 키만 합쳐라 (깊은 병합).
- agentOverrides 안에 이미 있는 역할은 통째로 교체해도 된다. 단 교체 전에 원래 내용을 출력해라.
- 값을 새로 지어내지 마라. 아래 있는 값을 그대로 써라.
- 쓰기 전에 "추가한 키" 와 "덮어쓴 키(이전 -> 이후)" 를 전부 나열해서 나에게 보여줘라.
- 쓴 다음 파일을 다시 읽어서 JSON 으로 파싱되는지 확인하고, subagents.agentOverrides 의
  키 목록과 subagents.modelScope.enforce 값을 출력해라.

```json
{
  "defaultModel": "my-local-provider/HCP-Big-Latest",
  "defaultThinkingLevel": "high",
  "httpIdleTimeoutMs": 900000,
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
      "cadence": {
        "everyNTools": 10
      },
      "children": {
        "enabled": true,
        "model": "my-local-provider/HCP-Medium-Latest",
        "cadence": {
          "everyNTools": 10
        }
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

기존에 있던 역할 하나만 남아 있으면 깊은 병합이 아니라 **교체**가 일어난 것이다.
`settings.json.bak` 으로 되돌리고 다시 시도한다. `modelScope.enforce` 는 `true` 여야 한다.

**보고할 것** — 추가/덮어쓴 키 목록, 그리고 마지막 확인 출력.
특히 **덮어쓴 키** 에 `defaultModel` 이 있으면 이전 값이 무엇이었는지 꼭 확인한다.

---

## 3단계 — subagent config.json 생성

`timeoutMs` 는 `settings.json` 이 아니라 **이 파일**에 있어야 한다. 자리를 틀리면 오류 없이 무시된다.

**붙여넣을 프롬프트**

```text
~/.pi/agent/extensions/subagent/config.json 을 만들어라.
디렉터리가 없으면 만들고, 파일이 이미 있으면 .bak 으로 백업한 뒤 아래 키만 병합해라.

```json
{
  "timeoutMs": 10800000,
  "toolTimeoutMs": 900000,
  "asyncByDefault": false
}
```

쓴 다음 파일 내용을 그대로 출력해라.
```

**보고할 것** — 최종 파일 내용.

---

## 4단계 — 배선 확인

설정이 실제로 먹었는지 pi 자신에게 확인시킨다.

**붙여넣을 프롬프트**

```text
subagent 도구로 사용 가능한 에이전트 목록을 조회해라 (action: "list").
각 에이전트의 이름과, 해석된 모델이 보이면 모델도 함께 보여줘라.

그다음 다음을 확인해서 알려줘라.
- oracle, worker, reviewer, scout 네 개가 모두 보이는가
- researcher 와 evidence-auditor 가 목록에서 빠졌는가 (빠져야 정상이다)
```

**성공 판정** — 네 역할이 보이고, researcher·evidence-auditor 가 안 보인다.

**보고할 것** — 목록 그대로.

---

## 5단계 — 서브에이전트 왕복 (첫 실전 확인)

부모가 자식을 실제로 띄우는지 본다. 여기까지 되면 배선은 끝이다.

**붙여넣을 프롬프트**

```text
scout 서브에이전트를 하나 띄워서 현재 디렉터리의 파일 목록을 조사하게 해라.
네가 직접 하지 말고 반드시 subagent 도구를 써라.

자식이 끝나면 다음을 알려줘라.
- 자식이 실제로 떴는가
- 자식이 어떤 모델로 돌았는가 (알 수 있으면)
- 자식이 돌려준 결과
```

**성공 판정** — 자식이 뜨고 결과가 돌아온다.

**실패하면** — 부모가 직접 처리해버린 것일 수 있다(설정 문제 아님). 그 경우 "직접 하지 말라"
고 한 번 더 강조해서 재시도한다. 그래도 안 되면 오류 메시지를 그대로 보고한다.

---

## 6단계 — 실제 작업 한 바퀴

버려도 되는 워크트리에서 작은 작업을 한 번 돌린다.

```powershell
cd <작업할 저장소>
git worktree add ../wt-test -b test/orch
cd ../wt-test
pi
```

**붙여넣을 프롬프트**

```text
scout 으로 <작은 모듈 이름> 관련 파일과 호출 지점을 찾아줘. 고치지는 말고.
```

**여기서 볼 것** — scout 이 **호출부를 빠뜨리지 않았는지** 직접 확인한다.
이게 scout 모델을 고른 이유이고, 사내 코드베이스에서만 확인할 수 있다.
부족하면 알려달라. 더 큰 모델로 올리겠다.

이어서 worker → reviewer 까지 한 바퀴 돌려보고, 끝나면 워크트리째 버린다.

```powershell
cd -
git worktree remove ../wt-test --force
```

---

## 막히면

각 단계에서 **오류 메시지를 요약하지 말고 그대로** 가져오면 된다.
pi 가 "했다" 고만 하고 근거를 안 보여주면, 그 단계 프롬프트 끝에 이 줄을 붙인다.

```text
실제 파일 내용과 실행한 명령의 출력을 그대로 보여줘라. 요약하지 마라.
```

## 관련 문서

- [오케스트레이션 개요](./README.md)
- [퇴근 원샷 프롬프트](./oneshot.md) · [결정 정책 템플릿](./decisions.example.md)
- [작업 저장소 AGENTS.md 규칙](./agents-md.snippet.md)
