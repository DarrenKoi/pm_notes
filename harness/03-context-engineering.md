---
tags: [harness-engineering, context-engineering, compaction, kv-cache]
level: intermediate
last_updated: 2026-09-11
---

# 03. 컨텍스트 엔지니어링 (Context Engineering)

> 컨텍스트 윈도우는 유한한 주의력 예산(attention budget)이다. 매 턴 "원하는
> 행동을 끌어낼 가능성이 가장 높은 최소한의 토큰"을 넣는 것이 목표다.

## 왜 필요한가? (Why)

- **길수록 나빠진다.** Chroma의 *Context Rot* 연구는 18개 모델을 테스트했는데,
  모든 모델이 입력이 길어질수록 단순한 검색·복사 과제에서도 성능이 떨어졌다.
  윈도우가 100만 토큰이라고 해서 100만 토큰을 잘 쓴다는 뜻은 아니다.
- **에이전트는 입력이 압도적으로 많다.** Manus는 입력과 출력의 토큰 비율이 평균
  약 100:1이라고 밝혔다. 컨텍스트는 매 턴 누적되는데 출력은 짧은 도구 호출
  하나뿐이기 때문이다. 비용과 지연 시간은 대부분 입력 토큰에서 나온다.
- **긴 작업은 결국 윈도우를 넘는다.** 몇 시간짜리 작업은 한 윈도우에 담기지
  않는다. 세션을 넘어 이어갈 방법이 필요하다.

## 핵심 개념 (What)

### 1. 컨텍스트를 구성하는 것들

```text
고정된 앞부분 (prefix):   [시스템 프롬프트] [도구 정의] [메모리/AGENTS.md]
계속 늘어나는 뒷부분:     [대화 이력] [도구 결과] [검색 문서]
```

앞부분은 가능한 한 고정하고, 뒷부분의 증가를 관리하는 것이 핵심이다.

### 2. 네 가지 전략

| 전략 | 방법 | 예 |
|---|---|---|
| **외부화 (Write)** | 컨텍스트 밖(파일, DB)에 적고 필요할 때 다시 읽는다 | todo.md, 진행 노트, 중간 산출물 파일 |
| **선별 (Select)** | 전부 미리 넣지 말고 필요할 때 가져온다 (just-in-time) | 파일 경로만 주고 `read_file`로 읽게 함, 스킬 문서의 점진적 공개 |
| **압축 (Compress)** | 오래된 이력을 요약하거나 지운다 | compaction, 오래된 도구 결과 삭제 |
| **격리 (Isolate)** | 하위 작업을 별도 컨텍스트에서 돌리고 결과 요약만 받는다 | 서브에이전트 ([09](./09-multi-agent.md)) |

### 3. KV 캐시를 깨지 않는 설계

Manus는 "KV 캐시 적중률이 프로덕션 에이전트에서 가장 중요한 단일 지표"라고 했다.
앞부분(prefix)이 바이트 단위로 같아야 캐시가 적중하고, 적중하면 첫 토큰
지연(TTFT)과 비용이 크게 줄어든다. 사내에서 직접 서빙하는 모델도 vLLM 같은
엔진의 prefix caching이 같은 원리로 동작한다.

- 시스템 프롬프트 맨 앞에 초 단위 타임스탬프 같은 매번 바뀌는 값을 넣지 않는다
- 이력은 **추가만(append-only)** 한다. 과거 메시지를 중간에서 고치면 그 뒤
  캐시가 모두 무효가 된다
- JSON 직렬화 순서를 고정한다 (`json.dumps(..., sort_keys=True)`)
- 도구를 턴마다 넣었다 뺐다 하지 않는다. Manus는 도구 정의는 그대로 두고 디코딩
  단계에서 선택지를 가리는(masking) 방식을 썼다
- compaction은 캐시를 깨므로 매 턴 조금씩 하지 말고, 한도에 가까워졌을 때 한
  번에 크게 한다

### 4. 주의를 끌어오는 기법

- **목표 되새기기(recitation)**: 긴 작업에서 todo.md를 계속 갱신하고 다시 읽게
  하면, 목표가 컨텍스트 끝(가장 주목받는 위치)에 반복해서 놓인다 (Manus)
- **에러를 지우지 않는다**: 실패한 시도와 에러 메시지를 남겨두면 모델이 같은
  실수를 덜 반복한다 (Manus)
- **AGENTS.md는 백과사전이 아니라 지도다**: OpenAI 팀은 거대한 AGENTS.md 하나가
  실패한 뒤 약 100줄짜리 목차로 바꿨다. 세부 내용은 `docs/` 아래 문서로 연결해
  필요할 때 읽게 했다

### 5. 세션을 넘는 긴 작업

Anthropic의 장기 실행 하네스는 첫 실행 때 initializer 에이전트가 환경(기능 목록,
git 저장소, 진행 추적 파일)을 만든다. 이후 세션마다 코딩 에이전트가 조금씩
진전시키고, **다음 세션을 위한 명확한 산출물**을 남긴다. 교대 근무자가 인수인계
노트를 남기는 것과 같다. 요약에만 기대지 않고, 구조화된 파일과 git 이력이 기억
역할을 한다.

## 어떻게 사용하는가? (How)

### Step 1. 오래된 도구 결과 지우기 (가장 싸고 먼저 할 것)

요약용 LLM 호출이 필요 없다. 다시 가져올 수 있는 결과(파일 내용, 검색 결과)는
지워도 안전하다.

```python
CLEARED = "[cleared: call the tool again if you need this]"


def clear_old_tool_results(messages: list[dict],
                           keep_last: int = 6) -> list[dict]:
    """최근 keep_last개를 뺀 tool 결과를 짧은 안내문으로 바꾼다."""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    for i in tool_idx[:-keep_last]:
        messages[i] = {**messages[i], "content": CLEARED}
    return messages
```

이 방법은 과거 메시지를 수정하므로 KV 캐시가 깨진다. 매 턴 하지 말고 한도에
가까울 때 한 번에 적용한다.

### Step 2. Compaction (요약으로 이력 압축)

```python
import json

COMPACT_PROMPT = """Summarize this agent transcript for handoff to yourself.
Keep: the goal, decisions and why, files/IDs touched (exact paths),
what was verified and how, errors still unresolved, the next concrete step.
Drop: raw tool output."""


def estimate_tokens(messages: list[dict]) -> int:
    # 거친 추정. 정확도가 필요하면 서빙 서버의 tokenizer로 센다
    return len(json.dumps(messages, ensure_ascii=False)) // 2


def compact(client, model: str, messages: list[dict],
            limit: int = 60_000, keep_last: int = 8) -> list[dict]:
    if estimate_tokens(messages) < limit:
        return messages
    cut = len(messages) - keep_last
    if cut <= 1:
        return messages
    # 최근 구간이 tool 결과로 시작하면 짝이 되는 tool_call이 잘려서
    # API 에러가 난다. 그래서 경계를 앞으로 당긴다.
    while cut > 1 and messages[cut]["role"] == "tool":
        cut -= 1
    old, recent = messages[1:cut], messages[cut:]
    summary = client.chat.completions.create(model=model, messages=[
        {"role": "system", "content": COMPACT_PROMPT},
        {"role": "user", "content": json.dumps(old, ensure_ascii=False)},
    ]).choices[0].message.content
    note = {"role": "user", "content": f"[Summary of earlier work]\n{summary}"}
    return [messages[0], note] + recent
```

`run_agent` 루프에서 모델 호출 직전에
`messages = compact(client, model, messages)`를 넣는다. 요약 품질도 eval
대상이다. 요약 뒤에 에이전트가 이미 한 일을 반복한다면 요약 프롬프트를 고친다.

### Step 3. 인수인계 파일 (세션을 넘는 작업)

```markdown
<!-- progress.md: 에이전트가 매 세션 끝에 갱신하고,
     다음 세션 시작 시 가장 먼저 읽는다 -->
# Goal
사내 문서 1,200건에서 표 추출 파이프라인 완성

# Done (verified)
- [x] 페이지 스크린샷 변환 — 샘플 50건 육안 확인
- [x] VLM 표 추출 — eval 30건 중 27건 통과 (scripts/eval_tables.py)

# In progress
- 병합 셀 처리: tests/test_merge.py 3건 실패, 원인 가설 = 행 병합 좌표 누락

# Next step
1. tests/test_merge.py 실패 로그 확인
2. ...

# Do not
- output/ 원본 폴더 덮어쓰지 말 것
```

## 학습 체크리스트

- [ ] 긴 작업을 돌리며 턴별 입력 토큰 수를 기록하고, 어느 턴부터 품질이
      떨어지는지 봤다
- [ ] 시스템 프롬프트에서 매 요청마다 바뀌는 값을 제거했다
- [ ] 도구 결과 지우기와 compaction을 각각 적용하고 eval 점수와 토큰 사용량을
      비교했다
- [ ] AGENTS.md(또는 시스템 프롬프트)를 목차 수준으로 줄이고 세부 내용은 파일로
      분리했다
- [ ] 사내 서빙 엔진의 prefix cache 적중률 지표를 찾아봤다

## 참고 자료 (References)

- [Effective context engineering for AI
  agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
  — Anthropic
- [Context Engineering for AI Agents: Lessons from Building
  Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
  — Yichao "Peak" Ji, 2025-07
- [Context Rot: How Increasing Input Tokens Impacts LLM
  Performance](https://www.trychroma.com/research/context-rot) — Chroma, 2025-07
- [Effective harnesses for long-running
  agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
  — Anthropic
- [Harness engineering](https://openai.com/index/harness-engineering/) — OpenAI,
  AGENTS.md를 지도로 쓰는 방식

## 관련 문서

- [02. 에이전트 루프](./02-agent-loop.md)
- [07. 상태와 복구](./07-state-and-recovery.md)
- [09. 멀티 에이전트](./09-multi-agent.md)
