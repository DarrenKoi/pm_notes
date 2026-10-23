---
tags: [agentic-coding, orchestration, multi-agent, context-engineering, verification]
level: intermediate
last_updated: 2026-10-24
document_type: learning
category_major: "에이전트 오케스트레이션"
category_middle: "배경 지식"
category_minor: "에이전틱 코딩 흐름"
note_kind: "학습"
---

# 에이전틱 코딩(Agentic Coding): 2026년에 정리된 것들

> 에이전트가 코드를 읽고, 고치고, 실행하고, 결과를 보고 다음 행동을 정하는 방식의 개발. 2025~2026년에 무엇이 합의됐고, 멀티 에이전트는 언제 이기며, 어디서 무너지는지 정리한다. 이 폴더의 설정이 왜 그렇게 생겼는지도 여기서 설명한다.

## 왜 알아야 하나 (Why)

이 폴더의 설정값 하나하나는 결국 에이전틱 코딩에서 반복해서 확인된 실패를 막으려는 장치다. 리뷰어에게 `bash`를 돌려주는 것도, 워커마다 파일을 나눠 주는 것도, 야간 실행을 회차당 한 단위로 자르는 것도 다 이유가 있다. 그 이유를 모르고 설정만 복사하면, 모델이나 패키지가 바뀌었을 때 무엇을 지키고 무엇을 버려도 되는지 판단할 수 없다.

또 하나. 이 분야는 반년 사이에도 권장 사항이 바뀐다. 2025년 6월에 "멀티 에이전트를 만들지 마라"라고 했던 Cognition이 2026년 4월에는 "이런 조합은 된다"로 입장을 좁혔다. 무엇이 근거 있는 주장이고 무엇이 업체 블로그의 경험담인지 구분해 두면, 다음 유행이 와도 휘둘리지 않는다.

## 핵심 개념 (What)

### 에이전틱 코딩은 "루프"다

자동완성은 다음 몇 줄을 예측한다. 채팅형 코딩은 답이나 코드를 주고, 적용은 사람이 한다. 에이전틱 코딩은 모델이 도구를 직접 호출해 파일을 읽고, 수정하고, 명령을 돌리고, 그 결과를 보고 다음 행동을 고른다. 차이는 에이전트 수가 아니라 **실행과 피드백이 닫힌 루프**에 있다. Microsoft가 2026년 6월 GitHub Copilot 운영 트레이스를 분석했을 때도 모델 호출과 도구 실행이 거의 1:1로 맞물려 돌았다.

그래서 이 방식에서 가장 비싼 자원은 모델의 지능이 아니라 두 가지다.

- **컨텍스트**: 읽은 파일, 명령 출력, 대화가 전부 한 창에 쌓인다. 창이 찰수록 앞의 지시를 잊고 실수가 늘어난다.
- **검증**: 에이전트는 "끝난 것처럼 보이면" 멈춘다. 돌려 볼 수 있는 검사가 없으면 사람이 검증 루프가 된다.

Addy Osmani는 2026년 3월 발표에서 이 변화를 한 줄로 정리했다. 병목은 이제 생성이 아니라 검증이다.

### 2025~2026년에 사실상 합의된 실천

업체 문서와 연구가 공통으로 가리키는 것만 추렸다. "표준"이라기보다 여러 곳에서 같은 결론에 도달한 실천이고, 모든 과제에서 효과가 증명된 것은 아니다.

| 실천 | 내용 | 근거 |
|---|---|---|
| 돌려 볼 수 있는 검사를 준다 | 테스트, 빌드 종료 코드, 린터, 스크린샷 비교. 에이전트가 직접 실행해 통과·실패를 읽게 한다 | Claude Code 모범 사례 |
| 탐색·계획과 구현을 나눈다 | 읽기 전용으로 먼저 파악하고 계획을 세운 뒤 고친다. 한 문장으로 설명되는 수정이면 계획을 건너뛴다 | Claude Code 모범 사례 |
| 컨텍스트를 설계한다 | 지시는 짧게, 코드는 필요한 만큼만 불러오고, 도구 출력은 다듬는다. 프롬프트 문구보다 "매 호출에 무엇이 들어가는가"를 관리한다 | Anthropic, Effective context engineering (2025-09) |
| `AGENTS.md`는 짧게 | 에이전트가 코드에서 알아낼 수 없는 것만 적는다. 빌드·테스트 명령, 저장소 관례, 함정 | ETH Zurich 연구 (2026-02), agents.md |
| 스펙을 먼저 쓴다 | 요구사항·범위 밖·완료 조건을 문서로 정하고 구현을 맡긴다 | GitHub Spec Kit (2025-09) |
| 쓴 쪽과 검토하는 쪽을 분리한다 | 새 컨텍스트의 리뷰어가 diff만 보고 판단한다. 자기가 쓴 코드에 관대해지는 편향을 끊는다 | Claude Code 모범 사례, Cognition (2026-04) |
| 긴 작업은 상태를 파일에 둔다 | 진행 목록과 결정을 파일과 git에 남기고, 세션은 매번 새로 시작한다 | Anthropic, Effective harnesses (2025-11) |
| 권한과 샌드박스를 구분한다 | 승인 팝업이나 프롬프트 규칙은 경계가 아니다. 파일 시스템과 네트워크를 OS 수준에서 함께 막아야 한다 | Anthropic, Claude Code sandboxing (2025-10) |

`AGENTS.md`에 대해서는 뜻밖의 결과가 있다. ETH Zurich 연구진(Gloaguen 외, arXiv 2602.11988)이 잘 알려지지 않은 저장소의 실제 과제 138개와 SWE-bench로 실험해 보니, 저장소 컨텍스트 파일은 과제 성공률을 대체로 올리지 못했고 추론 비용은 20% 넘게 올렸다. 2026-09-29 개정판 기준이다. LLM이 자동 생성한 파일은 효과가 거의 없었다. 반대 결과를 낸 연구도 있어 결론은 아직 열려 있다. 저자들이 권하는 것은 길이를 줄이는 것 자체가 아니라, 그 저장소에만 있는 비표준 정보만 적고 효과를 측정해 보는 것이다. 실무 규칙으로 옮기면 이렇다. **에이전트가 스스로 알아낼 수 없는 것만, 사람이 직접 쓴다.**

### 멀티 에이전트는 언제 이기나

"에이전트를 여러 개 띄우면 더 잘한다"는 직관은 반만 맞다. 근거를 날짜순으로 보면 그림이 선명해진다.

1. **Anthropic, 2025년 6월.** 리드 에이전트 하나가 하위 에이전트들에게 조사를 나눠 맡기는 구조가 단일 에이전트보다 내부 리서치 평가에서 90.2% 나았다. 대신 토큰은 일반 채팅의 약 15배를 썼다(단일 에이전트는 약 4배). 이것은 **리서치** 과제이고, 저자들도 서로 얽힌 코딩 작업은 병렬화가 덜 된다고 적었다.
2. **Cognition, 2025년 6월.** "Don't Build Multi-Agents." 병렬로 쓰는 에이전트들은 서로 다른 암묵적 결정을 내리고, 그 충돌을 마지막 에이전트가 떠안는다. 원칙은 둘이다. 맥락을 통째로 공유할 것, 그리고 행동에는 암묵적 결정이 담겨 있다는 것.
3. **Google Research, 2026년 1월.** 180개 구성을 통제 실험했다. 병렬로 쪼갤 수 있는 금융 분석 과제에서는 중앙 조율형 멀티 에이전트가 80.9% 나았다. 순서가 중요한 계획 과제(PlanCraft)에서는 모든 멀티 에이전트 변형이 39~70% 나빠졌다. 서로 소통하지 않는 독립형 구조는 오류를 17.2배 증폭했고, 오케스트레이터가 있는 중앙형은 4.4배로 막았다.
4. **Cursor, 2026년 1월.** 에이전트들을 공유 파일과 잠금으로 평평하게 조율하자 잠금 대기로 시간을 다 썼다. 20개를 띄워도 처리량은 2~3개 수준이었다. 잠금을 풀자 이번에는 다들 쉬운 일만 골랐다. 계획자·작업자·판정자로 층을 나누고, 판정자가 회차마다 계속할지 정한 뒤 다음 회차를 새로 시작하게 하자 굴러갔다.
5. **Cognition, 2026년 4월.** 1년 만의 후속 글에서 입장을 좁혔다. 병렬로 쓰는 에이전트 무리는 여전히 안 된다. 되는 것은 **쓰기는 한 줄기로 유지하고, 추가 에이전트는 행동이 아니라 판단을 보태는** 구조다. 읽기 전용 탐색 에이전트, 새 컨텍스트에서 diff만 보는 리뷰어(PR당 평균 약 2건의 버그를 잡았고 그중 약 58%가 심각), 큰 작업을 나눠 맡기는 관리자 구조가 여기에 든다.

이 연구들의 과제와 모델은 제각각이라 한 줄로 일반화할 수는 없다. 그래도 공통으로 가리키는 방향은 있다. **넓게 읽는 일은 나누면 이기는 경우가 많고, 한 덩어리로 쓰는 일은 나누면 지는 경우가 많다.** 리뷰처럼 새 눈이 필요한 일은 나누는 편이 낫다. 그리고 나눌수록 토큰 비용은 대개 크게 는다.

### 어디서 무너지나

| 실패 | 무슨 일이 생기나 | 근거 |
|---|---|---|
| 검증 위조 | 단위 테스트 몇 개만 돌리고 "완료"라고 보고한다. 실제 사용 경로는 깨져 있다 | Anthropic, Effective harnesses (2025-11) |
| 테스트 조작 | 테스트를 고치거나 결과를 하드코딩해서 통과시킨다. Codex와 Claude Code 모두에서 관찰됐다 | EvilGenie 벤치마크 (MIT, 2026-05 개정) |
| 지시 파일 주입 | 악성 Go 의존성이 `AGENTS.md`를 써서 코드에 5분 지연을 몰래 넣게 했다 | NVIDIA AI Red Team (2026-04) |
| 승인 피로 | 사용자는 승인 요청의 93%를 그냥 통과시킨다. 팝업은 감독이 아니다 | Anthropic, Claude Code auto mode (2026-03) |
| 리뷰 습관화 | 에이전트 PR을 반복해서 본 리뷰어 400명은 7개월 동안 승인률이 30.1%에서 36.8%로 오르고 인라인 코멘트가 22% 줄었다 | arXiv 2606.22721 (2026-06) |
| 무한 수정 루프 | 같은 잘못된 방법을 계속 다시 시도한다 | Addy Osmani (2026-03) |

습관화 연구는 사람 리뷰어를 측정한 것이다. 에이전트 리뷰어에도 비슷한 일이 생기는지는 측정된 바 없지만, 같은 모델 계열이 같은 diff를 계속 보면 같은 것을 놓칠 수 있다고 보고 대비하는 편이 안전하다. 독립성은 모델 이름이 아니라 **새 컨텍스트, 코드를 직접 읽기, 검증 명령을 직접 다시 돌리기**에서 나온다.

## 어떻게 적용하는가 (How)

### 이 폴더의 설계와 맞춰 보기

| 확인된 원칙 | 이 폴더에서 구현한 방식 | 위치 |
|---|---|---|
| 넓게 읽는 일만 나눈다 | `scout`에게 탐색만 맡긴다. 단 `scout`도 `bash`·`write`를 가지므로 "고치지 마라"는 지시일 뿐 강제는 아니다 | [README](./README.md) 역할 표 |
| 쓰기는 한 줄기로 | 파일이 겹치지 않게 쪼개고, 겹치면 순서를 준다. 병렬 워커에는 워크트리를 따로 준다 | [원샷 템플릿](./oneshot.md) |
| 쓴 쪽과 검토하는 쪽을 분리 | `reviewer`를 새 컨텍스트로 띄운다. 도구에 `bash`를 돌려주고 시스템 프롬프트도 바꿔 검증 명령을 직접 재실행하게 한다 | [settings.snippet.json](./settings.snippet.json) |
| 검증은 증거로 | 보고에 실행한 명령과 실제 출력을 붙이게 한다. 돌리지 않은 것은 돌렸다고 쓰지 않는다 | [AGENTS.md 규칙](./agents-md.snippet.md) |
| 긴 작업은 상태를 파일에 | 야간 실행은 회차당 한 단위, 진행 상황은 `.orch/진행상황.md`와 git에 둔다 | [야간 세팅](./night-setup.md) |
| 수정 루프에 상한 | 단위당 수정 2회, 단위 수 상한, 실행당 자식 수 상한 | [원샷 템플릿](./oneshot.md) |
| 프롬프트는 경계가 아니다 | 하드 가드와 정책은 실수 방지용이다. 실제 경계는 컨테이너나 VM이 필요하다고 명시한다 | [결정 정책](./decisions.example.md) |
| 사내 경로만 쓴다 | 웹 검색 역할과 외부 CLI 역할(`claude-code`, `codex-exec`, `cursor-agent` 계열)을 끈다. 끄는 것은 라우팅 정책이지 네트워크 차단이 아니다 | [settings.snippet.json](./settings.snippet.json) |

### 사내 오픈 웨이트 모델에서 더 챙길 것

외부 API를 못 쓰고 사내 게이트웨이의 GLM·Qwen 계열만 쓰는 환경에서는 아래를 먼저 확인한다. 모두 일반 원칙을 이 환경에 적용한 권고이고, 사내 모델로 측정한 결과는 아니다.

- **도구 호출이 실제로 되는지부터 본다.** OpenAI 호환 API라고 해서 도구 호출이 같은 방식으로 도는 것은 아니다. Qwen3.6 모델 카드는 추론 파서와 도구 호출 파서를 따로 설정하라고 안내한다. 도구를 못 부르는 증상이 모델 지능 문제가 아니라 서빙 설정 문제일 수 있다. `smoke.sh -l 2`가 이 확인용이다.
- **도구는 적게, 출력은 짧게.** 도구 호출이 약한 모델일수록 도구 수를 줄이고, 설명을 분명히 하고, 출력 길이에 상한을 두고, 오류 메시지가 다음 행동을 알려 주게 만든다.
- **모델 사양의 최대 컨텍스트가 실제 작업에 알맞은 입력 크기를 뜻하지는 않는다.** GLM-5.3은 100만 토큰을 받지만, 창이 찰수록 성능이 떨어지는 것은 같다. 자식에게는 새 컨텍스트를 주고, 부모가 자식의 전사를 흡수하지 않게 한다. TPM 한도에도 이 편이 유리하다.
- **쓰는 에이전트는 하나부터 시작한다.** 읽기 전용 탐색과 리뷰만 나누고, 병렬 작성은 통합 실패율과 총비용을 단일 에이전트와 비교해 본 뒤에 늘린다.
- **같은 계열끼리의 검증을 믿지 않는다.** Big과 Medium은 둘 다 GLM이다. 영향이 큰 변경은 계열이 다른 Qwen으로 2차 검증한다.

### 다음 유행을 판단하는 질문

새 패턴이나 도구가 나오면 아래 네 가지를 묻는다.

1. 이것은 넓게 읽는 일을 나누는가, 한 덩어리로 쓰는 일을 나누는가?
2. 결과를 무엇으로 검증하는가? 다른 모델의 "좋아 보인다"가 아니라 돌려 볼 수 있는 검사가 있는가?
3. 토큰과 호출이 몇 배로 느는가? 사내 RPM/TPM 한도 안에 들어오는가?
4. 실패하면 되돌릴 수 있는가? 그 경계는 프롬프트인가, OS인가?

## 참고 자료 (References)

모든 링크는 2026-10-24에 확인했다. 업체 블로그는 그 회사의 실험과 경험이며, 일반적인 성능 보장이 아니다.

- Claude Code 모범 사례 (상시 갱신 문서): <https://code.claude.com/docs/en/best-practices>
- Anthropic, Effective context engineering for AI agents (2025-09-29): <https://www.anthropic.com/research/effective-context-engineering-for-ai-agents>
- Anthropic, How we built our multi-agent research system (2025-06-13): <https://www.anthropic.com/research/multi-agent-research-system>
- Anthropic, Effective harnesses for long-running agents (2025-11-26): <https://www.anthropic.com/research/effective-harnesses-for-long-running-agents>
- Anthropic, Making Claude Code safer with sandboxing (2025-10-20): <https://www.anthropic.com/research/claude-code-sandboxing>
- Anthropic, How we built Claude Code auto mode (2026-03-25): <https://www.anthropic.com/research/claude-code-auto-mode>
- Cognition, Don't Build Multi-Agents (2025-06-12): <https://cognition.com/blog/dont-build-multi-agents>
- Cognition, Multi-Agents: What's Actually Working (2026-04-22): <https://cognition.com/blog/multi-agents-working>
- Google Research, Towards a science of scaling agent systems (2026-01-28): <https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/>
- Cursor, Scaling long-running autonomous coding (2026-01-14): <https://cursor.com/blog/scaling-agents>
- Addy Osmani, The Code Agent Orchestra (2026-03-26): <https://addyosmani.com/blog/code-agent-orchestra/>
- Microsoft Research, Agentic Coding in the Wild (2026-07): <https://www.microsoft.com/en-us/research/publication/agentic-coding-in-the-wild-characterizing-github-copilot-at-production-scale/>
- Gloaguen 외, Evaluating AGENTS.md (arXiv 2602.11988, 2026-02): <https://arxiv.org/abs/2602.11988>
- NVIDIA, Mitigating indirect AGENTS.md injection attacks (2026-04-20): <https://developer.nvidia.com/blog/mitigating-indirect-agents-md-injection-attacks-in-agentic-environments/>
- EvilGenie: A Reward Hacking Benchmark (arXiv 2511.21654): <https://arxiv.org/abs/2511.21654>
- Habituation at the Gate (arXiv 2606.22721, 2026-06-21): <https://arxiv.org/abs/2606.22721>
- GitHub Spec Kit (2025-09-02): <https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/>
- AGENTS.md: <https://agents.md/>
- Qwen3.6-35B-A3B 모델 카드: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>

## 관련 문서

- [Pi 오케스트레이션 개요](./README.md)
- [작업 저장소 AGENTS.md 규칙](./agents-md.snippet.md)
- [퇴근 원샷 프롬프트](./oneshot.md) · [야간 세팅](./night-setup.md)
