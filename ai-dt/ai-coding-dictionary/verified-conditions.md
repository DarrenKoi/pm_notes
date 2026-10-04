---
type: learning
tags: [ai-coding, source-verification, glossary]
aliases: [AI 코딩 용어의 적용 조건]
reviewed_on: 2026-10-04
review_status: partial
---

# AI 코딩 용어의 검증된 적용 조건

## 목적과 읽기 방법

7개 학습 노트는 모델·작업 단위·환경·실패·인계·메모리·협업 패턴으로 나눈 설명이다. 사전 저자의 비유와 공식 제품 계약을 구분해 사용한다. 각 용어의 고유 대화는 본문에 보존하고, 반복되는 검증 범위는 여기서 읽는다. 사내 사용 사례는 제안/가상 예시이며 실제 관측으로 검증하지 않았다. 작성일 2026-05-05와 확인일 2026-10-04는 별개다.

## 근거와 적용 범위 · 확인 2026-10-04

버전이 없는 제품 문서는 확인 당시 동적 문서다. 모든 dependency나 제품 동작을 로컬 재현했다고 주장하지 않는다. 가격 수치·현재 모델 추천은 재게시하지 않는다.

| 일차 자료/설명 출처 | 확인한 범위 |
|---|---|
| [원래 사전 main](https://github.com/mattpocock/dictionary-of-ai-coding) | 저자의 비유·용어 선택과 현재 추가 항목. 모델명·일반 법칙의 검증 근거로 삼지 않음 |
| [HF PEFT 동적 문서](https://huggingface.co/docs/peft/en/index) | 사용자 fine-tuning과 일부 파라미터 학습 가능; 실제 학습 미실행 |
| [HF Transformers main](https://huggingface.co/docs/transformers/main/en/generation_strategies) | greedy·sampling·beam 등 decoding 구분; 설치 버전과 동일하다는 주장 아님 |
| [DPO arXiv 2305.18290](https://arxiv.org/abs/2305.18290) | post-training 목적이 next-token likelihood 학습 하나로 제한되지 않음 |
| [OpenAI Responses 동적 문서](https://developers.openai.com/api/docs/guides/conversation-state) | previous_response_id/서버 state와 클라이언트 전체 history 전송 구분; 이전 입력 과금·입출력/추론 한도 확인 |
| [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling) | 한 모델 응답의 0/1/여러 function calls; 호출별 result 식별 |
| [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) | 정확한 prefix·최소 길이·cache usage 조건; API별 과금 |
| [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | read와 write 구분·TTL·breakpoint·최소 길이; write가 일반 입력보다 비쌀 수 있음 |
| [Claude Code subagents](https://code.claude.com/docs/en/sub-agents) | v2.1.219 이후 기본 깊이 3 안내; 깊이 환경 설정·권한별 조건 |
| [Claude Code memory](https://code.claude.com/docs/en/memory) | v2.1.277 이상 AGENTS.md 직접 지원, CLAUDE.md 존재·설정별 로딩 조건; 여기서는 실제 세션 미확인 |
| [Claude Code permissions](https://code.claude.com/docs/en/permissions) | 제품별 permission modes; bypass가 모든 관리/격리 경계 해제를 뜻하지 않음 |
| [Claude Code sandbox](https://code.claude.com/docs/en/sandboxing) | 위험 감소와 완전한 보안 경계 구분; 연결·마운트 정책 확인 |
| [Claude Code 실행 구조](https://code.claude.com/docs/en/how-claude-code-works) | context 관리·압축·도구 실행 역할; 같은 세션의 압축과 새 작업 구분 |
| [AGENTS.md convention](https://agents.md/) | 개방된 지침 파일 관례; 모든 제품의 자동 로드 보장 아님 |
| [Agent Skills specification](https://agentskills.io/specification) | metadata → 본문 → 필요한 resources 점진적 로드; 읽을 때 비용 발생 |
| [Attention Is All You Need v7](https://arxiv.org/abs/1706.03762v7) | query/head softmax와 mask; 전체 성능을 고정된 주의력 예산 하나로 설명하지 않음 |
| [Lost in the Middle v3](https://arxiv.org/abs/2307.03172v3) | 해당 실험의 관련 정보 위치별 성능; 모든 모델의 100k 임계값 입증 아님 |
| [Sycophancy 연구 v4](https://arxiv.org/abs/2310.13548v4) | 선호 판단이 가능한 기여 요인; 개별 답변의 유일 원인 확정 아님 |

## 실무에서 확인할 계약

1. **모델과 학습:** 추론의 고정 weights, 서비스의 conversation state, 학습의 weights/adapter 변경을 구분한다. API 문서 전달과 fine-tuning은 동일 작업이 아니며 총비용·데이터 권한·평가 목표로 선택한다.
2. **요청과 비용:** 사용자 turn 수·도구 실행 수·모델 요청 수를 각각 센다. 한 응답에서 여러 도구를 실행하고 results를 모을 수 있다. history를 클라이언트가 재전송하지 않아도 서버가 과거 입력을 사용/과금할 수 있다. cache write/read·일반 input·output·도구 비용은 usage와 가격표에서 분리한다.
3. **실패 진단:** 버전·입력·도구 결과·실행 로그를 먼저 확인한다. 사실성/충실성은 관점이며 docs 존재 여부만으로 원인이나 치료를 확정할 수 없다. 긴 입력 축소·관련 정보 재배치·압축은 반복 평가로 비교한다. “smart/dumb zone”, 고정 attention budget, 100k 임계값은 보편 법칙이 아니다.
4. **규칙과 메모리:** 실제 로드된 파일·우선순위·탐색 범위를 확인한다. 지침 파일·skill은 읽혀야 영향을 준다. lazy load에도 목록 metadata 및 본문 로드 후 비용은 남는다. 압축은 손실 가능성이 있어 원본 결정/근거를 파일로 보존하고 재확인한다.
5. **무인 위임:** context 격리와 파일/credential 격리는 다르다. 원본 read-only와 writable 작업 복사본을 구분한다. sandbox·승인·작업 제한을 실제 정책으로 확인하고 subagent 중첩·비동기 결과 수신 조건은 하네스별로 본다. 본문 AFK 예시는 실행 지시가 아니다.

## 남은 미확인

Brooks 책 본문의 정확한 인용 절, 사내 역량/발생 빈도/비용 절감률, 특정 모델의 성능·현재 계정 지원, 제품별 실제 지침 로딩·권한·subagent 실행은 미확인이다. 원문 main의 추가 용어를 모두 번역하거나 세션·인계 분류 체계를 새로 합치는 결정은 Claude 협의 대기다.

[목차로 돌아가기](./README.md) · [정리 기록](./organization-log.md)
