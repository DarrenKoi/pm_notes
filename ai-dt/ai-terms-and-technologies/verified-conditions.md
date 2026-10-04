---
type: learning
tags: [ai, glossary, source-verification]
aliases: [AI 입문 자료의 적용 조건]
reviewed_on: 2026-10-04
review_status: partial
---

# AI 입문 자료의 근거와 적용 조건

## 대표 문서와 통합본의 차이

[목차](./README.md)의 01~09는 예제·비교·점검표가 있는 상세 문서다. [통합본](./all-in-one.md)은 같은 개념군을 연속해 읽는 요약이다. 모든 상세 예제를 대체하지 않는다. 공통 출처·수치 가정·검증 범위는 이 문서를 대표로 읽고 각 본문의 실제 조건을 함께 확인한다. 사내 사례·AX 범주는 학습/계획 예시이며 실제 구현·측정 결과가 아니다.

## 일차 근거 · 확인 2026-10-04

원논문은 해당 연구 조건의 근거이며 오늘의 모든 제품 성능으로 확대하지 않는다. 동적 공식 문서는 확인 당시 내용으로, 특정 로컬 설치와 동일하다고 주장하지 않는다.

| 근거 | 적용 범위 |
|---|---|
| [모델·행동 출력](https://arxiv.org/abs/2307.15818) | RT-2 논문의 VLA 입력/행동 출력; 모든 로봇의 안전 보장 아님 |
| [MoE](https://arxiv.org/abs/1701.06538) | sparse gated experts의 일부 활성화; GPU별 상주량/속도는 serving 설계별 |
| [지식 증류](https://arxiv.org/abs/1503.02531) | teacher 분포/soft target과 student 학습; 성능 전부 보존 보장 아님 |
| [H200 사양](https://www.nvidia.com/en-us/data-center/h200/) | SXM/NVL 141GB 표 확인; 140GB·80%는 본문 계산 가정 |
| [GPT-3 few-shot](https://arxiv.org/abs/2005.14165) | in-context 예시와 weights 갱신을 구분; 모든 few-shot learning 정의로 확대하지 않음 |
| [RAG 원논문](https://arxiv.org/abs/2005.11401) | parametric와 non-parametric memory 결합; 원논문은 학습도 포함하므로 모든 RAG가 학습 없는 시스템이라는 뜻 아님 |
| [Retrieve/Rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) | bi-encoder/lexical 후보와 cross-encoder; 후보 수·latency/성능은 평가 |
| [Ontology](https://www.w3.org/TR/owl2-overview/) | OWL 2 Second Edition, 2012-12-11; 온톨로지의 형식 언어 예이며 모든 온톨로지가 OWL 필수 아님 |
| [LoRA](https://arxiv.org/abs/2106.09685) | base weights 고정·저랭크 업데이트; 모든 추가 모듈의 자동 고정 보장 아님 |
| [Judge 평가](https://arxiv.org/abs/2306.05685) | MT-Bench/Chatbot Arena 해당 실험의 position·verbosity·self-enhancement bias; 업무별 human calibration 필요 |
| [리스크·책임](https://airc.nist.gov/airmf-resources/airmf/) | NIST AI RMF 1.0의 risk 관리 참고; 본문 다섯 능력/AX 다섯 범주는 노트 작성자의 정리 |
| [AI 전력·냉각](https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai) | Energy and AI 2025 보고서의 2024 시설 구성·시나리오; 현재 모든 시설 수치로 확대하지 않음 |
| [Reasoning API](https://developers.openai.com/api/docs/guides/reasoning) | 동적 문서의 effort/reasoning 과금 설명; 계정·모델별 지원 확인 |
| [Context API](https://developers.openai.com/api/docs/guides/conversation-state) | history state/입출력·reasoning token 한도; 모든 입력을 항상 다시 전송하는 계약 아님 |
| [Decoding](https://huggingface.co/docs/transformers/main/en/generation_strategies) | main 문서의 greedy/sampling/beam; 설치 배포 버전 실행 검증 아님 |
| [긴 입력 활용](https://arxiv.org/abs/2307.03172v3) | 2023년 v3 위치별 실험; 모든 최신 모델/입력의 필연적 저하 아님 |
| [MCP](https://modelcontextprotocol.io/specification/2026-07-28/architecture) | 2026-07-28 host/client/server·resources/tools/prompts; 기능·인증 호환성 필요 |
| [Claude Code](https://code.claude.com/docs/en/sub-agents) | 동적 문서의 위임/중첩 설정; 이번 폴더의 실제 spawn 미실행 |
| [Codex](https://learn.chatgpt.com/docs/agent-configuration/subagents) | 공식 위임 구조·명시 요청/적용 지침 조건; 계정/제품별 UI 차이 |
| [Hermes](https://hermes-agent.nousresearch.com/docs/guides/delegation-patterns/) | 동적 문서, 기본 max_spawn_depth 1·중첩 opt-in·session lifetime 조건 |
| [Pi 확장](https://pi.dev/packages/pi-sub-agent) | Pi 패키지 카탈로그의 pi-sub-agent 0.1.5; 다른 subagent 패키지와 동치 아님/미설치 |
| [인젝션 방어](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) | 직접/간접·system prompt extraction·data exfiltration 및 다층 방어; 완전 방어 보장 아님 |
| [6월 12일 제공자 성명](https://www.anthropic.com/news/fable-mythos-access) | Anthropic의 foreign nationals 지시/모든 사용자 중단 설명; 정부 원문·현재 법적 범위는 별도 미확인 |
| [6월 30일 재배포 공지](https://www.anthropic.com/news/redeploying-fable-5) | 7월 1일 Fable 재제공 안내와 false positive 조건; 현재 접근권 단정 아님 |
| [7월 2일 safeguards](https://www.anthropic.com/news/fable-safeguards-jailbreak-framework) | CJS 제안 초안·제공자 입장; 합의된 표준이나 위험 해소 증명 아님 |
| [7월 16일 HF 공개](https://huggingface.co/blog/security-incident-july-2026) | HF의 내부 dataset/credential 접근·public tampering 증거 없음 설명; 당시 사용 LLM 불명 |
| [7월 21일/28일 OpenAI](https://openai.com/ko-KR/index/hugging-face-model-evaluation-security-incident/) | 평가 refusal/classifier 조건·후속 prototype 조치의 공개 설명; 독립 재현/최종 조사 결론 아님 |

## 학습에서 업무 적용으로 옮길 때

- **모델/GPU:** LLM과 SLM의 크기 경계는 보편 고정값이 아니다. VLM/VLA는 입출력 기준이다. 140GB × 80%=112GB는 decimal 단순 가정이며 BF16 2byte/4-bit 0.5byte weights 계산에는 KV cache·scale·통신·모듈이 빠져 있다. 표의 GPU 장수는 구매/처리량/실제 fit 보장값이 아니다.
- **검색:** 질문/문서 embedding 모델·revision·dimension·거리 정책을 맞춘다. 검색 provenance·ACL·유효 기준을 별도로 관리한다. chunk/reranker 개선은 검색과 생성 평가를 분리해 측정한다. 문서를 적재할 때 추론 모델 weights를 바꾸지 않는 일반 RAG pipeline과 원논문의 joint fine-tuning을 구분한다.
- **토큰:** 본문 한국어 조각 도식은 가상 예시다. 동일 tokenizer/revision으로 실제 문자열을 재어 비교한다. API·구독·로컬 비용, 입력·출력·추론 token과 제한을 구분한다. sampling 변동 감소는 사실성 보장이 아니다.
- **학습/평가:** fine-tuning도 지식을 학습할 수 있지만 특정 문서의 정확한 갱신·회수·삭제·ACL을 weights에 맡기지 않는다. 예시/학습/개선용 입력과 독립 평가를 분리하고 문서 원본·파생 데이터가 양쪽에 새지 않게 한다. 30~100개는 계획 초기값이며 표본 밖 회귀가 없다는 증명이 아니다. judge와 사람 평가의 일치를 확인한다.
- **보안 사례:** 06의 날짜별 사건은 당시 공개 입장과 조치를 읽는 기록이다. 법적 효력·현재 접근권·최종 조사·독립 원인 확정 자료로 취급하지 않는다. 위험한 샘플이나 credential을 실행·수집하지 않는다.

## 미확인

현재 모델별 성능·계정 권한·실제 GPU fit/serving·토큰 비교·RAG/학습 품질·사내 적용·사건 독립 재현/정부 원문·후속 최종 조사·Claude 의견은 미확인이다. 단일 사건으로 모든 최신 모델의 우열을 단정하지 않는다.

[정리 기록](./organization-log.md)
