---
tags: [ai, glossary, literacy, rag, agent, security, token, hallucination, fine-tuning, evaluation]
level: beginner
last_updated: 2026-07-28
---

# AI 용어 및 기술 소개

> AI를 처음 접하는 분도 주요 용어를 업무와 연결해 이해할 수 있도록 만든 입문 자료입니다. 어려운 수식보다는 “왜 필요한가 → 무엇인가 → 어디에 쓰는가 → 무엇을 조심해야 하는가” 순서로 설명합니다.

## 왜 이 자료가 필요한가요?

AI 관련 대화에는 LLM, RAG, 임베딩, 에이전트처럼 서로 연결된 용어가 한꺼번에 등장합니다. 용어를 따로 외우면 뜻은 알아도 전체 흐름을 놓치기 쉽습니다.

이 자료의 목표는 세 가지입니다.

1. AI 용어를 쉬운 한국어로 설명합니다.
2. 비슷해 보이는 개념의 차이를 분명히 합니다.
3. AI 결과를 그대로 믿지 않고 안전하게 활용하는 기준을 제공합니다.

## 권장 학습 순서

| 순서 | 주제 | 핵심 질문 | 문서 |
|---|---|---|---|
| 1 | AI 모델의 종류 | LLM, SLM, VLM, VLA는 무엇이 다른가요? | [01. AI 모델의 종류](./01-ai-model-families.md) |
| 2 | 벡터·임베딩·RAG·온톨로지 | 사내 지식을 어떻게 자산화하고 Agent에 연결하나요? | [02. 벡터, 임베딩, RAG와 온톨로지](./02-vector-embedding-and-rag.md) |
| 3 | 제로샷·퓨샷 | 예시는 언제, 얼마나 제공해야 하나요? | [03. 제로샷과 퓨샷](./03-zero-shot-and-few-shot.md) |
| 4 | 바이브 코딩·하네스·에이전트 | 모델이 어떻게 실제 업무를 수행하나요? | [04. 바이브 코딩에서 하네스까지](./04-vibe-coding-harness-agent-orchestration.md) |
| 5 | 리터러시·편향·슬롭 | AI 결과를 어떻게 판단하고 검증하나요? | [05. AI 리터러시, 편향과 슬롭](./05-ai-literacy-bias-and-slop.md) |
| 6 | 탈옥과 AI 보안 | AI 안전장치는 어떻게 우회되며 무엇을 조심해야 하나요? | [06. 탈옥과 AI 보안](./06-jailbreak-and-ai-security.md) |
| 7 | AX와 AI 데이터 센터 | 개인의 AI 활용이 어떻게 조직의 변화와 인프라로 이어지나요? | [07. AX와 AI 데이터 센터](./07-ax-and-ai-data-centers.md) |
| 8 | 토큰·컨텍스트·환각 | AI는 왜 앞의 대화를 잊고, 왜 매번 다르게 답하나요? | [08. 토큰, 컨텍스트 윈도우, 환각](./08-token-context-window-and-hallucination.md) |
| 9 | 프롬프트·RAG·파인튜닝 | 우리 업무에는 무엇이 필요하고, 잘 되는지 어떻게 아나요? | [09. 프롬프트, RAG, 파인튜닝 중 무엇을 선택할까](./09-prompt-rag-finetuning-and-evaluation.md) |

AI를 처음 접하신다면 01을 읽은 뒤 08을 먼저 보시길 권합니다. 토큰과 컨텍스트 윈도우를 알고 나면 이후 문서의 설명이 훨씬 쉽게 읽힙니다.

처음부터 끝까지 한 문서로 읽으려면 [AI 용어 및 기술 소개 통합본](./all-in-one.md)을 이용하시면 됩니다.

## 빠른 용어 찾기

| 궁금한 내용 | 찾아볼 용어 |
|---|---|
| ChatGPT 같은 서비스의 기본 모델 | LLM, Foundation Model |
| 작고 빠르게 실행하는 언어 모델 | SLM, Distillation |
| 사진이나 화면도 이해하는 모델 | VLM |
| 보고 판단해서 로봇을 움직이는 모델 | VLA |
| 답하기 전에 더 오래 생각하는 모델 | Reasoning Model, Test-time Compute |
| 크지만 일부만 계산에 쓰는 모델 구조 | MoE |
| AI 요금과 길이 제한의 기준 단위 | Token, Context Window |
| 그럴듯하지만 사실이 아닌 답변 | Hallucination |
| 같은 질문에 매번 다른 답이 나오는 이유 | Temperature, 비결정성 |
| 문장이나 이미지의 의미를 숫자로 표현 | Embedding, Vector |
| 내부 문서를 찾아 근거와 함께 답변 | RAG, 지식 자산화 |
| 문서를 검색 단위로 자르고 순위를 다시 매김 | Chunking, Reranker |
| 부서별 용어와 업무 대상의 관계를 정의 | Ontology |
| 예시 없이 또는 몇 개의 예시로 지시 | Zero-shot, One-shot, Few-shot |
| 모델 자체를 우리 업무에 맞게 조정 | Fine-tuning, LoRA |
| AI 품질을 반복 측정하는 시험지 | Golden Set, LLM-as-a-Judge |
| AI가 파일·검색·업무 도구를 사용 | Harness, Tool, AI Agent |
| 도구를 붙이는 공통 연결 규격 | MCP |
| 여러 에이전트와 작업 순서를 조정 | Orchestration |
| 검토 없이 AI 코드를 받아들이는 방식 | Vibe Coding |
| AI를 이해하고 비판적으로 사용하는 능력 | AI Literacy |
| 특정 집단에 불리한 AI 결과 | AI Bias |
| 품질이 낮은 AI 콘텐츠의 대량 생산 | AI Slop |
| AI 안전 규칙을 우회 | Jailbreak |
| 외부 문서가 AI의 지시를 바꿈 | Prompt Injection |
| 최근 AI 보안 사례 | Fable 5, OpenAI 모델과 Hugging Face |
| 조직과 업무 방식을 AI 중심으로 전환 | AX, AI Transformation |
| 대규모 AI 연산을 지원하는 시설 | AI Data Center |

## 읽을 때 기억하실 점

- AI 용어의 경계는 제품과 연구 분야에 따라 조금씩 달라질 수 있습니다.
- 모델 이름이나 제품 기능은 빠르게 바뀌지만, 모델·데이터·도구·검증을 구분하는 기본 원리는 비교적 오래 유지됩니다.
- AI가 자연스럽게 말한다고 해서 내용이 사실이라는 뜻은 아닙니다. 중요한 판단에는 출처 확인과 사람의 검토가 필요합니다.
- 사내 자료와 개인정보는 승인된 시스템과 보안 정책 안에서만 사용해야 합니다.

## 관련 학습 자료

- [Foundation Model / LLM 기초](../foundation%20model/README.md)
- [AI Coding Dictionary](../ai-coding-dictionary/README.md)
- [RAG 학습 자료](../rag/)
- [LangChain / LangGraph 학습 노트](../langchain/README.md)
- [MCP 기초](../mcp/README.md)
