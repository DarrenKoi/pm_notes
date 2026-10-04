# qwen3.8-27b 실무 활용 가이드

회사에서 셀프호스팅으로 운용 중인 **Qwen3.8-27B**(소형/중형 dense 모델)의 최대 효율 추출 노하우 정리.

> 작성일: 2026-09-12 · 출처는 각 문서 하단에 명시

## 문서 구성

| 파일 | 내용 |
|---|---|
| [01-model-overview.md](01-model-overview.md) | 모델 스펙, 아키텍처, 벤치마크, 강점/약점 |
| [02-sampling-and-modes.md](02-sampling-and-modes.md) | thinking/instruct 모드 전환과 공식 권장 샘플링 파라미터 |
| [03-serving-setup.md](03-serving-setup.md) | vLLM/SGLang 서빙 설정, 양자화, VRAM/컨텍스트 운영 팁 |
| [04-prompting-playbook.md](04-prompting-playbook.md) | 소형 모델 프롬프팅 베스트 프랙티스 (few-shot, 구조화 출력 등) |
| [05-tool-calling-agentic.md](05-tool-calling-agentic.md) | function calling, 에이전트 워크플로우 운영 노하우 |
| [06-rag-long-context.md](06-rag-long-context.md) | RAG/긴 컨텍스트 설계 — 소형 모델 특유의 함정과 완화책 |
| [07-limits-and-mitigation.md](07-limits-and-mitigation.md) | 알려진 한계(환각, 반복, 열화)와 극복 전략 |
| [08-checklist.md](08-checklist.md) | 운영 체크리스트 (요약본) |

## 핵심 요약 (TL;DR)

1. **모드를 갈아끼워라**: thinking 모드(정확도↑, 느림)와 instruct 모드(빠름)를 태스크별로 구분해 사용. 샘플링 파라미터가 모드마다 다르므로 반드시 구분해서 세팅.
2. **thinking이 기본값이다**: Qwen3.8은 기본이 thinking 모드(`enable_thinking=True`, `reasoning_effort=xhigh`). 루틴한 요청은 끄지 않으면 토큰 비용이 수 배로 늘어난다.
3. **출력 길이를 아낌없이**: reasoning + 최종 응답에 충분한 `max_tokens`를 주지 않으면 조용히 잘려서 품질이 나빠 보인다.
4. **툴콜은 Hermes/XML 파서 조합**: vLLM에서 `--tool-call-parser`를 반드시 명시. thinking 모드에서는 `tool_choice` 강제가 안 되므로 주의.
5. **소형 모델의 RAG는 "컨텍스트 활용"이 병목**: 검색 품질보다 프롬프트 구조(답변 중심 배치, 컨텍스트 압축, 적응형 검색)가 승부처.
6. **긴 대화에서는 thinking 이력을 잘라라** (`preserve_thinking` 제어): 안 그러면 컨텍스트가 급격히 부풀어 오른다.
