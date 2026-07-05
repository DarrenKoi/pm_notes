# LangChain 커리큘럼 커버리지 점검

점검 기준: `study_list.txt`의 4개 영역 / 13개 세부 항목.

## 요약

- 전체 항목: 13개
- 생성 문서: 13개
- 누락 문서: 없음
- 보강 완료: Agent API 구분(`create_agent` vs LangGraph `create_react_agent`), 이동된 LangGraph 문서 링크, `study_list.txt` 오탈자/줄바꿈

## 항목별 매핑

| `study_list.txt` 항목 | 생성 문서 | 상태 |
|---|---|---|
| OpenAI API 및 LangChain 구조 이해 | [01](./01-openai-api-and-langchain-structure.md) | 완료 |
| LCEL 실습 | [02](./02-lcel.md) | 완료 |
| Chain, Agent, Tool 정의 및 사용 | [03](./03-chain-agent-tool.md) | 보강 완료 |
| 외부 API 연동 기능 수행 Agent 설계 | [04](./04-external-api-agent.md) | 보강 완료 |
| LangGraph 개요 및 상태 머신 이해 | [05](./05-langgraph-overview-state-machine.md) | 링크 보강 완료 |
| 상태 기반 멀티스텝 대화 흐름 설계 | [06](./06-multistep-conversation-flow.md) | 링크 보강 완료 |
| Condition, Branching, Tool 결합 workflow 실습 | [07](./07-condition-branching-tool-workflow.md) | 링크 보강 완료 |
| 사용자 시나리오 기반 LangGraph Agent 구축 | [08](./08-langgraph-agent-scenario.md) | 보강 완료 |
| 문서 임베딩 및 FAISS 벡터 스토어 구축 | [09](./09-document-embedding-faiss.md) | 완료 |
| Retriever 구성 및 성능 튜닝 | [10](./10-retriever-tuning.md) | 완료 |
| RAG 기반 질의 응답 흐름 구성 | [11](./11-rag-qa-flow.md) | 보강 완료 |
| PDF, 웹 문서, 내부 지식 적용 사례 실습 | [12](./12-rag-document-sources.md) | 완료 |
| 주제 선정 및 요구 사항 정의, 구현, 테스트, 발표 및 피드백 | [13](./13-mini-project.md) | 보강 완료 |

## 남은 개선 후보

- 각 문서의 코드 블록을 실제 환경별로 분리 실행할 수 있도록 `examples/` 폴더를 추가한다.
- Mini Project용 평가셋 템플릿(`eval_set.jsonl`)과 채점 스크립트를 추가한다.
- 사내 엔드포인트 값은 현재 예시 placeholder이므로, 실제 배포 문서에는 환경변수 기반 설정 예제로만 남긴다.
