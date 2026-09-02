# LLMOps & 평가 커리큘럼 커버리지 점검

점검 기준: `study_list.txt`의 6개 영역 / 15개 세부 항목.

## 요약

- 전체 항목: 15개
- 생성 문서: 15개
- 누락 문서: 없음
- 특이사항: 외부 LLM API 차단 환경에 맞춰 모든 판정(judge)·임베딩 코드를 **사내 OpenAI 호환 엔드포인트**(Kimi-K2.5 / BGE-M3 / Qwen3-VL)로 병기. RAGAS/DeepEval도 사내 모델 주입 방식으로 정리. **Arize Phoenix**를 사내 LLM 관측성·모니터링 도구로 채택, 트레이싱(03)과 모니터링(12)에 반영. 추가 보강으로 release manifest, risk register, incident postmortem까지 운영 문서화.

## 항목별 매핑

| `study_list.txt` 항목 | 생성 문서 |
|---|---|
| LLMOps 개요 및 MLOps와의 차이, 라이프사이클 | [01](./01-llmops-overview-lifecycle.md) |
| 프롬프트 관리 및 버전 관리 | [02](./02-prompt-management-versioning.md) |
| 트레이싱·관측성(Observability) | [03](./03-tracing-observability.md) |
| LLM 평가 개요(패러다임) | [04](./04-llm-evaluation-overview.md) |
| 평가 데이터셋 구축 | [05](./05-eval-dataset-construction.md) |
| 자동 평가 지표 | [06](./06-automatic-metrics.md) |
| LLM-as-a-Judge | [07](./07-llm-as-a-judge.md) |
| RAG 평가(검색·생성 지표) | [08](./08-rag-evaluation.md) |
| Agent·Tool 평가 | [09](./09-agent-tool-evaluation.md) |
| 안전성·환각·가드레일 평가 | [10](./10-safety-hallucination-guardrails.md) |
| 온라인 평가와 배포(CI gate) | [11](./11-online-eval-deployment.md) |
| 모니터링과 드리프트 | [12](./12-monitoring-drift.md) |
| 실전 Mini Project | [13](./13-mini-project.md) |
| 아티팩트 계보와 release governance | [14](./14-artifact-lineage-governance.md) |
| Incident response와 postmortem | [15](./15-incident-response-postmortem.md) |

> 14·15는 `study_list.txt`에 없던 **추가 보강** 문서다. 나머지 01~13은 항목 그대로 대응한다.

## 문서 간 흐름(의존)

- 개념 축: 01(개요) → 04(평가 개요)가 전체를 관통.
- 데이터 축: 05(데이터셋) → 06~10(각 지표)이 공통으로 `eval_set.jsonl` 소비.
- 판정 축: 07(judge) → 08(faithfulness)·09(task success)·10(안전 판정)이 재사용.
- 운영 축: 03(트레이싱) → 11(CI/배포) → 12(모니터링) → 05(피드백 루프로 데이터셋 갱신) 순환.
- 거버넌스 축: 14(release manifest·승인 기준) → 11(배포) → 15(사고대응) → 05/10(평가셋·red team 보강) 순환.

## 남은 개선 후보

- 각 문서 코드 블록을 실제 실행 가능한 `examples/` 패키지로 분리(scorer 모듈화, `run_eval.py` 하네스 1개로 통합).
- `eval_set.jsonl` 샘플 20건과 golden 라벨 예시를 동봉.
- ~~사내 self-host observability(Langfuse 등) 연동 가이드를 03번 부록으로 추가.~~ → **완료**: Arize Phoenix self-host 설정·계측 가이드를 03번에 반영.
- judge 메타평가(인간 라벨 일치도) 실측 결과를 07번에 채워 넣기.
- 실제 사내 release manifest 샘플과 risk register 템플릿을 프로젝트별로 구체화.
- 사고 postmortem 예시 1건을 synthetic trace 기반으로 작성해 15번의 종료 조건을 검증.
- 사내 엔드포인트 값은 현재 placeholder이므로 실제 배포 문서에는 환경변수 기반 설정 예제로만 남긴다.
