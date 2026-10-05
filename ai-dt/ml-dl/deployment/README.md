---
tags: [model-deployment, serialization, experiment-tracking, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
category_major: "AI·DT"
category_middle: "머신러닝·딥러닝"
category_minor: "모델 저장·배포"
note_kind: "목차"
classified_on: "2026-10-05"
---

# 모델 배포 읽기 순서

학습 결과를 어떤 파일·입력 계약으로 보존하는지 확인한 뒤 API 실행과 실험 기록을 읽는다. 직렬화 성공·로컬 추론·실험 추적·업무 배포는 서로 다른 확인 단계다.

1. [모델 저장과 로딩](./model-saving-loading.md): 가중치/재개 상태/교환 그래프·메타데이터. 2026-10-04 개별 검토; CPU roundtrip·ONNX 수치 비교 확인.
2. [FastAPI 모델 서빙](./fastapi-model-serving.md): 저장 모델을 요청/응답 계약으로 실행. 2026-10-04 개별 검토; 실제 CPU TestClient/422·500·503·시작/정리 확인.
3. [실험 추적](./experiment-tracking.md): 설정·지표·아티팩트의 관계. 2026-10-04 개별 검토; 실제 SQL/모델 버전·alias·validation 선택·test 단일 평가 확인.
4. [정리 기록](./organization-log.md): 개별 수정·출처·실행 경계·보류 사항.

> [!warning] 현재 범위
> 원래 3개를 모두 개별 검토했다. 로컬 SQL registry API는 실행했으며 원격 서비스·업무 승인·GPU/장치·판본 변경·장애 복구·Obsidian 읽기 화면·Claude 협의는 미확인이다.
