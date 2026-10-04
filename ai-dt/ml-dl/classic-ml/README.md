---
tags: [ml, classical-ml, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
---

# 클래식 ML 읽기 순서

독립 표본의 학습·평가 흐름을 먼저 이해한 뒤 타겟과 비용에 맞는 모델을 선택한다. 이 폴더는 데이터 분할·모델·튜닝·평가의 교육용 예제를 보존한다. 실제 장비/시계열은 split 단위와 운영 타겟을 다시 정의해야 한다.

1. [워크플로우](./ml-workflow-overview.md): 분할·fold·fit·최종 평가. 2026-10-04 개별 검토.
2. [모델 평가](./model-evaluation.md): 지표·곡선·scorer. 2026-10-04 개별 검토.
3. [분류](./classification-recipes.md): 범주형 타겟 모델. 2026-10-04 개별 검토; native 모델 실행/읽기 화면 미확인.
4. [회귀](./regression-recipes.md): 연속 타겟 모델. 2026-10-04 개별 검토; XGBoost 실행 미확인.
5. [군집](./clustering-recipes.md): label 없는 그룹 탐색. 2026-10-04 개별 검토; 읽기 화면 미확인.
6. [튜닝](./hyperparameter-tuning.md): 학습 범위에서 후보를 선택. 2026-10-04 개별 검토; 축소 예산 실행/전체 예산·native 실행·읽기 화면 구분.
7. [정리 기록](./organization-log.md): 근거·변경·검증과 남은 작업.

워크플로우는 실행 순서, 모델 문서는 알고리즘별 선택, 평가 문서는 결과 해석, 튜닝 문서는 선택 절차를 설명한다. 유사 예제의 완전 통합은 다른 문서 검토와 Claude 협의 뒤 결정할 사항으로 보류했다. 원래 6개 문서는 모두 개별 검토했다. 검증 환경의 교육용 실행 결과이며 실제 업무 성능이나 모든 의존성의 실행 성공을 뜻하지 않는다. 분류·군집·튜닝의 Obsidian 읽기 화면 검증은 미완료다.
