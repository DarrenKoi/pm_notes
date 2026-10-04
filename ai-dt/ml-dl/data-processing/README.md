---
tags: [ml, data-processing, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
---

# ML 데이터 처리 읽기 순서

모델 학습 전에 원본 파일을 해석하고 분포·결측·수집 조건을 점검한 뒤, **학습 데이터에 맞춘 변환을 평가·운영 입력에 동일하게 적용**하는 과정을 다룬다. 레시피는 교육용이며 데이터 스키마·판본·분할을 결정해야 실행할 수 있다.

1. [공통 적용 조건](./verified-conditions.md): 판본, 예제 의존성, 학습/운영 구분과 검증 경계.
2. [로딩 포맷](./data-loading-formats.md): 파일·인코딩·타입·식별자 보존.
3. [EDA](./eda-recipes.md): 결측·분포·이상치·관계 탐색. 이상치는 삭제 명령이 아니다.
4. [피처 엔지니어링](./feature-engineering.md): 변환의 의미와 모델별 선택 조건.
5. [Pipeline 템플릿](./data-pipeline-template.md): fit/transform·CV·저장 경로를 하나의 흐름으로 구성.
6. [정리 기록](./organization-log.md): 네 원래 문서의 변경·근거·실행 결과·미확인.

로딩은 입력 계약, EDA는 관찰, 피처 문서는 변환 선택, Pipeline은 실행 구성이다. 유사한 예제라도 용도가 달라 고유 코드를 유지했고 공통 주의사항을 대표 안내로 모았다. 파일 이동은 없다.
