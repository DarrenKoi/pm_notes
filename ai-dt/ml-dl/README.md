---
tags: [ml, dl, scikit-learn, pytorch, pipeline]
level: beginner-to-advanced
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
---

# ML/DL 실전 가이드

> [!info] 검토 진행 범위
> 데이터 처리 4개 문서는 [해당 목차](./data-processing/README.md)의 결과를 확인하세요. 클래식 ML 6개는 [해당 목차](./classic-ml/README.md)에 개별 검토 결과와 실행·화면 미확인을 기록했습니다. 딥러닝5개는 [해당 목차](./deep-learning/README.md)에 CPU 검증 범위를 기록했습니다. 배포 3개는 [배포 목차](./deployment/README.md)에 CPU·TestClient·SQL/registry 검증 범위를 기록했습니다. 원래 19개 모두 개별 검토했고 전체 29개의 링크·메타데이터·Obsidian CLI 확인을 수행했습니다. 실제 읽기 화면의 일부와 Claude 협의는 미완료입니다. 원래 작성일 2026-02-14와 검토일을 구분합니다.

> 데이터 전처리부터 모델 배포까지, 입력·의존성·평가 조건을 확인하며 실행하는 ML/DL 교육용 레시피 모음

## 왜 필요한가? (Why)

- ML/DL 프로젝트는 매번 비슷한 패턴이 반복된다 (데이터 로드 → 전처리 → 학습 → 평가 → 배포)
- 조건과 검증 범위를 기록한 코드 템플릿을 모아두면 새 프로젝트 시작 시간을 크게 단축할 수 있다
- scikit-learn(클래식 ML)과 PyTorch(딥러닝) 중심으로 실무에서 바로 활용 가능한 레시피를 정리한다

## 학습 로드맵

```
1. 데이터 처리 ──→ 2. 클래식 ML ──→ 3. 딥러닝 ──→ 4. 배포
   (기초)            (핵심)          (심화)        (실전)
```

## 목차

### 📊 데이터 처리 (Data Processing)
- [데이터 로딩 & 포맷](./data-processing/data-loading-formats.md) - CSV, JSON, Parquet, Excel → pandas/polars
- [EDA 레시피](./data-processing/eda-recipes.md) - 탐색적 데이터 분석 패턴 모음
- [피처 엔지니어링](./data-processing/feature-engineering.md) - 인코딩, 스케일링, 변환
- [데이터 파이프라인 템플릿](./data-processing/data-pipeline-template.md) - sklearn Pipeline + ColumnTransformer 재사용 템플릿

### 🤖 클래식 ML (Classic ML)
- [ML 워크플로우 개요](./classic-ml/ml-workflow-overview.md) - Train/Val/Test 분할, 교차 검증, 메트릭 선택
- [분류 레시피](./classic-ml/classification-recipes.md) - LogReg, RandomForest, XGBoost, LightGBM
- [회귀 레시피](./classic-ml/regression-recipes.md) - Linear, Ridge, Lasso, GBR, XGBoost
- [클러스터링 레시피](./classic-ml/clustering-recipes.md) - KMeans, DBSCAN, 계층적 군집화
- [모델 평가](./classic-ml/model-evaluation.md) - 혼동 행렬, ROC-AUC, PR 곡선, 회귀 메트릭
- [하이퍼파라미터 튜닝](./classic-ml/hyperparameter-tuning.md) - GridSearch, RandomSearch, Optuna

### 🧠 딥러닝 (Deep Learning)
- [PyTorch 기초](./deep-learning/pytorch-basics.md) - Tensor, Dataset, DataLoader, GPU 관리
- [학습 루프 템플릿](./deep-learning/training-loop-template.md) - Early stopping, 체크포인팅, 로깅
- [CNN 이미지 분류](./deep-learning/cnn-image-classification.md) - torchvision, 전이학습
- [시퀀스 모델](./deep-learning/sequence-models.md) - RNN/LSTM/GRU, 시계열 예측
- [전이 학습](./deep-learning/transfer-learning.md) - 사전학습 모델 파인튜닝

### 🚀 배포 (Deployment)
- [모델 저장 & 로딩](./deployment/model-saving-loading.md) - joblib, torch.save, ONNX 변환
- [FastAPI 모델 서빙](./deployment/fastapi-model-serving.md) - REST API로 모델 서빙
- [실험 추적](./deployment/experiment-tracking.md) - MLflow, CSV 로깅

## 기술 스택

| 영역 | 라이브러리 |
|------|-----------|
| 데이터 처리 | pandas, polars, numpy |
| 시각화 | matplotlib, seaborn |
| 클래식 ML | scikit-learn, xgboost, lightgbm |
| 딥러닝 | PyTorch, torchvision |
| 배포 | FastAPI, joblib, ONNX |
| 실험 관리 | MLflow, optuna |

## 관련 문서
- [상위: AI/DT 학습 노트](../README.md)
- FastAPI 관련 구현은 다른 독립 주제의 자료이며 이 목차에서 깨진 외부 폴더 참조를 제거했다.
- [RAG 파이프라인](../rag/langgraph/)
