---
tags: [fastapi, model-serving, deployment, rest-api]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# FastAPI ML 모델 서빙 가이드

> 학습된 ML/DL 모델을 FastAPI REST API로 배포하여 다른 시스템에서 예측 결과를 호출할 수 있게 만드는 실무 가이드

## 왜 필요한가? (Why)

- **시스템 통합**: ML 모델을 학습시킨 것만으로는 실무에 적용할 수 없다. REST API로 감싸야 프론트엔드, MES, 다른 백엔드 서비스에서 예측 결과를 호출할 수 있다.
- **언어 독립성**: Python으로 학습한 모델을 Java, TypeScript 등 다른 언어 기반 시스템에서도 HTTP 요청으로 사용할 수 있다.
- **배포 표준화**: Docker + FastAPI 조합으로 어디서든 동일하게 동작하는 예측 서비스를 배포할 수 있다.
- **FastAPI 선택 이유**: 자동 OpenAPI 문서 생성, Pydantic 기반 요청/응답 검증, async 지원으로 ML 서빙에 최적화되어 있다.

---

## 핵심 개념 (What)

### FastAPI가 ML 서빙에 적합한 이유

| 특성 | 설명 |
|------|------|
| **Pydantic 스키마** | 입력 feature 검증을 자동으로 처리 (타입, 범위 등) |
| **자동 문서화** | `/docs`에서 Swagger UI로 바로 테스트 가능 |
| **비동기(async)** | I/O 바운드 작업(전처리, 후처리)에서 높은 동시성 |
| **Lifespan 이벤트** | 서버 시작 시 모델을 한 번만 로드하여 메모리 효율적 |
| **의존성 주입** | 모델 객체를 endpoint에 깔끔하게 전달 |

### 요청/응답 흐름

```
Client → HTTP POST /predict
  → Pydantic 입력 검증
  → 전처리 (numpy 변환 등)
  → model.predict() 또는 model(tensor)
  → 후처리 (라벨 변환 등)
  → Pydantic 응답 직렬화
  → JSON Response
```

### 동기 vs 비동기 추론

- **CPU 바운드 추론** (sklearn, 작은 모델): 일반 `def` 함수 사용 → FastAPI가 threadpool에서 자동 실행
- **I/O 바운드 작업** (외부 API 호출, DB 조회): `async def` 사용
- **GPU 추론** (PyTorch, TensorFlow): 일반 `def` 함수 사용 (GIL 영향 없이 threadpool 활용)

> **주의**: `async def` 안에서 CPU 바운드 동기 코드(예: `model.predict()`)를 직접 호출하면 이벤트 루프를 블로킹한다. CPU 바운드 추론은 일반 `def`로 정의하거나 `run_in_executor`를 사용해야 한다.

---

## 어떻게 사용하는가? (How)

### 1. 기본 구조: Lifespan으로 모델 로드

FastAPI의 lifespan 이벤트를 사용하여 서버 시작 시 모델을 메모리에 로드한다.

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

# 모델을 저장할 딕셔너리 (앱 전체에서 공유)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드, 종료 시 정리"""
    # --- Startup ---
    print("Loading ML models...")
    # 여기서 모델을 로드한다 (아래 섹션에서 구체적 예시)
    ml_models["my_model"] = load_my_model()
    print("Models loaded successfully.")

    yield  # 앱 실행 중

    # --- Shutdown ---
    print("Cleaning up models...")
    ml_models.clear()


app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0",
    lifespan=lifespan,
)
```

---

### 2. scikit-learn 모델 서빙

가장 일반적인 케이스. `joblib`로 저장된 모델을 로드하여 예측한다.

```python
# app/schemas.py
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """예측 요청 스키마 - feature 이름과 타입을 명시"""
    sepal_length: float = Field(..., ge=0, le=10, description="꽃받침 길이 (cm)")
    sepal_width: float = Field(..., ge=0, le=10, description="꽃받침 너비 (cm)")
    petal_length: float = Field(..., ge=0, le=10, description="꽃잎 길이 (cm)")
    petal_width: float = Field(..., ge=0, le=10, description="꽃잎 너비 (cm)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """예측 응답 스키마"""
    prediction: str = Field(..., description="예측된 클래스")
    confidence: float = Field(..., ge=0, le=1, description="예측 확률")
    model_version: str = Field(..., description="사용된 모델 버전")
```

```python
# app/main.py
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse

ml_models = {}
LABEL_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["iris"] = joblib.load("models/iris_model.joblib")
    ml_models["version"] = "1.0.0"
    yield
    ml_models.clear()


app = FastAPI(title="Iris Prediction API", lifespan=lifespan)


# CPU 바운드이므로 일반 def 사용 (FastAPI가 threadpool에서 실행)
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """단건 예측 엔드포인트"""
    # Pydantic 모델 → numpy 배열 변환
    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width,
    ]])

    # 예측 수행
    prediction = ml_models["iris"].predict(features)[0]
    probabilities = ml_models["iris"].predict_proba(features)[0]

    return PredictionResponse(
        prediction=LABEL_MAP[prediction],
        confidence=float(probabilities.max()),
        model_version=ml_models["version"],
    )
```

**모델 저장 참고** (학습 코드):

```python
# train.py - 모델 학습 후 저장
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "models/iris_model.joblib")
```

---

### 3. PyTorch 모델 서빙

PyTorch 모델은 `state_dict` 방식으로 로드하고, 추론 시 `torch.no_grad()`를 사용한다.

```python
# app/torch_serving.py
import torch
import torch.nn as nn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- 모델 정의 (학습 시 사용한 것과 동일해야 함) ---
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- 스키마 ---
class TorchPredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=10, max_length=10, description="10개의 입력 feature")


class TorchPredictionResponse(BaseModel):
    predicted_class: int
    probabilities: list[float]


# --- 앱 ---
ml_models = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 구조 생성 후 가중치 로드
    model = SimpleClassifier(input_dim=10, hidden_dim=64, output_dim=3)
    model.load_state_dict(torch.load("models/classifier.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # 평가 모드 (Dropout, BatchNorm 비활성화)
    ml_models["classifier"] = model
    yield
    ml_models.clear()


app = FastAPI(title="PyTorch Model API", lifespan=lifespan)


# CPU/GPU 바운드이므로 일반 def 사용
@app.post("/predict", response_model=TorchPredictionResponse)
def predict(request: TorchPredictionRequest):
    """PyTorch 모델 예측"""
    # 입력 전처리
    input_tensor = torch.tensor([request.features], dtype=torch.float32).to(DEVICE)

    # 추론 (gradient 계산 비활성화 → 메모리/속도 최적화)
    with torch.no_grad():
        logits = ml_models["classifier"](input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    probs = probabilities[0].cpu().numpy().tolist()
    predicted_class = int(np.argmax(probs))

    return TorchPredictionResponse(
        predicted_class=predicted_class,
        probabilities=[round(p, 4) for p in probs],
    )
```

---

### 4. 배치 예측 (Batch Prediction)

여러 건의 입력을 한 번에 처리하면 네트워크 오버헤드를 줄이고 모델의 벡터 연산 효율을 높일 수 있다.

```python
# app/batch.py
from pydantic import BaseModel, Field
from fastapi import APIRouter
import numpy as np

router = APIRouter()


class BatchPredictionRequest(BaseModel):
    """배치 예측 요청 - 여러 샘플을 리스트로 전달"""
    inputs: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=1000,  # 한 번에 최대 1000건
        description="2D 배열 형태의 입력 데이터 (samples x features)",
    )


class SingleResult(BaseModel):
    prediction: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    results: list[SingleResult]
    total_count: int


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """배치 예측 - 여러 건을 한 번에 처리"""
    from app.main import ml_models, LABEL_MAP

    features = np.array(request.inputs)

    # 배치 단위로 한 번에 예측 (sklearn은 내부적으로 벡터 연산)
    predictions = ml_models["iris"].predict(features)
    probabilities = ml_models["iris"].predict_proba(features)

    results = [
        SingleResult(
            prediction=LABEL_MAP[pred],
            confidence=float(prob.max()),
        )
        for pred, prob in zip(predictions, probabilities)
    ]

    return BatchPredictionResponse(
        results=results,
        total_count=len(results),
    )
```

사용 예시 (클라이언트 측):

```python
import httpx

response = httpx.post(
    "http://localhost:8000/predict/batch",
    json={
        "inputs": [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
            [5.8, 2.7, 4.1, 1.0],
        ]
    },
)
print(response.json())
# {
#   "results": [
#     {"prediction": "setosa", "confidence": 0.98},
#     {"prediction": "virginica", "confidence": 0.95},
#     {"prediction": "versicolor", "confidence": 0.91}
#   ],
#   "total_count": 3
# }
```

---

### 5. 에러 처리

ML API에서 자주 발생하는 에러 유형별 처리 방법.

```python
# app/errors.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


# --- 커스텀 예외 ---
class ModelNotLoadedError(Exception):
    """모델이 아직 로드되지 않았을 때"""
    pass


class PredictionError(Exception):
    """추론 중 에러 발생"""
    def __init__(self, detail: str):
        self.detail = detail


# --- 전역 예외 핸들러 등록 ---
def register_error_handlers(app: FastAPI):

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        return JSONResponse(
            status_code=503,
            content={"error": "model_not_loaded", "detail": "모델이 아직 로드되지 않았습니다. 잠시 후 재시도해주세요."},
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(request: Request, exc: PredictionError):
        logger.error(f"Prediction failed: {exc.detail}")
        return JSONResponse(
            status_code=500,
            content={"error": "prediction_failed", "detail": exc.detail},
        )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": exc.errors(),
            },
        )
```

엔드포인트에서의 사용:

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # 모델 로드 확인
    if "iris" not in ml_models:
        raise ModelNotLoadedError()

    try:
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]])
        prediction = ml_models["iris"].predict(features)[0]
        probabilities = ml_models["iris"].predict_proba(features)[0]
    except Exception as e:
        raise PredictionError(detail=f"추론 중 에러 발생: {str(e)}")

    return PredictionResponse(
        prediction=LABEL_MAP[prediction],
        confidence=float(probabilities.max()),
        model_version=ml_models["version"],
    )
```

**HTTP 상태 코드 가이드**:

| 상태 코드 | 용도 |
|-----------|------|
| `200` | 정상 예측 성공 |
| `422` | 입력 데이터 검증 실패 (Pydantic이 자동 처리) |
| `500` | 모델 추론 중 내부 에러 |
| `503` | 모델 미로드, 서비스 준비 안 됨 |

---

### 6. 헬스 체크 & 메타데이터

운영 환경에서 로드밸런서나 쿠버네티스가 서비스 상태를 확인할 수 있도록 헬스 체크 엔드포인트를 제공한다.

```python
# app/health.py
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])

# 서버 시작 시간 기록
_start_time = datetime.now()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    models_loaded: list[str]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    framework: str
    input_features: list[str]
    description: str


@router.get("/health", response_model=HealthResponse)
def health_check():
    """헬스 체크 - 로드밸런서, K8s liveness probe 용"""
    from app.main import ml_models

    uptime = (datetime.now() - _start_time).total_seconds()

    return HealthResponse(
        status="healthy" if ml_models else "degraded",
        uptime_seconds=round(uptime, 1),
        models_loaded=list(ml_models.keys()),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """모델 메타데이터 - 어떤 모델이 어떤 버전으로 서빙 중인지 확인"""
    from app.main import ml_models

    return ModelInfoResponse(
        model_name="iris-classifier",
        model_version=ml_models.get("version", "unknown"),
        framework="scikit-learn",
        input_features=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        description="Iris 품종 분류 모델 (RandomForest)",
    )
```

---

### 7. 완전한 프로젝트 구조

실무에서 ML API 프로젝트를 구성할 때 권장하는 디렉토리 레이아웃.

```
ml-api-project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱, lifespan, 라우터 등록
│   ├── schemas.py            # Pydantic 요청/응답 스키마
│   ├── errors.py             # 커스텀 예외 및 핸들러
│   ├── health.py             # 헬스 체크, 모델 정보 라우터
│   ├── batch.py              # 배치 예측 라우터
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py  # 입력 전처리 로직
│
├── models/                   # 저장된 모델 파일 (.joblib, .pt 등)
│   ├── iris_model.joblib
│   └── classifier.pt
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # pytest fixture (TestClient 등)
│   └── test_predict.py       # 예측 엔드포인트 테스트
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml            # 의존성 관리 (uv/poetry)
└── README.md
```

`main.py` 통합 예시:

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib

from app.errors import register_error_handlers
from app.health import router as health_router
from app.batch import router as batch_router

ml_models = {}
LABEL_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["iris"] = joblib.load("models/iris_model.joblib")
    ml_models["version"] = "1.0.0"
    yield
    ml_models.clear()


app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0",
    lifespan=lifespan,
)

# 에러 핸들러 등록
register_error_handlers(app)

# 라우터 등록
app.include_router(health_router)
app.include_router(batch_router)


# 메인 예측 엔드포인트는 여기에 정의하거나 별도 라우터로 분리
# (위의 섹션 2 코드 참고)
```

---

### 8. 실행 방법

**로컬 실행** (개발):

```bash
# 의존성 설치
pip install fastapi uvicorn joblib scikit-learn torch numpy

# 서버 시작 (개발 모드, 자동 리로드)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# API 문서 확인
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

**Docker 배포**:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn numpy

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```yaml
# docker-compose.yml
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models  # 모델 파일 마운트 (업데이트 용이)
    environment:
      - MODEL_PATH=/app/models/iris_model.joblib
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**빠른 테스트**:

```bash
# 헬스 체크
curl http://localhost:8000/health

# 단건 예측
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# 배치 예측
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}'
```

---

> **Note**: FastAPI 자체에 대한 더 깊은 학습(의존성 주입, 미들웨어, 인증 등)은 [web-development/python/fastapi/](../../../web-development/python/fastapi/) 폴더를 참고.

---

## 참고 자료 (References)

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)
- [Pydantic V2 문서](https://docs.pydantic.dev/latest/)
- [Uvicorn 배포 가이드](https://www.uvicorn.org/deployment/)
- [PyTorch - Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

## 관련 문서

- [FastAPI 기초](../../../web-development/python/fastapi/) - FastAPI 프레임워크 심화 학습
- [ML/DL 개요](../) - ML/DL 상위 주제
