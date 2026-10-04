---
tags: [fastapi, model-serving, deployment, rest-api]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# FastAPI ML 모델 서빙 가이드

> 저장 모델을 HTTP 입력/응답 계약으로 실행하는 교육용 예제. 파일 로딩·ASGI 요청 검증과 실제 운영 배포를 구분한다.

> [!info] 조립 순서와 실행 범위
> 1절은 lifespan 개념 예시, 2·4·5·6절은 Iris 앱의 서로 다른 파일, 7절은 그 앱의 import/라우터 조립이다. `app/schemas.py`→`app/errors.py`(5절 두 블록을 합침)→`app/iris.py`→`app/batch.py`/`app/health.py`→`app/main.py`를 만든 뒤 학습 파일을 생성하고8절 명령을 사용한다. `app/__init__.py`도 만든다. 3절 torch_serving.py는 별도 앱이며 입력10/output3으로 저장 문서의20/output2 모델과 호환되지 않는다. 실제 CPU TestClient 확인과 socket/Docker/업무 배포는 다르다. 공식 자료 확인일2026-10-04; Claude 협의·Obsidian 읽기 화면·GPU/컨테이너는 미확인이다.

## 왜 필요한가? (Why)

- **시스템 통합**: ML 모델을 학습시킨 것만으로는 실무에 적용할 수 없다. REST API 같은 경계를 제공하면 프론트엔드, MES, 다른 백엔드 서비스에서 예측 결과를 호출할 수 있다.
- **언어 독립성**: Python으로 학습한 모델을 Java, TypeScript 등 다른 언어 기반 시스템에서도 HTTP 요청으로 사용할 수 있다.
- **배포 표준화**: Docker + FastAPI 조합으로 의존성·아티팩트·장치·네트워크 조건을 고정하고 목표 환경에서 검증한 예측 서비스를 배포할 수 있다.
- **FastAPI 선택 이유**: 자동 OpenAPI 문서 생성, Pydantic 기반 요청/응답 검증, async 지원을 제공한다. 추론 스케줄링·batching·GPU 자원/동시성은 별도 설계다.

---

## 핵심 개념 (What)

### FastAPI가 ML 서빙에 적합한 이유

| 특성 | 설명 |
|------|------|
| **Pydantic 스키마** | 선언한 타입/범위/길이만 검증; 모델의 의미/전처리 계약은 직접 정의 |
| **자동 문서화** | `/docs`에서 Swagger UI로 바로 테스트 가능 |
| **비동기(async)** | 비차단 I/O await를 공유; CPU 전처리/후처리가 자동 병렬화되지 않음 |
| **Lifespan 이벤트** | worker 프로세스마다 시작 시 로드; 여러 worker는 모델 메모리도 별도 |
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
- **GPU 추론** (PyTorch, TensorFlow): 동기 호출을 `def`로 옮길 수 있으나 GIL 해제/메모리/stream/thread 안전을 일반 보장하지 않음. GPU 요청 수·batch·worker는 측정으로 정함

> **주의**: `async def` 안에서 CPU 바운드 동기 코드(예: `model.predict()`)를 직접 호출하면 이벤트 루프를 블로킹한다. CPU 바운드 추론은 일반 `def`로 정의하거나 `run_in_executor`를 사용해야 한다.

---

## 어떻게 사용하는가? (How)

### 1. 기본 구조: Lifespan으로 모델 로드

lifespan은 요청 전에 로드하고 종료 때 정리하는 경계다. 아래 load_my_model은 사용자 구현 placeholder라 standalone 실행 예제가 아니다. 실제 Iris 로딩은2절에 한 번 정의하며7절에서 재사용한다. app.state는 같은 앱/worker의 요청 간 상태이고 다른 프로세스와 공유되지 않는다.

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

# load_my_model()은 실제 로더로 교체해야 함


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드, 종료 시 정리"""
    # --- Startup ---
    print("Loading ML models...")
    # 여기서 모델을 로드한다 (아래 섹션에서 구체적 예시)
    app.state.ml_models = {"my_model": load_my_model()}
    print("Models loaded successfully.")

    try:
        yield  # 앱 실행 중
    finally:
        app.state.ml_models.clear()


app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0",
    lifespan=lifespan,
)
```

---

### 2. scikit-learn 모델 서빙

Iris 분류 예시다. `joblib`로 저장된 모델을 로드하여 예측한다.

```python
# app/schemas.py
from pydantic import BaseModel, Field, FiniteFloat, ConfigDict


class PredictionRequest(BaseModel):
    """예측 요청 스키마 - feature 이름과 타입을 명시"""
    sepal_length: FiniteFloat = Field(..., ge=0, le=10, description="꽃받침 길이 (cm)")
    sepal_width: FiniteFloat = Field(..., ge=0, le=10, description="꽃받침 너비 (cm)")
    petal_length: FiniteFloat = Field(..., ge=0, le=10, description="꽃잎 길이 (cm)")
    petal_width: FiniteFloat = Field(..., ge=0, le=10, description="꽃잎 너비 (cm)")

    model_config = ConfigDict(strict=True, extra="forbid", **{
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
    })


class PredictionResponse(BaseModel):
    """예측 응답 스키마"""
    prediction: str = Field(..., description="예측된 클래스")
    confidence: FiniteFloat = Field(..., ge=0, le=1, description="예측 클래스의 모델 점수; 보정 확률 아님")
    model_version: str = Field(..., description="사용된 모델 버전")
```

```python
# app/iris.py - 모델 로딩/단건 예측의 대표 정의
import os
import logging
from pathlib import Path
from time import monotonic
from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import FastAPI, Request
from app.schemas import PredictionRequest, PredictionResponse
from app.errors import PredictionError, require_model

logger = logging.getLogger(__name__)
LABEL_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    default_path = Path(__file__).resolve().parent.parent / "models" / "iris_model.joblib"
    model_path = Path(os.environ.get("MODEL_PATH", str(default_path)))
    model = joblib.load(model_path)  # 신뢰한 파일·동일 의존성/입력 계약
    if getattr(model, "n_features_in_", None) != 4 or list(model.classes_) != [0, 1, 2]:
        raise ValueError("Iris의4 feature/클래스0·1·2 모델 계약과 불일치")
    app.state.ml_models = {"iris": model}
    app.state.model_version = os.environ.get("MODEL_VERSION", "1.0.0")  # 아티팩트와 맞춰 관리
    app.state.started_at = monotonic()
    try:
        yield
    finally:
        app.state.ml_models.clear()

app = FastAPI(title="Iris Prediction API", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, http_request: Request):
    model = require_model(http_request)
    features = np.array([[request.sepal_length, request.sepal_width,
                          request.petal_length, request.petal_width]], dtype=np.float64)
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        index = list(model.classes_).index(prediction)
        return PredictionResponse(prediction=LABEL_MAP[prediction],
                                  confidence=float(probabilities[index]),
                                  model_version=http_request.app.state.model_version)
    except Exception as exc:
        logger.exception("Iris prediction failed")
        raise PredictionError("추론 처리 실패") from exc
```

**모델 저장 참고** (학습 코드): 전체 Iris150개 학습은 API 흐름 fixture이며 독립 test 성능 측정이 아니다. 프로젝트 루트에서 실행한다.

```python
# train.py - 모델 학습 후 저장
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
from pathlib import Path

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/iris_model.joblib")
```

---

### 3. PyTorch 모델 서빙

PyTorch 모델은 state_dict/동일 구조로 로드하고 추론 시 no_grad를 사용한다. 이 조각은 신뢰한 models/classifier.pt(10/64/3 구조)가 준비돼야 한다. API에 연결하기 전 전처리/클래스 의미를 고정한다. softmax는 모델 점수이며 보정된 정확도/확률을 보장하지 않는다. 큰 finite float가 float32에서 inf로 변하는 조건도 거부한다.

```python
# app/torch_serving.py
import torch
import torch.nn as nn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat, ConfigDict
from typing import Annotated
from app.errors import register_error_handlers

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
    model_config = ConfigDict(strict=True, extra="forbid")
    features: list[FiniteFloat] = Field(..., min_length=10, max_length=10, description="10개의 입력 feature")


class TorchPredictionResponse(BaseModel):
    predicted_class: int = Field(ge=0, le=2)
    probabilities: list[Annotated[FiniteFloat, Field(ge=0, le=1)]] = Field(min_length=3, max_length=3)


# --- 앱 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 구조 생성 후 가중치 로드
    model = SimpleClassifier(input_dim=10, hidden_dim=64, output_dim=3)
    model.load_state_dict(torch.load("models/classifier.pt", map_location="cpu", weights_only=True))
    model.to(DEVICE)
    model.eval()  # 평가 모드 (Dropout을 끄고 BN은 저장된 통계 사용)
    app.state.ml_models = {"classifier": model}
    try:
        yield
    finally:
        app.state.ml_models.clear()


app = FastAPI(title="PyTorch Model API", lifespan=lifespan)
register_error_handlers(app)


# CPU/GPU 바운드이므로 일반 def 사용
@app.post("/predict", response_model=TorchPredictionResponse)
def predict(request: TorchPredictionRequest):
    """PyTorch 모델 예측"""
    models = getattr(app.state, "ml_models", {})
    if "classifier" not in models:
        raise HTTPException(status_code=503, detail="모델 준비 안 됨")
    # 입력 전처리
    input_tensor = torch.tensor([request.features], dtype=torch.float32).to(DEVICE)

    if not torch.isfinite(input_tensor).all():
        raise HTTPException(status_code=422, detail="float32로 표현할 수 없는 feature")

    # 추론 (gradient 계산 비활성화 → 메모리/속도 최적화)
    with torch.no_grad():
        logits = models["classifier"](input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    probs = probabilities[0].cpu().numpy().tolist()
    predicted_class = int(np.argmax(probs))

    return TorchPredictionResponse(
        predicted_class=predicted_class,
        probabilities=probs,
    )
```

---

### 4. 배치 예측 (Batch Prediction)

여러 건의 입력을 한 번에 처리하면 네트워크 오버헤드를 줄이고 모델의 벡터 연산 효율을 높일 수 있다.

```python
# app/batch.py
from pydantic import BaseModel, Field, FiniteFloat, ConfigDict
from typing import Annotated
from fastapi import APIRouter, Request
from app.iris import LABEL_MAP
from app.errors import require_model, PredictionError
import numpy as np

router = APIRouter()


IrisRow = Annotated[list[Annotated[FiniteFloat, Field(ge=0, le=10)]], Field(min_length=4, max_length=4)]

class BatchPredictionRequest(BaseModel):
    """배치 예측 요청 - 여러 샘플을 리스트로 전달"""
    model_config = ConfigDict(strict=True, extra="forbid")
    inputs: list[IrisRow] = Field(
        ...,
        min_length=1,
        max_length=1000,  # 한 번에 최대 1000건
        description="2D 배열 형태의 입력 데이터 (samples x features)",
    )


class SingleResult(BaseModel):
    prediction: str
    confidence: FiniteFloat = Field(ge=0, le=1)


class BatchPredictionResponse(BaseModel):
    results: list[SingleResult]
    total_count: int


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, http_request: Request):
    """배치 예측 - 여러 건을 한 번에 처리"""
    model = require_model(http_request)

    features = np.array(request.inputs, dtype=np.float64)

    # 배치 단위로 한 번에 예측 (sklearn은 내부적으로 벡터 연산)
    try:
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)
    except Exception as exc:
        raise PredictionError("배치 추론 처리 실패") from exc

    results = [
        SingleResult(
            prediction=LABEL_MAP[pred],
            confidence=float(prob[classes.index(pred)]),
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
    timeout=10.0,
)
response.raise_for_status()
print(response.json())
# 출력 구조만 예시; 점수0.98/0.95/0.91은 측정값 아님
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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
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
        logger.error("Prediction failed")
        return JSONResponse(
            status_code=500,
            content={"error": "prediction_failed", "detail": "추론 처리에 실패했습니다."},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()],
            },
        )
```

공통 조회 함수(위 errors.py 뒤에 추가): 2·4절 엔드포인트는 이 함수를 사용한다. 원래 중복 /predict 예제는 해당 단건 예측에 합쳤다. RequestValidationError만422로 처리하며 내부 Pydantic/응답 검증 오류를 요청 실수로 바꾸지 않는다. ctx의 exception 객체나 inf 입력을 그대로 JSON에 넣지 않는다. client500 응답에 경로/내부 예외를 보내지 않는다.

```python
# app/errors.py - 위 블록 뒤에 추가; 단건/배치 예측이 함께 사용

def require_model(http_request: Request, name: str = "iris"):
    models = getattr(http_request.app.state, "ml_models", {})
    if name not in models:
        raise ModelNotLoadedError()
    return models[name]
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

/health는 프로세스가 요청에 응답하는 liveness200, /ready는 iris 모델 준비 여부에 따라200/503이다. JSON의 degraded만 바꾸고 항상200을 반환하면 HTTP probe가 실패로 인식하지 않는다. started_at은 startup 후 monotonic 기준이며 wall clock 변경에 영향받지 않는다. models_loaded에 version metadata 키를 섞지 않는다. 이는 모델 품질·외부 의존성·실제 Kubernetes probe 설정 검증은 아니다.

```python
# app/health.py
from time import monotonic
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(tags=["health"])


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
def health_check(request: Request):
    """헬스 체크 - 로드밸런서, K8s liveness probe 용"""
    models = getattr(request.app.state, "ml_models", {})
    started_at = getattr(request.app.state, "started_at", None)
    uptime = monotonic() - started_at if started_at is not None else 0.0

    return HealthResponse(
        status="alive",
        uptime_seconds=round(uptime, 1),
        models_loaded=list(models.keys()),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request):
    """모델 메타데이터 - 어떤 모델이 어떤 버전으로 서빙 중인지 확인"""


    return ModelInfoResponse(
        model_name="iris-classifier",
        model_version=getattr(request.app.state, "model_version", "unknown"),
        framework="scikit-learn",
        input_features=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        description="Iris 품종 분류 모델 (RandomForest)",
    )

@router.get("/ready")
def readiness(request: Request):
    models = getattr(request.app.state, "ml_models", {})
    ready = "iris" in models
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "ready" if ready else "not_ready"})
```

---

### 7. 완전한 프로젝트 구조

위 교육용 Iris 앱과 별도 PyTorch 앱을 조립하는 디렉토리 예시다. tests/와 utils/는 확장 위치를 보여주며 이 문서에 구현 전체가 들어 있다는 뜻은 아니다.

```
ml-api-project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱, lifespan, 라우터 등록
│   ├── iris.py              # 대표 모델 로딩/단건 예측(2절)
│   ├── torch_serving.py      # 별도 PyTorch 앱(3절)
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
# app/main.py - 2절 대표 로딩/단건 예측을 재사용; 중복 lifespan 없음
from app.iris import app
from app.errors import register_error_handlers
from app.health import router as health_router
from app.batch import router as batch_router

register_error_handlers(app)
app.include_router(health_router)
app.include_router(batch_router)
```

---

### 8. 실행 방법

**로컬 실행** (개발):

```bash
# 의존성 설치
pip install fastapi==0.142.2 uvicorn==0.54.0 pydantic==2.13.5 joblib==1.6.0 scikit-learn==1.9.1 numpy==2.5.3
# torch_serving 앱은 별도 호환 torch 설치 필요; 검증 환경은 torch2.14.1
# 실제 프로젝트는 전이 의존성/플랫폼까지 lock하고 같은 학습 환경의 artifact 사용

# 서버 시작 (개발 모드, 자동 리로드)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# API 문서 확인
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
# 별도 PyTorch 앱: 가중치/torch 설치 후 uvicorn app.torch_serving:app --host 127.0.0.1 --port 8001
```

**Docker 배포 예시**: Iris 앱/CPU/worker1 범위이며 torch_serving 의존성은 포함하지 않는다. Docker/실제 socket/보안·인증·TLS·worker/GPU 자원·부하·lock 재현은 미검증이다. MODEL_PATH는2절 로더에서 읽는다. 버전 문자열과 실제 파일의 일치·artifact 검증은 운영 책임이다. 기본 Swagger/ReDoc HTML은 외부 CDN asset을 사용할 수 있어 오프라인/사내 차단 환경에서 실제 화면을 별도로 확인해야 한다.

```dockerfile
# Dockerfile
FROM python:3.14-slim

WORKDIR /app

# Iris 앱만 포함한 예시; 실제 이미지/플랫폼 lock/build는 미검증
RUN pip install --no-cache-dir fastapi==0.142.2 uvicorn==0.54.0 pydantic==2.13.5 joblib==1.6.0 scikit-learn==1.9.1 numpy==2.5.3

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```yaml
# docker-compose.yml
services:
  ml-api:
    build: .
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./models:/app/models:ro  # 교체 파일은 자동 재로딩되지 않음; worker 재시작 필요
    environment:
      - MODEL_PATH=/app/models/iris_model.joblib
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=5)"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**빠른 테스트**:

```bash
# 헬스 체크
curl http://localhost:8000/health
curl http://localhost:8000/ready

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

> **적용 조건**: 본문은 HTTP 입력/모델 파일/ASGI 흐름의 교육용이다. 업무 인증·미들웨어·네트워크/문서 자원 정책은 여기서 검증하지 않았다.

---

## 참고 자료 (References)

2026-10-04 공식 문서 대조. 로컬 CPU 검증 판본: Python3.14.2/FastAPI0.142.2/Pydantic2.13.5/Starlette1.7.0/Uvicorn0.54.0/httpx0.28.1/torch2.14.1/sklearn1.9.1. 로컬 TestClient는 httpx0.28.1로 통과했으나 Starlette1.7.0의 httpx→httpx2 전환 경고가 발생했다. 실제 HTTP client 예제와 TestClient dependency를 혼동하지 않는다. FastAPI 등의 웹 문서는 rolling 문서이며 판본 간 자동 호환을 주장하지 않는다.

- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/): startup/shutdown 경계; worker마다 로드.
- [FastAPI async](https://fastapi.tiangolo.com/async/): def threadpool와 비차단 async I/O의 차이.
- [FastAPI 오류 처리](https://fastapi.tiangolo.com/tutorial/handling-errors/): RequestValidationError와 내부 validation error 구분.
- [FastAPI TestClient lifespan](https://fastapi.tiangolo.com/advanced/testing-events/): with TestClient로 시작/종료 실행.
- [Starlette TestClient](https://starlette.dev/testclient/): 테스트 HTTP client/설치 조건; 이 실행은 httpx2 미설치.
- [FastAPI worker](https://fastapi.tiangolo.com/deployment/server-workers/): 별도 프로세스/모델 메모리 조건.
- [Pydantic config](https://pydantic.dev/docs/validation/latest/api/pydantic/config/): strict/extra 계약.
- [Pydantic types](https://pydantic.dev/docs/validation/latest/api/pydantic/types/): FiniteFloat/길이·범위 검증.
- [Uvicorn settings](https://uvicorn.dev/settings/): reload/workers·bind 조건.
- [FastAPI 오프라인 문서 자원](https://fastapi.tiangolo.com/how-to/custom-docs-ui-assets/): Swagger/ReDoc CDN과 자체 호스팅 조건.
- [PyTorch2.14 Autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html): eval와 no_grad의 독립성.
- [Docker Compose services](https://docs.docker.com/reference/compose-file/services/): healthcheck command/ports/read-only volume.
- [PyTorch 저장/복원](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): 동일 구조/state_dict/eval 로딩.

## 관련 문서

- [모델 저장과 로딩](./model-saving-loading.md): 파일 포맷·입력/의존성 계약.
- [배포 읽기 순서](./README.md): 저장→서빙→실험 기록.
