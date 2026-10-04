---
tags: [model-deployment, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
---

# 모델 배포 정리 기록

## 범위와 저장/로딩 개별 수정

원래 3개를 모두 전체 읽고 개별 검토했다. 아래 기록은 문서별 검토 이력이다. 원래 작성일2026-02-14와 검토일2026-10-04를 분리했다. 파일 이동·실행 코드·첨부 변경은 없다.

직렬화 비교의 무조건 안전/빠름/작음 단정을 포맷·의존성·장치/EP·자원·측정 조건으로 고쳤다. joblib 기본 compress=0과 압축/mmap 조건, pickle 기반/동일 sklearn 판본·커스텀 객체 경로 조건을 설명했다. Pipeline은 전처리 불일치를 줄이며 보안을 보장한다는 뜻이 아니다. state_dict에 persistent buffer가 포함되고 모델 구조/optimizer는 별도인 점, eval과 no_grad의 차이를 복구했다. 전체 모델 pickle에 소스가 포함돼 클래스 정의가 필요 없다는 오류를 고쳤다.

최소 checkpoint는 실제 모델 config를 저장해20/64/2 hardcode를 제거하고 전체 파라미터를 같은 순서의 단일 group으로 전달한 Adam 재생성만 지원함을 명시하고 다른 group 입력을 저장 전에 거부했다. CPU map_location/weights_only=True를 사용한다. 원래100epoch 루프와 loss=0.5를 placeholder로 분리해 실제 학습/모든 상태/RNG replay·장애 무손실 복구라고 부르지 않는다. saved epoch+1은 유지한다. scheduler/early stopping/RNG/worker/AMP·원자적 저장은 구현하지 않은 범위다.

PyTorch2.14 dynamo=True exporter의 dynamic_shapes/forward 인자명 x를 사용한다. dummy batch2·batch range1~128·opset18·작은 모델 external_data=False를 명시하고 원래 dynamic_axes/opset17 조각과 legacy 조건 차이를 설명했다. ONNX Runtime CPU EP·PyTorch 출력 수치 비교·checker/동등성/안전성의 차이를 추가했다. symbolic dim_param을 출력해 batch를 잘못0으로 표시하던 부분을 고쳤다. batch min/max는 서빙 guard가 아니며 large external data/opset·IR/EP 호환은 별도 검증한다.

metadata는 UTC·framework 판본/실제 config·출력 계약·weights hash를 기록한다. 원래 미측정accuracy0.95/F1=0.93은 원래 예시값이라고 보존 설명하고 JSON 값은 null로 표시했다. hash는 출처 신뢰의 근거가 아니다. .gitignore !규칙의 부모 폴더/JSON 조건을 설명했다. shell/file 생성 pickle payload는 무해한 print callable 시연으로 바꿔 역직렬화 실행 원리를 보존했다. weights_only의 DoS/메모리 제한과 safetensors dense/contiguous·공유 텐서 조건을 명시했다. 가장 안전이라는 절 제목1개를 텐서 데이터 분리로 바꿨다.

공식 근거11건을2026-10-04 대조하고 문서/실행 판본을 분리했다. PyTorch2.14·sklearn1.9.1·joblib1.6.0; 로컬 Python3.14.2/torch2.14.1/onnx1.23.1/onnxscript0.7.2/onnxruntime1.30.0/safetensors0.8.0. safetensors main 문서가 설치 판본의 증거라는 주장은 하지 않는다. DVC/MLflow artifact 역할은 공식 자료와 대조했지만 실제 서비스는 검증하지 않았다.

## Claude 협의

HERDR_ENV=1 현재 pane 조회는 pane_not_found였다. 전용 Claude pane 연결이 불가해 업무 저장 포맷·장애 복구/배포 요구·문서 간 대표 예제 통합 판단은 보류했다. 다른 pane을 제어하거나 Claude의 의견을 만들어 쓰지 않았다. 공식 계약과 직접 재현 가능한 오류 수정은 진행했다.

## 세 단계 검증

1. 원래29절·fence16개(Python14/gitignore1/text1)·작성일·고유 RF/pickle/Pipeline/classifier/state_dict/full-model/checkpoint/ONNX/metadata/security/safetensors 예제 흐름을 대조했다. 가장 안전 절 제목1개만 수정했다. 새 목차/기록을 같은 deployment 폴더에 저장했다. 실행 파일·첨부 이동/수정은 없다.
2. AST14개·실제 CPU 검증9종 통과. RF100trees/1,000행의 joblib/pickle 예측 동일·Scaler/LR Pipeline roundtrip·state_dict와 전체 trusted pickle의 eval logits 동일, 클래스 없는 새 process에서 전체 pickle 로딩 실패, 원래placeholder10checkpoint 및 별도 실제Adam1step/config7-5-3/모델·moment·LR·epoch 복원과 여러 optimizer group 저장 거부/파일 미생성을 확인했다. 실제 dynamo ONNX18/IR10·symbolic batch·checker·CPU ORT batch1/5/128의 수치동등성(rtol1e-4/atol1e-5)·feature21/float64 거부·runtime batch129 허용을 확인했다. single ONNX file·UTC/config/hash/JSON null·무해 pickle print 실행·safetensors 모든 tensor roundtrip도 통과했다. 모든 생성 파일/패키지는 임시 폴더/가상환경 안에만 있다. 신경망은 랜덤/최소 step 예제로 정확도/전이 효과·전체 학습·GPU/RNG/장애 복구의 근거가 아니다.
3. 검토 원문1개+목차/기록2개 총3개 상대 참조·앵커 새 오류0개·unique YAML3개·Obsidian CLI properties3개 검토일 확인을 수행했다. 원래 ../deployment/ 자기 폴더 참조를 새 읽기 목차로 연결했다. ai-dt 비 Markdown18개 초기 해시 동일·diff 공백 검사 통과. 실제 읽기 화면은 앞선 반복 CUA timeout 이후 미완료로 유지한다. 나머지2개까지 검증했다고 기록하지 않는다.

## 남은 미확인

실제 학습 성능·GPU·장치/판본 변경·대형/공유 텐서·custom op·다른 EP·RNG/worker replay·원자적 장애 복구·DVC/MLflow 서비스·업무 저장 승인·Claude 협의·Obsidian 읽기 화면은 미확인이다. 다음 개별 문서는 FastAPI 모델 서빙이다.

## FastAPI 서빙 개별 검토

교육용 HTTP 계약과 실제 socket/Docker/운영 배포를 구분하고 파일 조립 순서를 명시했다. lifespan의 load_my_model은 placeholder로 표시하며 실제 Iris 로딩은2절 app/iris.py 한 곳에 정의해7절 조립에서 재사용한다. 원래 중복 /predict의 오류 처리 문맥은2절 단건 예측에 합치고5절 공통 require_model 조회를 단건/배치에서 사용한다. app.main을 라우터 안에서 다시 import하던 결합을 request.app.state 모델 조회로 복구했다. 원래 학습·Iris·torch·배치·오류·헬스·Docker 예제의 용도는 보존했다.

strict/extra-forbid/FiniteFloat와 Iris feature 범위0~10을 명시하고 배치의 각 행4개·1~1,000행 계약을 적용했다. 단건/배치 점수는 실제 예측 클래스의 classes_ 위치를 사용하며 softmax/모델 점수를 보정 확률이라고 부르지 않는다. RequestValidationError만422로 처리하고 내부/response validation은500으로 유지한다. 직렬화할 수 없는 ctx/비유한 입력을 오류 JSON에 포함하지 않는다. client500 응답에 내부 path/detail을 보내지 않는다. 신뢰한 모델 파일·4 feature/클래스0·1·2 계약·MODEL_PATH/실제 artifact와 version 문자열의 일치 조건을 설명했다. 학습 모델 저장 전에 models/를 만든다. Iris150개 전체 학습은 API fixture이며 독립 성능 측정이 아니다.

Torch는 저장 문서의20/output2와 다른10/64/3 구조를 명시했다. CPU map_location/weights_only=True·eval/no_grad를 구분하고 app.state/cleanup을 적용한다. 길이/finite 및 float32 변환 overflow를422로 거부하고 모델 없는 상태는503을 반환한다. 응답3개 점수/클래스 범위를 선언하고 반올림을 제거해 합계 왜곡을 줄였다. GPU thread/GIL/worker 안전성은 자동 보장하지 않는다.

헬스는 liveness /health200과 특정iris 준비 여부의 /ready200/503을 구분한다. version metadata 키 때문에 모델이 준비됐다고 판정하지 않는다. monotonic uptime·worker마다 별도 메모리 조건을 설명했다. Docker/Compose는 Iris-only worker1·확인된 상위 의존성 예시·Python 기반 /ready probe·읽기 전용 mount를 사용한다. slim 이미지에 없는 curl 의존성과 사용하지 않던 pyproject COPY를 제거했다. MODEL_PATH를 실제 읽으며 파일 교체가 기존 worker를 자동 reload한다는 해석을 제거했다. 외부 CDN 문서 자원·운영 인증/부하/lock 조건은 미확인이다.

공식 자료13건을2026-10-04 대조하고 로컬 Python3.14.2/FastAPI0.142.2/Pydantic2.13.5/Starlette1.7.0/Uvicorn0.54.0/httpx0.28.1/torch2.14.1/sklearn1.9.1 실행 판본을 기록했다. Starlette는 TestClient의 httpx 사용을 지원하되 deprecated/httpx2 전환 안내를 제공하며 실제 경고가 발생했다. 이 실행은 httpx2 검증이 아니다. HTTP client의 timeout과 TestClient의 timeout 경고를 구분했다.

HERDR_ENV=1 현재 pane 조회는 pane_not_found였다. Claude 연결이 불가해 업무 인증/worker/GPU·전체 문서 대표 구조 재설계 판단은 보류했다. 다른 pane을 제어하거나 Claude 의견을 만들지 않았다. 위 통합은 기존 같은 앱 예제의 누락/중복 정의와 직접 확인한 계약 오류를 복구한 범위다.

## FastAPI 세 단계 검증

1. 원래17절/fence17개(Python11/bash2/text2/Dockerfile1/YAML1)·작성일·고유 예제 문맥을 대조했다. 중복 Iris lifespan/단건 오류 예측은 대표 정의를 재사용하고 모든 caller를 갱신했다. 파일 이동·실행 파일·첨부 수정은 없다. 원래 다른 주제로 향하던 존재하지 않는 web-development 참조2개를 제거하고 일반 학습 조건을 보존했다.
2. Python AST11개·실제 CPU 검증9종 통과. 원문 fence로 임시 app 패키지를 조립해 실제 Iris RF100trees/150행 학습·저장·import·TestClient lifespan 공유 모델·단건/배치200와 점수 일치, 누락/extra/string/bool/range/null/NaN/Infinity/잘못된JSON·ragged/feature/1,000초과422, 모델 캐시 삭제/metadata만 존재503 및 liveness200, 고장 모델500의 내부detail 미노출·response validation500을 확인했다. MODEL_PATH/20feature 불일치/없는 파일 startup 실패·version fixture-v2·종료 정리, OpenAPI5개 경로/docsHTML/CDN 주소와 실제 AnyIO worker thread 호출, placeholder loader 주입/정리도 확인했다. 별도 torch10/64/3 랜덤 state 파일의 실제 CPU no_grad/eval·점수3개 합계·모델없는503·길이/type/NaN/float32overflow422·정리를 확인했다. 원문 HTTP client는 실제 ASGI response를 연결해 실행했으며 socket 연결은 아니다. bash2 syntax·YAML1 parse·healthcheck Python AST1·Iris-only Docker worker1 정적 조건을 확인했다. 실제 Docker build/health command/socket/GPU/부하/CDN UI·정확도 측정의 근거는 아니다. 모든 임시 모델/패키지는 저장소 밖에 생성했다.
3. 검토 원문2개+목차/기록2개 총4개 상대 참조·앵커 새 오류0개·unique YAML4개·CLI properties4개 검토일 확인을 수행했다. ai-dt 비 Markdown18개 초기 해시 동일·diff 공백 검사 통과. 실제 Obsidian 읽기 화면은 앞선 반복 CUA timeout으로 미완료다. 추적1개까지 검토했다고 기록하지 않는다.

GPU/thread 동시성·서비스 부하·실제 socket/컨테이너·전체 dependency lock·보정/일반화 성능·운영 인증/네트워크/CDN 화면·artifact/version 실업무 승인·Claude 협의·Obsidian 읽기는 미확인이다. 다음 개별 문서는 실험 추적이다.


## 실험 추적 개별 검토

CSV 평면 표와 JSONL run/artifact 로거는 용도가 달라 각각 유지하고 차이를 설명했다. 같은 본문의 MLflow 설정·자동 기록·수동 기록·조회·registry·독립 end-to-end 예제 순서를 밝혔고 fraud 이름은 원래 API 예시이며 Iris 실습을 실제 사기 탐지로 해석하지 않도록 했다. 가상 accuracy/epoch loss/설정·approve tag를 실제 학습/업무 승인으로 제시하지 않는다. 원래 개념·학습 모델 3개/100·300·200 estimators·CV5·고유 artifact/report/CSV/JSONL 예시는 유지했다.

기본 file backend를 추측하지 않고 local SQLite URI·artifact 경로를 명시했다. model 없이 빈 run을 등록하던 예제를 실제 log_model 반환 model_uri로 연결하고 실제 반환 version을 사용했다. 존재하지 않던 version2는 두 번째 등록으로 생성하며 같은 모델 artifact의 API 시연이라는 조건을 표시했다. Stage deprecated 시작은 2.9.0으로 특정하고 alias/tag가 승인/이미 로딩한 모델 자동 교체를 구현하지 않음을 명시했다. deprecated artifact_path 위치 인자를 name으로 바꿨다. sklearn autolog 공식 지원1.5.2~1.9.0과 로컬1.9.1을 구분해 unsupported 시 비활성화를 명시했다. Lightning 미설치 시 autolog/학습 미실행을 출력하며 일반 optimizer 루프 자동 추적이라고 설명하지 않는다. query는 max_results5이지 모든 run이 아니며 미관측 accuracy의 문자열 포맷 오류를 수정했다.

MLflow3.16.1/skops0.16.0에서 직접 학습한 RF 저장도 Tree 신뢰 확인이 필요해 실행이 한 번 실패했다. 공식 skops 신뢰 조건/API와 설치 구현의 오류를 대조하고 직접 생성한 교육 모델에 한해 sklearn.tree._tree.Tree를 지정했다. 외부 파일의 미신뢰 타입을 자동 일괄 허용하지 않는다. default skops/직렬화 성공이 변조된 tree의 추론/자원/의존성 안전성을 보장하지 않음을 기록했다. 모델 환경 추론에서 pip 판본 미설치 경고가 발생했으므로 완전 환경 lock을 확보했다고 주장하지 않는다.

통합 예제는 341/114/114 train/validation/test로 분리하고 악성 label0의 F1/precision/recall을 명시했다. CV5에서 Scaler를 각 fold train으로 학습하고 동일 Pipeline을 저장/로딩한다. 이번 UUID batch의 FINISHED run만 validation F1로 비교하며 test는 선택 후 한 모델에 한 번만 사용한다. JSON 평가 artifact를 고정 /tmp 이름으로 덮어쓰지 않는다. CSV/JSONL은 비유한/미관측/빈 비교·최소화 방향·태스크 필터를 확인한다. JSONL은 활성 상태·중복 시작·UTF-8·deepcopy·같은 artifact basename 충돌 거부·손상된 JSON line 위치를 처리하되 동시 writer/원자적 장애 복구는 구현하지 않았다.

공식 자료 8건을 2026-10-04 대조했다. 로컬 Python3.14.2/MLflow3.16.1/sklearn1.9.1/skops0.16.0/pandas3.0.3/SQLAlchemy2.1.3으로 실행했다. HERDR_ENV=1 current pane 조회는 pane_not_found였고 전용 Claude pane 연결이 불가했다. 업무 추적/승인·대표 저장 정책·전체 중복 재설계의 협의는 보류했고 공식 계약과 직접 재현한 오류를 수정했다. Claude의 의견이나 연결 성공을 만들지 않았다.

## 실험 추적 및 ML/DL 전체 세 단계 검증

1. 추적 원래18절/fence13개(Python10/bash2/도식1)와 원래 작성일을 대조했다. 배포 원문3개64절/fence46개·ML/DL 원문19개/9,572행이 모두 현재 경로에 존재하고 원래 fence 수가 유지된다. 고유 예제의 용도·합성값·실제 API 선택 조건을 기록했다. 실행 파일·첨부·다른 주제로의 이동은 없다.
2. 추적 AST10개·실제 CPU/SQL 검증6종 통과. CSV 빈 표/미관측/NaN/최소화/태스크 필터/저장·재로딩, autolog 비활성 상태에서 RF 실제 학습, 수동 metric10step/tag/config artifact download, 실제 모델 log/register2versions/alias/load 예측 동일, 세 모델 전체 설정/CV5/341·114·114/Scaler train 평균/validation 선택/current batch/선택 run 하나만 test metric/등록 Pipeline, JSONL lifecycle/NaN·Infinity·bool·문자·null 거부/동일 artifact 충돌 거부/deepcopy/최소화/필터/손상 line 실패를 확인했다. validation F1은 RF2개0.94117647·GB0.94382022이고 선택 GB의 test F1은0.90109890이었다. 이는 교육 fixture의 재현 결과이며 의료·업무 일반화 근거가 아니다. Lightning/MLflow UI/socket/원격 서버/인증·운영 승인은 실행하지 않았다. 모든 DB/artifact/패키지는 저장소 밖 임시 폴더에 생성했다. bash2 syntax 확인. ML/DL 전체 Python AST214개 오류0.
3. 배포 현재5개·ML/DL 전체29개 상대 링크/앵커 새 오류0, unique YAML29개, 확인된 pm_notes vault의 Obsidian CLI properties29개 검토일 확인을 수행했다. 기존 상위/주제 간 참조만 기존 오류로 분리했다. ai-dt 비 Markdown18개 초기 해시 동일. 반복 CUA timeout 이후 실제 Obsidian 읽기 화면은 미완료로 유지한다. CLI 확인을 화면 렌더링 성공으로 해석하지 않는다.

원격 registry/서비스·Lightning 호환/학습·의존성 이관·동시 writer/원자적 복구·업무 승인·Claude 협의·Obsidian 읽기 화면은 미확인이다. ML/DL 개별 문서 검토는19개 모두 완료했으나 외부 및 화면 검증의 한계는 남아 있다.
