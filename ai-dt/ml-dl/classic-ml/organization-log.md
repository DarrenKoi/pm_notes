---
tags: [ml, classical-ml, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
---

# 클래식 ML 정리 기록

## 범위와 개별 결과

원래 문서 6개를 모두 전체 읽고 개별 검토했다. 아래 누적 기록의 문서 수는 당시 검증 범위이며 최종 폴더 범위는 마지막 절에 기록했다. 원래 작성일 2026-02-14는 보존하고 검토일을 분리했다. 파일 이동이나 실행 코드/첨부 변경은 없다.

워크플로우에서 자동 누수 방지·seed 재현·분류 층화 필수·R² 0~1/피처 추가 보장·weighted F1 불균형 해결 단정을 수정했다. ROC-AUC의 decision score 입력과 breast cancer의 0=malignant/1=benign을 구분했다. 수동 fold의 pandas 열/label indexing 오류를 위치 선택으로 고쳤다. 앞 코드에 의존하는 데모와 잘못된 예의 실행 범위를 명시했다. 공식 scikit-learn 1.9.1 자료 7건과 확인일을 문서에 기록했다.

## Claude 협의

HERDR_ENV=1에서 현재 pane 조회를 재시도했으나 pane_not_found였다. 작업 전용 Claude pane에 연결할 수 없어 모델·평가 예제의 대표 통합, 분할 프로토콜의 실제 업무 선택은 보류한다. 다른 pane을 제어하지 않았다. 공식 자료와 직접 재현 가능한 오류의 수정은 진행했다.

## 검증

1. 원래 워크플로우 절 27개·fence 15개를 보존했다. StratifiedKFold 절 제목의 필수 단정만 조건으로 바꿨고 고유 데모를 제거하지 않았다. 원래 날짜를 보존했다.
2. Python AST 13개와 7종 로컬 검증이 통과했다. 실제 내장 breast cancer 569개 표본의 전체 학습/CV/검증/최종 평가 템플릿, 비연속 인덱스 pandas와 NumPy의 수동 CV, 불균형 weighted F1, 음수 R², label 의미, 같은 학습 표본의 중첩 OLS를 검증했다. 실행 판본 Python 3.14.2·scikit-learn 1.9.1·pandas 3.0.3. 95:5 입력의 다수 클래스 예측은 accuracy=0.95/weighted F1=0.925641/관심 클래스 F1=0, R² 반례=-13.5였다. 데모 결과는 임상·설비·운영 성능 증명이 아니다.
3. 검토한 원래 문서와 새 목차/기록 총 3개의 상대 참조·앵커·메타데이터에 오류 0개, CLI properties 3개 검토일 확인이 통과했다. 실제 pm_notes/Obsidian 1.13.7 읽기 화면에서 목차 → 워크플로우 이동과 날짜·한글·표/fence 표시를 확인했다. 폴더 전체 검사에서는 미검토 classification-recipes.md의 검토일 부재로 중단됐으므로 이 결과를 전체 통과로 바꾸지 않았다.

다른 5개 문서의 링크/메타데이터까지 통과했다고 주장하지 않는다. 실제 운영 데이터의 성능·평가 프로토콜·Claude 협의와 문서 간 대표 통합은 미확인이다.

## 모델 평가 개별 검토와 세 단계 검증

RMSE의 삭제된 squared=False 호출을 현재 root_mean_squared_error로 교체했다. 혼동행렬의 실제/예측 축·0/1 순서를 데모와 맞췄고 AP와 사다리꼴 PR-AUC, ROC 순위와 확률 품질, micro/weighted 평균, MAPE/잔차/군집 해석의 조건을 보충했다. 수익 scorer는 list 입력을 지원하고 빈/unknown/비 0·1 label을 거부한다. LogisticRegression >=1.8 grid는 deprecated penalty 대신 l1_ratio로 L1/L2를 선택한다. 잘못된 예시 report는 실제 검증 환경의 출력으로 교체했고 출처 API 7건·확인일·이전 판본 적용 차이를 기록했다. 비용값은 실제 업무 사실로 바꾸지 않았다. 존재하지 않는 cross-validation 링크는 워크플로우로 복구하고 같은 폴더에 없는 feature-engineering 링크는 제거했다.

1. 원래 절 20개·fence 16개·작성일을 보존하고 고유 데모를 제거하지 않았다. API·입력 검증·잘못된 출력 수치 수정은 기존 용도를 유지했다.
2. Python AST 15개, 모든 실제 Python 블록 순차 실행(CV·GridSearch 포함), PNG 8개 생성과 8종 검증이 통과했다. 수익 계산 -150, 잘못된 입력 거부, 혼동행렬, AP=0.583333/사다리꼴 PR-AUC=0.416667 반례, micro=accuracy, 삭제 squared 인자 실패 재현과 현재 RMSE를 확인했다. Python 3.14.2/scikit-learn 1.9.1. 첫 검증 스크립트의 출력 개수 9개 기대를 실제 savefig 8개와 대조해 고치고 전체 재검증했다. Agg show·font cache·physical core 탐지 경고가 발생했고 숨기지 않았다. headless PNG 생성은 사용자 plot 화면·한글 글리프 품질의 증명이 아니다.
3. 현재 검토한 원문 2개+목차/기록 2개만 검사해 상대 링크·앵커·중복 메타데이터 오류 0개, CLI properties 4개와 실제 README → 평가 문서 읽기 이동·한글/표/날짜 확인이 통과했다. 나머지 4개까지 검토했다는 뜻은 아니다.

HERDR_ENV=1 현재 pane 재확인도 pane_not_found였다. Claude 협의가 필요한 대표 지표/비용·문서 통합 선택은 보류한다. 실제 장비/업무 비용, 실데이터 성능·배포 후 모니터링 설계, 한글 plot 화면은 미확인이다.

## 회귀 개별 검토와 세 단계 검증

공통 셋업을 train/validation/test로 나누고 early stopping·비교·잔차 분석은 validation, 선택된 모델만 최종 test에 평가하도록 수정했다. RidgeCV/LassoCV 앞의 train 전체 스케일링이 내부 fold 통계를 섞던 예제는 원본 train의 Pipeline + GridSearchCV로 고쳤고 계수·alpha·피처 선택·경로 목적을 보존했다. Lasso 경로의 절편 차이를 위해 타겟 중심화를 추가했다. XGBoost stopping 인자를 생성자에 넣고 eval metric을 명시했다. 필수 스케일링·속도/성능·자동 과적합 방지·정규 잔차·Q-Q 검정 단정은 적용 조건으로 수정했다. 표의 abs 수식으로 Markdown 열 파손을 복구했다. 원래 상위 링크 1개를 실제 같은 폴더 목차로 바꿨다.

1. 원래 제목 18개·Python fence 10개·작성일을 보존했다. 원래 고유 예제·비교·잔차 흐름은 유지했다. 실제 실행 파일/첨부를 변경하지 않았다.
2. 공식 자료 7건을 대조하고 날짜·문서 판본·설치 판본을 구분했다. Python AST 10개와 scikit-learn 흐름 검증 6종이 통과했다. 실제 블록에 합성 120행 fetch adapter를 주입하고 XGBoost 후보를 제외해, 선형/Ridge/Lasso/GBR 4개 비교·CV·분할 분리·선택한 test 평가·잔차 PNG 총 6개를 확인했다. 테스트 실행에서 n_jobs 두 곳만 1로 제한했다. 실제 California housing 다운로드나 원본 규모·병렬 실행의 성능 증거는 아니다. Python 3.14.2/scikit-learn 1.9.1. XGBoost 공식 문서 표시는 3.4.2였으나 그 버전 설치는 불가했고 PyPI 제공 3.4.1 설치 후 macOS libomp.dylib 부재로 import 실패했다. early stopping·best_iteration·XGBoost 비교 실행은 AST/공식 API 대조만 완료했다. Agg/한글 glyph 경고는 실제 발생했고 plot 한글 화면은 미확인이다.
3. 현재 검토 원문 3개+목차/기록 2개, 상대 참조의 새 오류 0개·기존 다른 주제 참조 2회 유지·CLI properties 5개 확인. 실제 README → 회귀 읽기 화면 이동·날짜/한글/표를 확인했다. Lasso 수식 pipe가 열을 깨는 문제를 발견해 abs 표현으로 고치고 3열/피처 선택 설명 표시를 재검증했다.

HERDR_ENV=1의 현재 pane은 여전히 pane_not_found. 실업무 분할·모델/alpha 최적 선택·완전 대표 예제 통합의 Claude 협의는 보류했다. 실제 데이터 성능·California housing 다운로드·XGBoost/OpenMP·한글 plot·병렬 실행은 미확인이다. 분류·군집·튜닝 3개 문서는 다음 개별 검토 대상이다.

## 분류 개별 검토와 세 단계 검증

train/validation/test를 분리하고 모델별 예측/early stopping·비교를 validation에 제한했다. 비교 뒤 선택을 고정한 모델만 test에 평가한다. XGBoost early_stopping_rounds=50을 생성자에 추가했고 LightGBM subsample_freq=1로 subsample=0.8이 실제 행 bagging을 하도록 수정했다. 로지스틱 >=1.8 L2 규제는 l1_ratio=0, 무작위 category는 Generator seed를 명시했다. 0/1의 두 train 클래스가 없으면 가중치 비율을 계산하지 않으며 label=1을 무조건 minority로 부르지 않는다. class_weight·확률 품질·계수/MDI·category 입력·속도 및 모델 선택 조건을 설명했다. 없는 evaluation-metrics 링크는 실제 model-evaluation.md로 복구했다.

1. 원래 절16개·fence7개·작성일을 보존했다. 공통 준비·별도 category 데모·통합 비교의 용도 차이를 유지했다. 출처/하드웨어/피처/round가 없던 100K 속도 수치는 미측정으로 바꿨다. 원래 제시값은 LR <1초/RF 5~15초/XGB 10~30초/LGBM 3~10초였고 검증 가능한 근거가 없어 현재 성능으로 쓰지 않는다.
2. 공식 자료6건·확인일·표시 판본과 설치 판본을 기록했다. AST6개·로컬 검증9종·PNG1개 통과: 실제 LR/RF 개별 블록과 5,000행 통합 비교, 60/20/20 크기·validation 평가·선택된 test 평가·규제/확률 열·가중비율2/invalid 입력 거부를 확인했다. native import를 제외하고 n_jobs 두 곳만 1로 제한한 테스트이며 전체 네 모델/원래 병렬 속도를 검증한 것은 아니다. Python3.14.2/scikit-learn1.9.1. 설치한 XGBoost3.4.1/LightGBM4.7.0은 둘 다 macOS libomp.dylib 부재로 import 실패. 공식 XGBoost 문서는3.4.2로 표시됐으나 설치 가능 판본과 구분했다. native stopping/category 실행은 AST/공식 근거만 확인했고 미검증이다. Agg 화면 표시 경고가 발생했다.
3. 검토한 원문4개+목차/기록2개 총6개의 상대 참조·앵커·메타데이터 새 오류0개, 기존 다른 주제 참조3회 유지, CLI properties6개 검토일 확인 통과. 분류 문서의 실제 읽기 탐색은 미완료: CUA 화면 조회가124초 timeout, 재접속도34초 timeout/kernel reset. CLI 성공을 실제 화면 렌더링 성공으로 해석하지 않는다. 이전 회귀/평가 화면 확인은 이번 분류 확인을 대신하지 않는다.

HERDR_ENV=1 현재 pane 조회는 pane_not_found로 Claude 협의 불가. 모델 선택/대표 통합·실제 비용/threshold는 보류했다. native 의존성·실데이터 품질·category 운영 계약·확률 보정·속도·분류 읽기 화면은 남은 미확인이다. 군집/튜닝 원문2개는 아직 개별 검토 대기다.

## 군집 개별 검토와 세 단계 검증

TSNE n_iter를 현재 max_iter로 복구했다. DBSCAN k-distance는 자신을 포함한 min_samples=5와 맞췄고 노이즈 제외 silhouette에서 전부 노이즈·한 군집·singleton/빈 입력은 NaN으로 미평가를 보존한다. unknown/길이 불일치 label을 거부하고 유효한 -1 점수와 미평가를 구분한다. KMedoids/평균 중심·거리/단위·스케일 필수·density 차이·DBSCAN 순서/메모리·Agglomerative threshold/Ward·dendrogram lastp·K 후보/Elbow 차이·PCA/t-SNE 표시 해석을 공식 자료7건과 조건으로 수정했다. 실제 업무 군집 선택이나 고장 판정을 만들어 쓰지 않았다.

1. 원래 절15개·fence15개·작성일을 보존했고 고유 예제를 제거하지 않았다. 기존 상위/다른 주제 참조2회는 유지했다.
2. Python AST14개·검증9종이 통과했다. 원래500행 합성 입력과 실제 모든 블록 순차 실행·PNG8개, NaN 예외 입력·유효 점수 sklearn 일치·노이즈 제외 subset 일치·unknown 거부·제거된 n_iter 실패·현재TSNE500x2·자기 포함 다섯째 이웃을 검증했다. Python3.14.2/sklearn1.9.1/SciPy1.18.1. 검증 스크립트가 나중 K-loop로 덮인 k 변수를 검사하던 오류를 고쳐 fitted NearestNeighbors.n_neighbors를 직접 확인하고 전체 재실행했다. Agg/한글 glyph 경고가 실제 발생했으며 plot 한글 품질은 미확인이다. 실제 장비·대용량·새 표본 운영·HDBSCAN/KMedoids 실행을 검증하지 않았다.
3. 검토 원문5개+목차/기록2개 총7개의 상대 참조/앵커/메타데이터 새 오류0개·기존 상위/다른 주제 참조5회 유지, CLI properties7개 확인. CUA Obsidian getApp은 127초 timeout으로 실제 군집 읽기 화면 확인 미완료. CLI와 이전 문서 화면 성공을 이번 군집 화면 검증으로 대신하지 않았다.

HERDR_ENV=1 현재 pane 재조회도 pane_not_found. 실제 군집 목적·거리·대표 예제/중복 통합의 Claude 협의는 보류다. 공식 문서·로컬 계산으로 분명한 수정을 진행했다. 군집의 실업무 의미/속도·미구현 모델·plot한글·Obsidian 읽기 화면은 미확인이다. 남은 튜닝 원문1개를 개별 검토한 뒤 폴더 전체 목록 대조를 수행한다.

KMedoids API 웹 페이지는 조회 불가여서 공식 scikit-learn-contrib GitHub의 main 구현으로 medoid가 실제 표본이라는 계약을 확인하고 근거 링크를 교체했다. 설치 판본/실행은 검증하지 않았다.

## 튜닝 개별 검토와 세 단계 검증

관심 label을 malignant=0으로 고정한 F1 scorer를 탐색과 pruning에 맞췄다. pruning 예제의 early stopping 입력은 각 fold 학습 부분 내부에서 나눠 평가 fold와 분리했다. 학습 안의 CV로 후보를 고른 뒤 최종 test를 사용하는 조건, nested CV와 선택 편향, sampler·seed·병렬 실행·예산의 한계를 설명했다. LightGBM 행 subsampling 조건과 randint의 상한 제외를 명시했다. 비용 표는 108/128 후보와 5-fold의 fit 수를 구분하고 임의의 추천 범위를 최적 성능의 근거로 쓰지 않았다. 문서 내 RF와 뒤 SVC grid의 객체 재정의·단독 native 목적함수·가상 study 시각화의 용도를 구분했다. 공식 자료 7건과 확인일을 남겼다.

1. 원래 절24개·fence12개·작성일을 보존했고 고유한 Grid/Random/Optuna/Pipeline/pruning/시각화 예제를 제거하지 않았다.
2. Python AST11개와 검증 항목7개를 확인했다. 실제 breast cancer 569행에 RF Grid 2후보·GradientBoosting Random 2회·RF/Pipeline TPE 각2trial을 실행했다. 원래 108/128 후보와 50/100회 예산을 실행한 것은 아니다. 별도 toy study에서 실제 MedianPruner의 PRUNED 상태, 8trial study의 Plotly figure4개 직렬화와 Matplotlib figure 생성을 확인했다. native 학습/stopping은 libomp 부재로 실행하지 못해 AST의 입력 분리와 공식 API 대조에 한정했다. Python3.14.2/sklearn1.9.1/Optuna5.0.0/Plotly7.1.0/SciPy1.18.1. Matplotlib Optuna API의 ExperimentalWarning을 확인했으며 브라우저 plot 표시·실제 성능 최적화를 검증하지 않았다.
3. 아래 폴더 전체 검사에서 상대 참조·앵커·메타데이터·CLI를 확인했다. 반복된 CUA timeout 이후 튜닝 읽기 화면은 미완료로 남겼다.

HERDR_ENV=1 현재 pane 조회는 pane_not_found로 Claude 협의 불가였다. 업무 scorer/튜닝 예산·대표 통합 선택은 보류했고 공식 계약·직접 계산으로 확인한 수정만 반영했다.

## 클래식 ML 폴더 전체 대조 결과

원래6개 제목 총120개·fence75개를 대조하고 각 문서의 고유 흐름·작성일을 보존했다. Python AST는 총69개다. 목차와 기록을 포함한 현재8개 문서의 YAML 중복 키/검토일, 상대 링크·앵커 검사는 새 오류0개였다. 원래 상위/다른 주제 참조6회는 유지했으며 새 크로스 링크·이동은 없다. ai-dt의 비 Markdown 자료18개는 초기 해시와 동일했다. pm_notes vault의 CLI properties8개에서 검토일을 확인했다.

실제 읽기 화면은 README→워크플로우/평가/회귀를 확인했고 회귀 표 오류 수정 뒤 재확인했다. 분류·군집·튜닝 읽기는 CUA 연결 timeout으로 미완료다. CLI 확인은 화면 렌더링 검증을 대신하지 않는다. native XGBoost/LightGBM, 원래 전체 탐색 예산·병렬 성능, 외부 데이터 다운로드·실업무 데이터/비용/평가 계약, plot 한글 품질, Claude 협의와 대표 통합은 남은 미확인이다.
